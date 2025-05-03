import numpy as np
import pandas as pd
from numba import jit, njit
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LassoLars

c = 3e18


# function to generate zgrid
def generate_zgrid(zgrid_seps, zgrid_stepsizes, z_fitting_max):
    zgrid = []
    for i in range(len(zgrid_seps)-1):
        zgrid1 = np.arange(zgrid_seps[i], zgrid_seps[i+1], zgrid_stepsizes[i])
        zgrid.extend(zgrid1)
    zgrid.append(z_fitting_max)
    zgrid = np.array(zgrid)
    return zgrid

# function to convert f_nu to f_lambda 
def fnu2flambda(wl, flux):
    flux = flux * c/wl**2
    return flux

# function to convert f_lambda to f_nu
def flambda2fnu(wl, flux):
    flux = flux * wl**2/c
    return flux


# Read the input file and prepare them for dictionary learning
def read_file(pathfile, Ndat, centering=False, error_method=0, SNR=np.inf, f_lambda_mode=True, 
              add_fluctuations=False, flux_fluctuation_scaling=1.0):
    """
    Return: ztrue, lamb_obs, spec_obs, spec_obs_original, err_obs
    """
    data = np.load(pathfile)
    # If the input file denotes whether it's flambda or fnu, read it
    try:
        data_is_flambda = data['save_flambda']
    except:
        data_is_flambda = True     # if input data doesn't contain this parameter, assume input catalog is f_lambda

    ztrue = data['z']
    lamb_obs = data['wavelengths']
    spec_obs = data['spectra']

    if error_method == 0:
        try:
            data['error']
            err_obs = data['error'] * flux_fluctuation_scaling
            # SNR = 1
        except:
            err_obs = np.full(spec_obs.shape, fill_value=np.median(spec_obs))/SNR
    elif error_method == 1:
        err_obs = np.full(spec_obs.shape, fill_value=np.median(spec_obs))/SNR
    elif error_method == 2:
        err_obs = spec_obs/SNR

    # If the dictionary learning code is running in f_lambda mode but input data is f_nu, convert data to f_lambda
    if f_lambda_mode and not data_is_flambda:
        print("Convert input from f_nu to f_lambda")
        spec_obs = fnu2flambda(lamb_obs*10000, spec_obs)
        err_obs = fnu2flambda(lamb_obs*10000, err_obs)
    # if in f_nu mode but data is f_lambda, convert to f_nu
    elif not f_lambda_mode and data_is_flambda:
        print("Convert input from f_lambda to f_nu")
        spec_obs = flambda2fnu(lamb_obs*10000, spec_obs)
        err_obs = flambda2fnu(lamb_obs*10000, err_obs)

    # Read the original spectra without noise
    try:
        spec_obs_original = data['spectra_original']
    except: 
        spec_obs_original = spec_obs.copy()

    if add_fluctuations:
        spec_obs = np.random.normal(spec_obs, err_obs)

    ztrue = ztrue[:Ndat]
    spec_obs = spec_obs[:Ndat]
    spec_obs_original = spec_obs_original[:Ndat]
    err_obs = err_obs[:Ndat]
    if centering:
        spec_obs = spec_obs - np.mean(spec_obs, axis=1)[:,None]
        spec_obs_original = spec_obs_original - np.mean(spec_obs_original, axis=1)[:,None]

    return ztrue, lamb_obs, spec_obs, spec_obs_original, err_obs

# Initialize dictionary
def initialize_dicts(Ndict, dictionary_fluctuation_scaling, 
                     templates_EAZY, num_EAZY_as_dict=1, lamb_rest=np.arange(0.01,6.0,0.01)):
    """
    Return: D_rest
    """
    D_rest_list = []
    # Load EAZY as initialized dictionary
    if num_EAZY_as_dict > 0:
        for i in range(num_EAZY_as_dict):
            D_rest_list.append(templates_EAZY[i]*0.1)
    # template_scale = np.std(D_rest_list[0])
    # Rest of the dictionary to be randomly initialized
    for i in range(Ndict-num_EAZY_as_dict):
        D_rest_list.append(np.random.randn(len(lamb_rest))*dictionary_fluctuation_scaling[i])
    D_rest_list.append(np.ones_like(lamb_rest))
    D_rest = np.vstack(tuple(D_rest_list))
    # note that the last row is a constant DC value to enable the code to fit out the average value
    # D_rest_initial = D_rest.copy()
    return D_rest


# Read SPHEREx filters
def read_filters(filter_location):
    """
    Return: Nfilt, filt_length, filt_norm, filt_all_lams, filt_all_lams_reshaped
    """
    # if convolve_filter:
    filtlams = []  # for storing filter wavelengths
    for i in range(1,103):
        if len(str(i)) == 1:
            idx_str = '00'+str(i)
        elif len(str(i)) == 2:
            idx_str = '0'+str(i)
        elif len(str(i)) == 3:
            idx_str = str(i)
        filt = pd.read_csv(filter_location+f'spherex_paoyu_{idx_str}.txt', sep='\s+')
        lams_i = filt['lambda']
        filtlams.append(lams_i/10000) # convert to um
    ftrans = filt['transmission'].to_numpy()   # All transmission looks the same, so just take the last one
    filtlams = np.array(filtlams)
    filt_norm = np.zeros(len(filtlams))

    # Calculate filter normalization
    # Also save all filters' wavelength points in a single 1-dimensional array
    filt_all_lams = []
    for i in range(len(filtlams)):
        lb = filtlams[i]
        filt_norm[i] = lb[-3]-lb[2]
        filt_all_lams.extend(lb[2:-2])
    filt_all_lams = np.array(filt_all_lams)
    filt_length = len(lb[2:-2])  # each filters length
    Nfilt = len(filtlams)
    filt_all_lams_reshaped = filt_all_lams.reshape((Nfilt, filt_length))

    return Nfilt, filt_length, filt_norm, filt_all_lams, filt_all_lams_reshaped


# Load EAZY templates
def load_EAZY(lamb_um, eazy_templates_location):
    dfs = []
    for i in range(1,8):
        dfi = pd.read_csv(eazy_templates_location+f'eazy_v1.1_sed{i}.dat', names=['lambda_ang','flux'], sep='\s+')
        # if i != 6:
        dfs.append(np.interp(lamb_um*10000, dfi['lambda_ang'], dfi['flux']/np.std(dfi['flux'])))
        # else:
        # dfs.append(np.interp(lamb_um*10000, dfi['lambda_ang'], -1.0*dfi['flux']/np.std(dfi['flux'])))
    # templates_EAZY = np.vstack([dfs, np.ones_like(lamb_um)])
    templates_EAZY = dfs
    return templates_EAZY


# function to integrate fluxes convolved with filters and return all 102 values
@njit(fastmath=True)
def f_convolve_filter(wl, flux, filter_infos):
    Nfilt, filt_length, filt_norm, filt_all_lams, filt_all_lams_reshaped = filter_infos

    # flux_conv = np.zeros(102)
    # for i in range(102):
    #     lb = filtlams[i]
    #     f_interp = np.interp(lb, wl, flux)
    #     fnu_i = np.trapz(f_interp*ftrans, lb)/filt_norm[i]
    #     flux_conv[i] = fnu_i

    # manually calculate area below curves
    f_interps = np.interp(filt_all_lams, wl, flux).reshape((Nfilt, filt_length))
    # flux_conv = np.sum(f_interps, axis=1) / filt_length   # Numba doesn't support np.mean with arguments
    df_interps = np.diff(f_interps)
    dl = np.diff(filt_all_lams_reshaped)
    f_ints = dl * (0.5*df_interps + f_interps[:,:-1])
    flux_conv = np.sum(f_ints, axis=1)/filt_norm
    return flux_conv


# utility function to apply a redshift to spectral dictionary
@jit(nopython=True,fastmath=True)
def apply_redshift(D,z,lamb_in,lamb_out):
    # initialize output dictionary
    D_out = np.zeros((D.shape[0],len(lamb_out)))
    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in*(1+z)
    for i in range(D.shape[0]):
        D_out[i,:] = np.interp(lamb_out,lamb_inx1pz,D[i,:])
    return D_out

# apply redshift function with additional option to convolve with filters
@jit(nopython=True,fastmath=True)
def apply_redshift1(D,z,lamb_in,lamb_out, filter_infos, conv=False):
    # initialize output dictionary
    D_out = np.zeros((D.shape[0],len(lamb_out)))
    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in*(1+z)
    for i in range(D.shape[0]):
        if conv:
            D_conv = f_convolve_filter(lamb_inx1pz, D[i,:], filter_infos)
            D_out[i,:] = D_conv
        else:
            D_out[i,:] = np.interp(lamb_out,lamb_inx1pz,D[i,:])
    return D_out



# Adaptive grid search with increasing step size toward higher z
@jit(nopython=True,fastmath=True)
def fit_spectrum(lamb_data, spec_data, err_data, lamb_D, D, zgrid, filter_infos, lassolars=False, alpha=0, lars_positive=False, alpha_ns_scaling=False,
                 zgrid_searchsize=0.02, zgrid_errsearchsize=0.03, z_fitting_max=2.0, probline=0.317/2, 
                 zinput=False, conv_first=False, conv_last=False, error=False, second_stage=True):

    if not zinput:
        # consider each redshift from 0-2
        ztrial0 = zgrid.copy()
        # calculate residual at each redshift
        residual_vs_z0 = np.inf + np.zeros_like(ztrial0) # initialize to infinity
        # loop over trial redshifts
        for k in range(ztrial0.shape[0]):
            # make this redshifted template
            D_thisz = apply_redshift1(D,ztrial0[k],lamb_D,lamb_data, filter_infos, conv=conv_first)
            # D_thisz = D_thisz/err_data
            # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
            # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
            if not lassolars:   # use OLS fitting
                params, model = fit_models_ols(D_thisz, spec_data, err_data)
            else:
                params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, positive=lars_positive, unit_X=True, 
                                                    unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling)

            # calculate the RMS residual
            residual_vs_z0[k] = np.sum((model - spec_data)**2/err_data**2)
        
        # find the trial redshift with the lowest residual
        kbest = int(np.where(residual_vs_z0 == np.min(residual_vs_z0))[0][0])
        residues0 = residual_vs_z0.copy()    # save this residue for error estimation later
        # note the redshift with the lowest residual
        z = ztrial0[kbest]
        # if kbest > 0:
        #     zmin = ztrial0[kbest-1]

        # create second round ztrial and residues regardless for output format purpose
        zmin = z - zgrid_searchsize
        zmax = z + zgrid_searchsize
        if zmin < 0:
            zmin = 0.0
        if zmax > z_fitting_max:
            zmax = z_fitting_max
        ztrial = np.arange(zmin, zmax, 0.001)
        residual_vs_z = np.inf + np.zeros_like(ztrial) # initialize to infinity
        # second round
        if second_stage:
            # calculate residual at each redshift
            # loop over trial redshifts
            for k in range(ztrial.shape[0]):
                # make this redshifted template
                D_thisz = apply_redshift1(D,ztrial[k],lamb_D,lamb_data, filter_infos, conv=conv_last)
                # D_thisz = D_thisz/err_data
                # params = inv(D*D')*D*s'
                # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
                # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))

                if not lassolars:   # use OLS fitting
                    params, model = fit_models_ols(D_thisz, spec_data, err_data)
                else:
                    params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, positive=lars_positive, unit_X=True, 
                                                        unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling)
                # calculate the RMS residual
                residual_vs_z[k] = np.sum((model - spec_data)**2/err_data**2)
            
            # find the trial redshift with the lowest residual
            kbest = int(np.where(residual_vs_z == np.min(residual_vs_z))[0][0])
            # note the redshift with the lowest residual
            z = ztrial[kbest]        

    else:
        z = zinput
        error = False
    # redo the fit at this redshift
    # make this redshifted template
    D_thisz = apply_redshift1(D,z,lamb_D,lamb_data, filter_infos, conv=conv_last)
    # D_thisz = D_thisz/err_data
    # fit the data to this template
    # params = inv(D*D')*D*s'
    # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
    # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))

    if not lassolars:   # use OLS fitting
        params, model = fit_models_ols(D_thisz, spec_data, err_data)
    else:
        params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, positive=lars_positive, unit_X=True, 
                                            unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling)
    # calculate the model for these parameters and this template
    # model = np.zeros_like(lamb_data)
    # for i in range(D.shape[0]):
        # model += params[i]*D_thisz[i,:]
    # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
    model0 = model.copy()
    if zinput:
        residues0 = np.array([np.sum((model - spec_data)**2/err_data**2)])
        residual_vs_z = np.array([np.sum((model - spec_data)**2/err_data**2)])

    params0 = params.copy()
    if not error:
        return z,0.0,0.0,params0,model0,(ztrial0,ztrial),(residues0,residual_vs_z)

    else:
        min_res = np.min(residues0)
        prob0 = np.exp(-(residues0-min_res)/2)
        # calculate integration and normalize prob0
        dz0 = np.diff(ztrial0)
        dprob0 = np.diff(prob0)
        d_area0 = dz0 * (prob0[:-1] + 0.5*dprob0)
        total_prob0 = np.sum(d_area0)
        prob0 = prob0/total_prob0
        d_area0 = d_area0/total_prob0

        c_d_area0 = np.cumsum(d_area0)  # cumulative area from z=0
        # zlow_0 = ztrial0[np.argmin(np.abs(c_d_area0 - probline))]
        zlow_0 = ztrial0[np.argwhere((c_d_area0 - probline)>0)[0][0]]
        # reverse cumulative area
        ztrial0_r = ztrial0[::-1]
        c_d_area0_r = np.cumsum(d_area0[::-1])
        # zhigh_0 = ztrial0_r[np.argmin(np.abs(c_d_area0_r - probline))+1]    # because it is reverse the index had to be added by 1
        zhigh_0 = ztrial0_r[np.argwhere((c_d_area0_r - probline)>0)[0][0]-1]

        # second round with more precision
        zlow_zmin = zlow_0 - zgrid_errsearchsize
        zlow_zmax = zlow_0 + zgrid_errsearchsize
        if zlow_zmin < 0:
            zlow_zmin = 0.0
        if zlow_zmax > z_fitting_max:
            zlow_zmax = z_fitting_max
        ztrial1_low = np.arange(zlow_zmin, zlow_zmax, 0.001)
        residual_vs_z_low = np.inf + np.zeros_like(ztrial1_low)
        for k in range(ztrial1_low.shape[0]):
            # make this redshifted template
            D_thisz = apply_redshift1(D,ztrial1_low[k],lamb_D,lamb_data, filter_infos, conv=conv_last)
            # D_thisz = D_thisz/err_data

            # params = inv(D*D')*D*s'
            # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
            # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
            if not lassolars:   # use OLS fitting
                params, model = fit_models_ols(D_thisz, spec_data, err_data)
            else:
                params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, positive=lars_positive, unit_X=True, 
                                                    unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling)
            # calculate the model from these parameters and this template
            # model = np.zeros_like(lamb_data)
            # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))

            # calculate the RMS residual
            residual_vs_z_low[k] = np.sum((model - spec_data)**2/err_data**2)

        zhigh_zmin = zhigh_0 - zgrid_errsearchsize
        zhigh_zmax = zhigh_0 + zgrid_errsearchsize
        if zhigh_zmin < 0:
            zhigh_zmin = 0.0
        if zhigh_zmax > z_fitting_max:
            zhigh_zmax = z_fitting_max
        ztrial1_high = np.arange(zhigh_zmin, zhigh_zmax, 0.001)
        residual_vs_z_high = np.inf + np.zeros_like(ztrial1_high)
        for k in range(ztrial1_high.shape[0]):
            # make this redshifted template
            D_thisz = apply_redshift1(D,ztrial1_high[k],lamb_D,lamb_data, filter_infos, conv=conv_last)
            # D_thisz = D_thisz/err_data

            # params = inv(D*D')*D*s'
            # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
            # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
            if not lassolars:   # use OLS fitting
                params, model = fit_models_ols(D_thisz, spec_data, err_data)
            else:
                params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, positive=lars_positive, unit_X=True, 
                                                    unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling)                
            # calculate the model from these parameters and this template
            # model = np.zeros_like(lamb_data)
            # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))

            # calculate the RMS residual
            residual_vs_z_high[k] = np.sum((model - spec_data)**2/err_data**2)

        # combine residues[0] with two more residue arrays
        ztrial1_unsorted = np.concatenate((ztrial0, ztrial1_low, ztrial1_high))

        sort_idx = np.argsort(ztrial1_unsorted)
        ztrial1 = ztrial1_unsorted[sort_idx]
        residues1 = np.concatenate((residues0, residual_vs_z_low, residual_vs_z_high))
        min_res = np.min(residues1)
        prob1 = np.exp(-(residues1[sort_idx]-min_res)/2)
        dz1 = np.diff(ztrial1)
        dprob1 = np.diff(prob1)
        d_area1 = dz1 * (prob1[:-1] + 0.5*dprob1)
        total_prob1 = np.sum(d_area1)
        # total_prob1 = np.trapz(prob1, x=ztrial1)
        prob1 = prob1/total_prob1
        d_area1 = d_area1/total_prob1

        c_d_area1 = np.cumsum(d_area1)
        # zlow_1 = ztrial1[np.argmin(np.abs(c_d_area1 - probline))]
        zlow_1 = ztrial1[np.argwhere((c_d_area1 - probline)>0)[0][0]]

        # reverse cumulative area
        ztrial1_r = ztrial1[::-1]
        c_d_area1_r = np.cumsum(d_area1[::-1])
        # zhigh_1 = ztrial1_r[np.argmin(np.abs(c_d_area1_r - probline))+1]
        zhigh_1 = ztrial1_r[np.argwhere((c_d_area1_r - probline)>0)[0][0]-1]

        return z,zlow_1,zhigh_1,params0,model0,(ztrial0,ztrial),(residues0,residual_vs_z)
    

@jit(nopython=True,fastmath=True)
def fit_models_ols(D_thisz, spec_data, err_data):
    spec_data_reshaped = np.reshape(spec_data/err_data,(len(spec_data/err_data),1))
    D_thisz = D_thisz/err_data
    # params = inv(D*D')*D*s'
    params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped
    model = ((D_thisz*err_data).T @ params).reshape(len(err_data))
    # calculate the model from these parameters and this template
    # model = np.zeros_like(lamb_data)
    # for i in range(D.shape[0]):
        # model += params[i]*D_thisz[i,:]
    return params, model

@jit(nopython=True,fastmath=True)
def fit_model_larslasso(D_thisz, spec_data, err_data, alpha=0.0, positive=False, unit_X=True, unit_y=True, 
                        max_iter=200, decimals=10, alpha_ns_scaling=False):
    X = (D_thisz/err_data).T    # currently D_thisz is X.T in usual LARSlasso convention
    X = np.ascontiguousarray(X)
    y = spec_data/err_data
    params = _larslasso(X, y, alpha=alpha, positive=positive, unit_X=unit_X, unit_y=unit_y, 
                        max_iter=max_iter, decimals=decimals, alpha_ns_scaling=alpha_ns_scaling)
    params = np.reshape(params, (D_thisz.shape[0], 1))
    model = ((D_thisz).T @ params).reshape(len(err_data))
    return params, model

@jit(nopython=True,fastmath=True)
def _larslasso(X, y, alpha=0.0, positive=False, unit_X=True, unit_y=True, centering_X=False, centering_y=False, 
                 intercept=False, max_iter=200, decimals=14, alpha_ns_scaling=False):
    '''
    Return LARS-lasso fitting coefficients. X is the atoms array, y is the target vector.

    Parameters
    ----------
    X : ndarray of shape (ns, nf)
        Atom array with ns samples and nf features

    y : ndarray of shaoe (ns, )
        Target vector with ns sample points to fit

    alpha : float, default=0.0
        Regularization parameter for LARS-lasso algorithm. alpha>=0. Alpha=0 will yield OLS solution.
        The actual mathematical meaning of this parameter is the lowest required dot product value between atoms and target.
    
    positive : bool, default=False
        If True, force all coefficients to be positive
    
    unit_X : bool, default=True
        If False, do not normalize the input atoms (along axis=0)
    
    unit_y : bool, default=True
        If False, do not normalize the input target vector
    
    centering_X : bool, default=False
        If True, center all the atoms in X (no way to recover intercept for now)

    centering_y : bool, default=False
        If True, center the target vector y (no way to recover intercept now)
    
    intercept : bool, default=False
        If True, return intercept (not implemented at the moment)
    
    max_iter : int, default=200
        Max allowed iteration for the LARS-lasso algorithm

    decimals : int, default=14
        Number of decimals to round coefficients for stability

    alpha_ns_scaling: bool, default=False
        If True, alpha_i in each step will be alpha_i/ns 
        This is so that alpha is more consistent with lasso L1-regularization definition.

    Returns
    -------
    coef : ndarray of shape (nf, )
        Fitted coefficients

    '''

    nf = X.shape[1]
    ns = X.shape[0]
    # Xmeans = np.sum(X, axis=0)/X.shape[0]    # numba doesn't support np.mean(X, axis=0)
    # ymean = np.mean(y)
    # Xnorms = np.sqrt(np.sum(X**2, axis=0))  # numba doesn't support np.linalg.norm with second argument
    # ynorm = np.linalg.norm(y)
    Xnorms = np.ones(nf)
    ynorm = 1.0
    # if centering_X:
    #     X = X - Xmeans
    # if centering_y:
    #     y = y - ymean
    if unit_X:
        Xnorms = np.sqrt(np.sum(X**2, axis=0))  # norm need to be re-calculated if centered
        X = X/Xnorms
    if unit_y:
        ynorm = np.linalg.norm(y)
        y = y/ynorm

    Cov = X.T @ y
    G = X.T @ X

    coef = np.zeros(nf)
    prev_coef = np.zeros(nf)
    alphas = np.array([0.0])
    prev_alphas = np.array([0.0])
    # coef_paths = [coef]

    n_iter = 0
    nA = 0
    indices = np.arange(nf) # index number of atoms, always track the true index number with this
    idx_A_tf = np.full(nf, False)   # active atoms selection mask
    idx_rest_tf = np.full(nf, True) # non-active atoms mask
    sign_active = np.zeros(nf)      # active atoms will have 1 or -1

    drop = False
    Cov_copy = Cov.copy()
    G_copy = G.copy()

    while True:
        if Cov[idx_rest_tf].size:
            if positive:
                c_idx_arg = np.argmax(Cov[idx_rest_tf])              # index number for current Cov selection
            else:
                c_idx_arg = np.argmax(np.fabs(Cov[idx_rest_tf]))     # index number for current Cov selection
            c_idx = indices[idx_rest_tf][c_idx_arg]             # find the true index number
            c_ = np.max(np.fabs(Cov[idx_rest_tf]))
            c_ = Cov[c_idx]
            if positive:
                c = c_
            else:
                c = np.fabs(c_)
        else:
            c = 0.0

        if alpha_ns_scaling:
            alphas[0] = c/ns
        else:
            alphas[0] = c

        if alphas < alpha and n_iter > 0:
            # print('End early')
            ss = (prev_alphas - alpha) / (prev_alphas - alphas)
            coef = prev_coef + ss * (coef - prev_coef)
            alphas[0] = alpha
            break
        if n_iter >= max_iter or nA >= nf:
            # print('Stopping criteria met')
            break

        if not drop:
            if positive:
                sign_active[c_idx] = np.ones_like(c_)
            else:
                sign_active[c_idx] = np.sign(c_)
            idx_A_tf[c_idx] = True
            idx_rest_tf[c_idx] = False
            nA += 1

        Cov_rest = Cov[idx_rest_tf]
        XA = X[:, idx_A_tf]
        Xr = X[:, idx_rest_tf]
        GA = XA.T @ XA
        one_A = sign_active[idx_A_tf]

        AA = 1/np.sqrt(one_A.T @ np.linalg.inv(GA) @ one_A)
        wA = AA * np.linalg.inv(GA) @ one_A
        uA = XA @ wA
        corr_eq_dir = Xr.T @ uA

        g1 = (c-Cov_rest)/(AA-corr_eq_dir)
        g1[g1<=0] = np.inf
        if g1.size == 0:
            g1 = np.array([np.inf])
        if positive:
            g = min(min(g1[g1>0]), c/AA)
        else:
            g2 = (c+Cov_rest)/(AA+corr_eq_dir)
            # print(g2)
            g2[g2<=0] = np.inf
            if g2.size == 0:
                g2 = np.array([np.inf])
            g = min(min(g1[g1>0]), min(g2[g2>0]), c/AA)

        drop = False
        z = -coef[idx_A_tf]/wA
        if z[z>0].size>0 and np.min(z[z>0]) < g:
            # print('Drop!')
            drop = True
            min_pos_z = np.min(z[z>0])
            idx_drop1 = np.where(z==min_pos_z)[0]
            idx_drop = indices[idx_A_tf][idx_drop1]
            g = min_pos_z

        n_iter += 1
        prev_coef = coef
        prev_alphas[0] = alphas[0]
        coef = np.zeros_like(coef)
        coef[idx_A_tf] = prev_coef[idx_A_tf] + g * wA
        Cov[idx_rest_tf] = Cov[idx_rest_tf] - g * corr_eq_dir
        # coef_paths.append(coef)

        if drop:
            nA -= 1            
            temp = Cov_copy[idx_drop] - G_copy[idx_drop] @ coef
            Cov[idx_drop] = temp
            sign_active[idx_drop] = 0
            idx_A_tf[idx_drop] = False
            idx_rest_tf[idx_drop] = True

    coef = np.around(coef, decimals=decimals)
    # if intercept:
    #     b = ymean - Xmeans @ coef
    # else:
    #     b = 0.0
    # model = X @ coef + b

    # return coef, coef_paths
    return coef / Xnorms * ynorm
    # return coef




# Adaptive grid search with increasing step size toward higher z
# testing LarsLasso with scikit-learn, SLOW!
# @jit(nopython=True,fastmath=True)
def fit_spectrum1(lamb_data, spec_data, err_data, lamb_D, D, zgrid, filter_infos, alpha, lars_positive, NMF=False, NMF_tolerance=1e-3, NMF_cutoff=20000,
                 zgrid_searchsize=0.02, zgrid_errsearchsize=0.03, z_fitting_max=2.0, probline=0.317/2, 
                 zinput=False, conv_first=False, conv_last=False, error=False, second_stage=True):
    # reshape array in preparation for later calculation
    spec_data_reshaped = np.reshape(spec_data/err_data,(len(spec_data/err_data),1))
    spec_data_wt = spec_data/err_data
    lassolars1 = LassoLars(alpha=alpha, positive=lars_positive)
    ztrial0 = zgrid.copy()
    ztrial = np.arange(0, 0.1, 0.001)

    if not zinput:
        # consider each redshift from 0-2
        ztrial0 = zgrid.copy()
        # calculate residual at each redshift
        residual_vs_z0 = np.inf + np.zeros_like(ztrial0) # initialize to infinity
        # loop over trial redshifts
        for k in range(ztrial0.shape[0]):
            # make this redshifted template
            D_thisz = apply_redshift1(D,ztrial0[k],lamb_D,lamb_data, filter_infos, conv=conv_first)

            # params = inv(D*D')*D*s'
            if not NMF:
                D_thisz = D_thisz/err_data
                # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
                # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
                lassolars1.fit(D_thisz.T, spec_data_wt)
                coef1 = lassolars1.coef_
                params = coef1[:,None]
                model = (D_thisz.T @ coef1 + lassolars1.intercept_) * err_data

            else:
                T = D_thisz/err_data**2
                TT = T @ T.T
                b = T @ (spec_data/err_data**2)
                c0 = np.ones(len(D_thisz))
                score = 1.0
                ci = c0.copy()
                nmf_i = 0
                ci_old = c0.copy()
                while score >= NMF_tolerance:
                    # print('here')
                    ci_old = ci
                    d = TT @ ci
                    ci = ci * np.abs(b)/d
                    score = (np.sum(np.abs(ci-ci_old)))/(np.sum(ci_old))
                    nmf_i += 1
                    if nmf_i > NMF_cutoff:
                        break
                    # del ci_old
                params = ci.reshape((len(D_thisz), 1))
                model = ((D_thisz).T @ params).reshape(len(lamb_data))

            # calculate the model from these parameters and this template
            # model = np.zeros_like(lamb_data)
            # for i in range(D.shape[0]):
                # model += params[i]*D_thisz[i,:]

            # calculate the RMS residual
            residual_vs_z0[k] = np.sum((model - spec_data)**2/err_data**2)
        
        # find the trial redshift with the lowest residual
        kbest = int(np.where(residual_vs_z0 == np.min(residual_vs_z0))[0][0])
        residues0 = residual_vs_z0.copy()    # save this residue for error estimation later
        # note the redshift with the lowest residual
        z = ztrial0[kbest]
        # if kbest > 0:
        #     zmin = ztrial0[kbest-1]

        # create second round ztrial and residues regardless for output format purpose
        zmin = z - zgrid_searchsize
        zmax = z + zgrid_searchsize
        if zmin < 0:
            zmin = 0.0
        if zmax > z_fitting_max:
            zmax = z_fitting_max
        ztrial = np.arange(zmin, zmax, 0.001)
        residual_vs_z = np.inf + np.zeros_like(ztrial) # initialize to infinity
        # second round
        if second_stage:
            # calculate residual at each redshift
            # loop over trial redshifts
            for k in range(ztrial.shape[0]):
                # make this redshifted template
                D_thisz = apply_redshift1(D,ztrial[k],lamb_D,lamb_data, filter_infos, conv=conv_last)
                # D_thisz = D_thisz/err_data
                # params = inv(D*D')*D*s'
                # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
                if not NMF:
                    D_thisz = D_thisz/err_data
                    # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
                    # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
                    lassolars1.fit(D_thisz.T, spec_data_wt)
                    coef1 = lassolars1.coef_
                    params = coef1[:,None]
                    model = (D_thisz.T @ coef1 + lassolars1.intercept_) * err_data

                else:
                    T = D_thisz/err_data**2
                    TT = T @ T.T
                    b = T @ (spec_data/err_data**2)
                    c0 = np.ones(len(D_thisz))
                    score = 1.0
                    ci = c0.copy()
                    nmf_i = 0
                    ci_old = c0.copy()
                    while score >= NMF_tolerance:
                        ci_old = ci
                        d = TT @ ci
                        ci = ci * np.abs(b)/d
                        score = (np.sum(np.abs(ci-ci_old)))/(np.sum(ci_old))
                        nmf_i += 1
                        # del ci_old
                        if nmf_i > NMF_cutoff:
                            break
                    params = ci.reshape((len(D_thisz), 1))
                    model = ((D_thisz).T @ params).reshape(len(lamb_data))
                # calculate the model from these parameters and this template
                # model = np.zeros_like(lamb_data)
                # for i in range(D.shape[0]):
                    # model += params[i]*D_thisz[i,:]
                # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))

                # calculate the RMS residual
                residual_vs_z[k] = np.sum((model - spec_data)**2/err_data**2)
            
            # find the trial redshift with the lowest residual
            kbest = int(np.where(residual_vs_z == np.min(residual_vs_z))[0][0])
            # note the redshift with the lowest residual
            z = ztrial[kbest]        

    else:
        z = zinput
        error = False
    # redo the fit at this redshift
    # make this redshifted template
    D_thisz = apply_redshift1(D,z,lamb_D,lamb_data, filter_infos, conv=conv_last)
    # D_thisz = D_thisz/err_data

    # fit the data to this template
    # params = inv(D*D')*D*s'
    # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
    if not NMF:
        D_thisz = D_thisz/err_data
        # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
        # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
        lassolars1.fit(D_thisz.T, spec_data_wt)
        coef1 = lassolars1.coef_
        params = coef1[:,None]
        model = (D_thisz.T @ coef1 + lassolars1.intercept_) * err_data
    else:
        T = D_thisz/err_data**2
        TT = T @ T.T
        b = T @ (spec_data/err_data**2)
        c0 = np.ones(len(D_thisz))
        score = 1.0
        ci = c0.copy()
        nmf_i = 0
        ci_old = c0.copy()
        while score >= NMF_tolerance:
            # print(score)
            ci_old = ci
            d = TT @ ci
            ci = ci * np.abs(b)/d
            score = (np.sum(np.abs(ci-ci_old)))/(np.sum(ci_old))
            nmf_i += 1
            if nmf_i > NMF_cutoff:
                # print(spec_data[0], '\t', nmf_i, '\t', score)
                break
            # del ci_old
        # print(nmf_i)
        params = ci.reshape((len(D_thisz), 1))
        model = ((D_thisz).T @ params).reshape(len(lamb_data))
    # calculate the model for these parameters and this template
    # model = np.zeros_like(lamb_data)
    # for i in range(D.shape[0]):
        # model += params[i]*D_thisz[i,:]
    # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
    model0 = model.copy()
    if zinput:
        residues0 = np.array([np.sum((model - spec_data)**2/err_data**2)])
        residual_vs_z = np.array([np.sum((model - spec_data)**2/err_data**2)])

    params0 = params.copy()
    if not error:
        return z,0.0,0.0,params0,model0,(ztrial0,ztrial),(residues0,residual_vs_z)

    else:
        min_res = np.min(residues0)
        prob0 = np.exp(-(residues0-min_res)/2)
        # calculate integration and normalize prob0
        dz0 = np.diff(ztrial0)
        dprob0 = np.diff(prob0)
        d_area0 = dz0 * (prob0[:-1] + 0.5*dprob0)
        total_prob0 = np.sum(d_area0)
        prob0 = prob0/total_prob0
        d_area0 = d_area0/total_prob0

        c_d_area0 = np.cumsum(d_area0)  # cumulative area from z=0
        # zlow_0 = ztrial0[np.argmin(np.abs(c_d_area0 - probline))]
        zlow_0 = ztrial0[np.argwhere((c_d_area0 - probline)>0)[0][0]]
        # reverse cumulative area
        ztrial0_r = ztrial0[::-1]
        c_d_area0_r = np.cumsum(d_area0[::-1])
        # zhigh_0 = ztrial0_r[np.argmin(np.abs(c_d_area0_r - probline))+1]    # because it is reverse the index had to be added by 1
        zhigh_0 = ztrial0_r[np.argwhere((c_d_area0_r - probline)>0)[0][0]-1]

        # second round with more precision
        zlow_zmin = zlow_0 - zgrid_errsearchsize
        zlow_zmax = zlow_0 + zgrid_errsearchsize
        if zlow_zmin < 0:
            zlow_zmin = 0.0
        if zlow_zmax > z_fitting_max:
            zlow_zmax = z_fitting_max
        ztrial1_low = np.arange(zlow_zmin, zlow_zmax, 0.001)
        residual_vs_z_low = np.inf + np.zeros_like(ztrial1_low)
        for k in range(ztrial1_low.shape[0]):
            # make this redshifted template
            D_thisz = apply_redshift1(D,ztrial1_low[k],lamb_D,lamb_data, filter_infos, conv=conv_last)
            # D_thisz = D_thisz/err_data

            # params = inv(D*D')*D*s'
            # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
            if not NMF:
                D_thisz = D_thisz/err_data
                # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
                # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
                lassolars1.fit(D_thisz.T, spec_data_wt)
                coef1 = lassolars1.coef_
                params = coef1[:,None]
                model = (D_thisz.T @ coef1 + lassolars1.intercept_) * err_data
            else:
                T = D_thisz/err_data**2
                TT = T @ T.T
                b = T @ (spec_data/err_data**2)
                c0 = np.ones(len(D_thisz))
                score = 1.0
                ci = c0.copy()
                nmf_i = 0
                ci_old = c0.copy()
                while score >= NMF_tolerance:
                    ci_old = ci
                    d = TT @ ci
                    ci = ci * np.abs(b)/d
                    score = (np.sum(np.abs(ci-ci_old)))/(np.sum(ci_old))
                    # del ci_old
                    # if ci == ci_old:    # sometimes tolerance can never be achieve; if the solution doesn't change then break out of while loop
                    #     break
                    nmf_i += 1
                    if nmf_i > NMF_cutoff:
                        break
                params = ci.reshape((len(D_thisz), 1))
                model = ((D_thisz).T @ params).reshape(len(lamb_data))
            # calculate the model from these parameters and this template
            # model = np.zeros_like(lamb_data)
            # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))

            # calculate the RMS residual
            residual_vs_z_low[k] = np.sum((model - spec_data)**2/err_data**2)

        zhigh_zmin = zhigh_0 - zgrid_errsearchsize
        zhigh_zmax = zhigh_0 + zgrid_errsearchsize
        if zhigh_zmin < 0:
            zhigh_zmin = 0.0
        if zhigh_zmax > z_fitting_max:
            zhigh_zmax = z_fitting_max
        ztrial1_high = np.arange(zhigh_zmin, zhigh_zmax, 0.001)
        residual_vs_z_high = np.inf + np.zeros_like(ztrial1_high)
        for k in range(ztrial1_high.shape[0]):
            # make this redshifted template
            D_thisz = apply_redshift1(D,ztrial1_high[k],lamb_D,lamb_data, filter_infos, conv=conv_last)
            # D_thisz = D_thisz/err_data

            # params = inv(D*D')*D*s'
            # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
            if not NMF:
                D_thisz = D_thisz/err_data
                # params = np.linalg.inv(D_thisz @ D_thisz.transpose()) @ D_thisz @ spec_data_reshaped    # TESTING
                # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))
                lassolars1.fit(D_thisz.T, spec_data_wt)
                coef1 = lassolars1.coef_
                params = coef1[:,None]
                model = (D_thisz.T @ coef1 + lassolars1.intercept_) * err_data
            else:
                T = D_thisz/err_data**2
                TT = T @ T.T
                b = T @ (spec_data/err_data**2)
                c0 = np.ones(len(D_thisz))
                score = 1.0
                ci = c0.copy()
                nmf_i = 0
                ci_old = c0.copy()
                while score >= NMF_tolerance:
                    ci_old = ci
                    d = TT @ ci
                    ci = ci * np.abs(b)/d
                    score = (np.sum(np.abs(ci-ci_old)))/(np.sum(ci_old))
                    # del ci_old
                    # if ci == ci_old:    # sometimes tolerance can never be achieve; if the solution doesn't change then break out of while loop
                    #     break
                    nmf_i += 1
                    if nmf_i > NMF_cutoff:
                        break
                params = ci.reshape((len(D_thisz), 1))
                model = ((D_thisz).T @ params).reshape(len(lamb_data))
            # calculate the model from these parameters and this template
            # model = np.zeros_like(lamb_data)
            # model = ((D_thisz*err_data).T @ params).reshape(len(lamb_data))

            # calculate the RMS residual
            residual_vs_z_high[k] = np.sum((model - spec_data)**2/err_data**2)

        # combine residues[0] with two more residue arrays
        ztrial1_unsorted = np.concatenate((ztrial0, ztrial1_low, ztrial1_high))

        sort_idx = np.argsort(ztrial1_unsorted)
        ztrial1 = ztrial1_unsorted[sort_idx]
        residues1 = np.concatenate((residues0, residual_vs_z_low, residual_vs_z_high))
        min_res = np.min(residues1)
        prob1 = np.exp(-(residues1[sort_idx]-min_res)/2)
        dz1 = np.diff(ztrial1)
        dprob1 = np.diff(prob1)
        d_area1 = dz1 * (prob1[:-1] + 0.5*dprob1)
        total_prob1 = np.sum(d_area1)
        # total_prob1 = np.trapz(prob1, x=ztrial1)
        prob1 = prob1/total_prob1
        d_area1 = d_area1/total_prob1

        c_d_area1 = np.cumsum(d_area1)
        # zlow_1 = ztrial1[np.argmin(np.abs(c_d_area1 - probline))]
        zlow_1 = ztrial1[np.argwhere((c_d_area1 - probline)>0)[0][0]]

        # reverse cumulative area
        ztrial1_r = ztrial1[::-1]
        c_d_area1_r = np.cumsum(d_area1[::-1])
        # zhigh_1 = ztrial1_r[np.argmin(np.abs(c_d_area1_r - probline))+1]
        zhigh_1 = ztrial1_r[np.argwhere((c_d_area1_r - probline)>0)[0][0]-1]

        return z,zlow_1,zhigh_1,params0,model0,(ztrial0,ztrial),(residues0,residual_vs_z)




# residue and likelihood plots for an individual galaxy
def igplots(ig, lamb_obs, spec_obs, spec_obs_original, err_obs, lamb_rest, D_rest, zgrid, filter_infos, ztrue, figsize=(10,12)):
    fig, ax = plt.subplots(4,2, figsize=figsize, num=0)

    # delete ax[0,1]
    # fig.delaxes(ax[0,1])
    # adjust ax[0,0] to center
    # ax[0,0].set_position([0.125, 0.75, 0.775, 0.15])
    ax[0,0].errorbar(lamb_obs, spec_obs_original[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1)
    ax[0,0].errorbar(lamb_obs, spec_obs[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1, alpha=0.8)

    # share x axis
    rows = [1,2,3]
    cols = [0,1]
    for ri in range(len(rows)):
        for ci in range(len(cols)):
            ax[rows[ri], cols[ci]].sharex(ax[1,0])

    # no convolution runs
    convs = [False, True]
    full_zgrid = np.arange(min(zgrid,), max(zgrid), 0.001)
    model0 = None

    for i in range(2):

        conv = convs[i]
        # full grid
        t1 = time.time()
        z,zl,zh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
                                                            full_zgrid, filter_infos, conv_first=conv, conv_last=conv, error=True, second_stage=False)
        if i == 0:
            model0 = model.copy()

        t2 = time.time()
        min_res = np.max((np.min(residues[0]),np.min(residues[1])))
        ymin = (min(residues[0])-max(residues[0]))/10
        ymax = max(residues[0])*1.2

        ax[1,i].plot(ztrials[0],residues[0],'.-',linewidth=1,markersize=2)
        ax[1,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
        ax[1,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
        ax[1,i].set_ylim(ymin,ymax)
        ax[1,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s, conv={conv}")
        ax[1,i].grid()
        ax[1,i].set_ylabel(r'Full grid $\chi^2$')    

        # convert to likelihood
        prob0 = np.exp(-(residues[0]-min_res)/2)
        # normalization
        total = np.trapz(prob0, x=ztrials[0])
        # total = 1.0
        prob0 = prob0/total
        ymin=(min(prob0)-max(prob0))/10
        ymax=max(prob0)*1.2

        ax[2,i].plot(ztrials[0],prob0,'.-',linewidth=1,markersize=2)
        ax[2,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
        ax[2,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
        ax[2,i].vlines(x=zl,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
        ax[2,i].vlines(x=zh,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
        ax[2,i].set_ylim(ymin,ymax)
        ax[2,i].set_title(rf"$zp={z:.3f}^{{+{zh-z:.3f}}}_{{-{z-zl:.3f}}}$, zh-zl={zh-zl:.3f}")
        ax[2,i].grid()
        ax[2,i].set_ylabel('Full grid likelihood')

        # optimized zgrid
        t1 = time.time()
        z,zl,zh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
                                                            zgrid, filter_infos, conv_first=conv, conv_last=conv, error=True)
        t2 = time.time()

        for j in range(len(ztrials)):
            ztrial1 = ztrials[j]
            prob1 = np.exp(-(residues[j]-min_res)/2)/total
            ax[3,i].plot(ztrial1,prob1,'.-',linewidth=1,markersize=2)
        ax[3,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
        ax[3,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
        ax[3,i].vlines(x=zl,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
        ax[3,i].vlines(x=zh,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
        ax[3,i].set_ylim(ymin,ymax)
        ax[3,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s, conv={conv}")
        ax[3,i].grid()
        ax[3,i].set_ylabel('Optimized grid likelihood')

    ax[0,0].plot(lamb_obs, model0, '-', markersize=5, linewidth=1, marker='s', fillstyle='none', c='teal')

    fig.supxlabel('ztrials')
    fig.suptitle(f"i={ig}, ztrue={ztrue[ig][0]:.5f}")
    
    fig.tight_layout()

    pos00 = ax[0,0].get_position()
    pos01 = ax[0,1].get_position()

    fig.delaxes(ax[0,1])
    ax[0,0].set_position([pos00.x0, pos00.y0, pos01.x1-pos00.x0, pos00.y1-pos00.y0])
    # plt.show()
    # plt.close(fig)




# residue and likelihood plots for an individual galaxy
# this one doesn't output convolved plots
def igplots2(ig, lamb_obs, spec_obs, spec_obs_original, err_obs, lamb_rest, D_rest, zgrid, filter_infos, ztrue, figsize=(10,12)):
    fig, ax = plt.subplots(3,2, figsize=figsize, num=0)

    # delete ax[0,1]
    # fig.delaxes(ax[0,1])
    # adjust ax[0,0] to center
    # ax[0,0].set_position([0.125, 0.75, 0.775, 0.15])
    ax[0,0].errorbar(lamb_obs, spec_obs_original[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1)
    ax[0,0].errorbar(lamb_obs, spec_obs[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1, alpha=0.8)

    # share x axis
    rows = [1,2]
    cols = [0,1]
    for ri in range(len(rows)):
        for ci in range(len(cols)):
            ax[rows[ri], cols[ci]].sharex(ax[1,0])
    ax[1,0].sharey(ax[1,1])
    ax[2,0].sharey(ax[2,1])

    full_zgrid = np.arange(min(zgrid,), max(zgrid), 0.001)
    grids = [full_zgrid, zgrid]
    second_stages = [False, True]
    sub_titles = ['Full Grid', 'Optimized Grid']
    j_end = [1,2]

    model0 = None
    total0 = None

    for i in range(2):

        grid = grids[i]
        second_stage = second_stages[i]
        # full grid
        t1 = time.time()
        z,zl,zh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
                                                            grid, filter_infos, error=True, second_stage=second_stage)
        if i == 0:
            model0 = model.copy()

        t2 = time.time()
        if i == 0:
            min_res = np.min(residues[0])
        elif i == 1:
            min_res = np.min((np.min(residues[0]),np.min(residues[1])))

        ymin = (min(residues[0])-max(residues[0]))/10
        ymax = max(residues[0])*1.2

        for j in range(j_end[i]):
            ax[1,i].plot(ztrials[j],residues[j],'.-',linewidth=1,markersize=2)
        ax[1,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
        ax[1,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
        ax[1,i].set_ylim(ymin,ymax)
        ax[1,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s")
        ax[1,i].grid()
        ax[1,i].set_ylabel(rf'{sub_titles[i]} $\chi^2$')    


        if i == 0:
            prob0 = np.exp(-(residues[0]-min_res)/2)
            # normalization
            total0 = np.trapz(prob0, x=ztrials[0])
            prob0 = prob0/total0
            pmin = (min(prob0)-max(prob0))/10
            pmax = max(prob0)*1.2


        for j in range(j_end[i]):
            ztrial1 = ztrials[j]
            prob1 = np.exp(-(residues[j]-min_res)/2)
            prob1 = prob1/total0
            ax[2,i].plot(ztrial1,prob1,'.-',linewidth=1,markersize=2)
        ax[2,i].vlines(x=ztrue[ig][0],ymin=pmin,ymax=pmax,alpha=0.6)
        ax[2,i].vlines(x=z,ymin=pmin,ymax=pmax,alpha=0.6,color='salmon')
        ax[2,i].vlines(x=zl,ymin=pmin,ymax=pmax,alpha=0.5,color='teal', linestyle='--')
        ax[2,i].vlines(x=zh,ymin=pmin,ymax=pmax,alpha=0.5,color='teal', linestyle='--')
        ax[2,i].set_ylim(pmin,pmax)
        ax[2,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s")
        ax[2,i].grid()
        ax[2,i].set_ylabel(f'{sub_titles[i]} likelihood')


    ax[0,0].plot(lamb_obs, model0, '-', markersize=5, linewidth=1, marker='s', fillstyle='none', c='teal')

    fig.supxlabel('ztrials')
    fig.suptitle(f"i={ig}, ztrue={ztrue[ig][0]:.5f}")
    
    fig.tight_layout()

    pos00 = ax[0,0].get_position()
    pos01 = ax[0,1].get_position()

    fig.delaxes(ax[0,1])
    ax[0,0].set_position([pos00.x0, pos00.y0, pos01.x1-pos00.x0, pos00.y1-pos00.y0])
    # plt.show()
    # plt.close(fig)

