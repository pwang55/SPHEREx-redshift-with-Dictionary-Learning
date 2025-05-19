import numpy as np
import pandas as pd
from numba import jit, njit
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
import yaml


c = 3e18

class keywords:
    def __init__(self, config=''):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        larslasso_kws = config['LARSlasso']
        zgrid_kws = config['Zgrid']
        self.larslasso_kws = larslasso_kws
        self.zgrid_kws = zgrid_kws

        zmax = zgrid_kws['zmax']
        zmin = zgrid_kws['zmin']
        dz = zgrid_kws['dz']
        scale_1plusz = zgrid_kws['scale_1plusz']
        local_finegrid = zgrid_kws['local_finegrid']
        local_finegrid_size = zgrid_kws['local_finegrid_size']
        local_finegrid_dz = zgrid_kws['local_finegrid_dz']
        testing_zgrid = zgrid_kws['testing_zgrid']
        if testing_zgrid:
            local_finegrid = True

        filter_location = config['Directory_locations']['filter_location']
        if filter_location != 'None':
            convolve_filters = config['Fitting']['convolve_filters']
            convolve_filter = convolve_filters[0] # choose to convolve templates with filter or not in the first stage of optimized grid search
            last_stage_convolve_filter = convolve_filters[1]   # whether to colvolve with filters in the last stage of grid search 
            fitting_convolve_filter = convolve_filters[2]
            self.fitting_convolve_filter = fitting_convolve_filter
        else:
            convolve_filter = False
            last_stage_convolve_filter = False
            fitting_convolve_filter = False
            self.fitting_convolve_filter = fitting_convolve_filter
        # baseline keywords
        self.fit_spectrum_kws = {
                            'larslasso': larslasso_kws['LARSlasso'],
                            'alpha': larslasso_kws['alpha'],
                            'alpha_sig': larslasso_kws['alpha_sigma'],
                            'lars_positive': larslasso_kws['positive'],
                            'LARSlasso_alpha_selection_only': larslasso_kws['alpha_selection_only'],
                            'alpha_ns_scaling': larslasso_kws['alpha_scaling'],
                            'zmax': zmax,
                            'zmin': zmin,
                            'dz': dz,
                            'scale_1plusz': scale_1plusz,
                            'probline': config['Fitting']['probline'],
                            'conv': convolve_filter,
                            'conv_finegrid': last_stage_convolve_filter,
                            'error': False,
                            'local_finegrid': local_finegrid,
                            'local_finegrid_size': local_finegrid_size,
                            'local_finegrid_dz': local_finegrid_dz
                            }

    def train_fit(self):
        train_fit_kws = self.fit_spectrum_kws.copy()
        return train_fit_kws
    def validation_fit(self):
        validation_fit_kws = self.fit_spectrum_kws.copy()
        validation_fit_kws['error'] = True
        validation_fit_kws['conv_finegrid'] = self.fitting_convolve_filter
        validation_fit_kws['LARSlasso_alpha_selection_only'] = self.larslasso_kws['alpha_selection_only_fitting']
        return validation_fit_kws



# generate zgrid with (1+z) scaling
def generate_zgrid(zmin=0.0, zmax=3.0, dz=0.002, scale_1plusz=True, testing_zgrid=False):
    if not testing_zgrid:
        zgridlist = []
        zgi = zmin
        while zgi < zmax:
            zgridlist.append(zgi)
            if scale_1plusz:
                zgi += dz * (1+zgi)
            else:
                zgi += dz
        zgrid = np.array(zgridlist)
    else:
        zgrid_separation = np.array([0, 0.1, 0.3, 0.5, 1, 1.5])
        zgrid_separation = np.append(zgrid_separation, zmax)
        zgrid_stepsizes = np.array([0.002, 0.005, 0.01, 0.01, 0.01, 0.02])
        zgrid = []
        for i in range(len(zgrid_separation)-1):
            zgrid1 = np.arange(zgrid_separation[i], zgrid_separation[i+1], zgrid_stepsizes[i])
            zgrid.extend(zgrid1)
        zgrid.append(zmax)
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


class Catalog:
    def __init__(self, pathfile='', Ndat=None, centering=False, max_SNR=np.inf, min_SNR=-np.inf, SNR_channel=51):
        # print('Read from file...')
        data = np.load(pathfile)

        ztrue = data['z']
        lamb_obs = data['wavelengths']
        spec_obs = data['spectra']
        err_obs = data['error']
        snr = spec_obs/err_obs

        try:
            desi_flag = data['desi_flag']
        except:
            desi_flag = np.zeros(len(spec_obs))
        try:
            spec_obs_original = data['spectra_original']
        except: 
            spec_obs_original = spec_obs.copy()

        snr_mask = (snr[:,SNR_channel]>min_SNR) & (snr[:,SNR_channel]<max_SNR)
        ztrue = ztrue[snr_mask][:Ndat]
        spec_obs = spec_obs[snr_mask][:Ndat]
        spec_obs_original = spec_obs_original[snr_mask][:Ndat]
        err_obs = err_obs[snr_mask][:Ndat]
        desi_flag = desi_flag[snr_mask][:Ndat]

        if centering:
            spec_obs = spec_obs - np.mean(spec_obs, axis=1)[:,None]
            spec_obs_original = spec_obs_original - np.mean(spec_obs_original, axis=1)[:,None]

        self.ztrue = ztrue
        self.lamb_obs = lamb_obs
        self.spec_obs = spec_obs
        self.spec_obs_original = spec_obs_original
        self.err_obs = err_obs
        self.desi_flag = desi_flag
        self.snr = snr

        self.pathfile = pathfile
        self.centering = centering
        self.Ndat = Ndat
        self.max_SNR = max_SNR
        self.min_SNR = min_SNR
        self.SNR_channel = SNR_channel
        self.snr_i = snr[:,51]

    def obj(self, idx):
        obji = SimpleNamespace()
        obji.z = self.ztrue[idx]
        obji.lbs = self.lamb_obs
        obji.spec = self.spec_obs[idx]
        obji.err = self.err_obs[idx]
        obji.spec0 = self.spec_obs_original[idx]
        obji.desi = self.desi_flag[idx]
        obji.snr = self.snr[idx]
        return obji

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
    if filter_location != '':
        filter_location = filter_location.rstrip('/') + '/'    
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
        filt_norm[i] = max(lb[ftrans>0])-min(lb[ftrans>0])
        filt_all_lams.extend(lb[ftrans>0])
    filt_all_lams = np.array(filt_all_lams)
    filt_length = len(lb[ftrans>0])  # each filters length
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

# Function for linear interpolation with 2-d array
@jit(nopython=True, fastmath=True)
def multiInterp1(x, xp, fp):
    '''
    Interpolate 2-D matrix fp along axis=1 at point x
    Caution: a very long x array might make it slower than using np.interp with for loop

    Parameters
    ----------
    x : ndarray of shape n
        Data points to interpolate
    
    xp : ndarray of shape np
        X coordinate points for matrix fp in axis=1
    
    fp : ndarray of shape (nf, np)
        2-D matrix containing nf rows of Y values along axis=1
    
    Returns
    -------
    f : ndarray of shape (nf, n)
        Interpolated 2-D matrix
        
    '''
    ji = np.searchsorted(xp, x) - 1
    jf = ji+1
    xj = (x-xp[ji])/(xp[jf]-xp[ji])
    f = (1-xj) * fp[:,ji] + xj * fp[:,jf]
    return f


# function to integrate fluxes convolved with filters and return all 102 values
@njit(fastmath=True)
def f_convolve_filter(wl, flux, filter_info=None):
    if filter_info != None:
        Nfilt, filt_length, filt_norm, filt_all_lams, filt_all_lams_reshaped = filter_info
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
    else:
        flux_conv = flux
    return flux_conv


# utility function to apply a redshift to spectral dictionary
@jit(nopython=True, fastmath=True)
def apply_redshift(D,z,lamb_in,lamb_out):
    # initialize output dictionary
    D_out = np.zeros((D.shape[0],len(lamb_out)))
    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in*(1+z)
    D_out = multiInterp1(lamb_out, lamb_inx1pz, D)
    return D_out

# apply redshift function with additional option to convolve with filters
@jit(nopython=True, fastmath=True)
def apply_redshift1(D,z,lamb_in,lamb_out, filter_info=None, conv=False):
    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in * (1+z)
    if conv and filter_info != None:
        D_out = np.zeros((D.shape[0],len(lamb_out)))
        for i in range(D.shape[0]):
            D_conv = f_convolve_filter(lamb_inx1pz, D[i,:], filter_info)
            D_out[i,:] = D_conv
        # else:
        #     D_out[i,:] = np.interp(lamb_out,lamb_inx1pz,D[i,:])
    else:
        D_out = multiInterp1(lamb_out, lamb_inx1pz, D)
    return D_out

# apply redshift to all zgrid points and output 3-d array
@njit(fastmath=True)
def apply_redshift_all(D, zgrid, lamb_in, lamb_out, filter_info=None, conv=False):
    nz = zgrid.shape[0]
    D_all = np.zeros((nz, D.shape[0], lamb_out.shape[0]))
    for i in range(nz):
        D_thisz = apply_redshift1(D, zgrid[i], lamb_in, lamb_out, filter_info=filter_info, conv=conv)
        D_all[i] = D_thisz
    return D_all


    
# Fit data with dictionary across the whole redshift grid
@jit(nopython=True, fastmath=True)
def fit_zgrid(lamb_data, spec_data, err_data, lamb_D, D_rest=None, D_allz=None, zgrid=None, zinput=False, filter_info=None,
                larslasso=False, alpha=0, alpha_sig=0.0, lars_positive=False, alpha_ns_scaling=False, LARSlasso_alpha_selection_only=False,
                dz=0.002, zmin=0.0, zmax=3.0, scale_1plusz=True, error=False, probline=0.317/2, 
                conv=False, conv_finegrid=False, local_finegrid=False, local_finegrid_size=0.03, local_finegrid_dz=0.001):
    # Check if D_rest is necessary but not given
    if (zinput or local_finegrid) and D_rest is None:
        raise ValueError("D_rest is required but not given for zinput=True")
    if D_rest is None and D_allz is None:
        raise ValueError("Both D_rest and D_allz are not given")
    if D_rest is None and D_allz is not None:
        D_rest = np.zeros((8,lamb_D.shape[0])) # set D_rest to some float array so numba can fix D_rest type

    # if redshift is fixed and doesn't need error, have only single point in the redshift grid and turn off local_finegrid
    if zinput and not error:
        zinput_float = np.float64(zinput)
        zgrid = np.array([zinput_float])
        local_finegrid = False
        zbest0 = zinput_float
        D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filter_info=filter_info, conv=conv)
        D_allz = D_zinput.reshape((1, D_zinput.shape[0], D_zinput.shape[1]))
    
    else:
        # if zgrid is not given, create a default zgrid
        zmin = np.float64(zmin)
        zmax = np.float64(zmax)
        if zgrid is None:
            zgridlist = [] 
            zgi = zmin
            while zgi < zmax:
                zgridlist.append(zgi)
                if scale_1plusz:
                    zgi += dz * (1+zgi)
                else:
                    zgi += dz
            zgrid = np.array(zgridlist)    
        else:
            zgrid_limit = (zgrid<=zmax) & (zgrid>=zmin)
            zgrid = zgrid[zgrid_limit]

        # if zinput is still given, add z=zinput to zgrid
        if zinput:
            # unfortunately numba doesn't support np.insert
            zinput_float = np.float64(zinput)
            zgrid_copy = zgrid.copy()
            zgrid = np.hstack((zgrid_copy, np.array([zinput_float]))) # input of hstack in numba have to be all arrays
            idx_zgrid = np.argsort(zgrid)
            zgrid = zgrid[idx_zgrid]
            zbest0 = zinput_float
            idx_zbest0 = np.argmax(idx_zgrid)

        # check if D_allz is given
        if D_allz is None:
            D_allz = apply_redshift_all(D_rest, zgrid, lamb_D, lamb_data, filter_info=filter_info, conv=conv)
        elif zinput:
            # if D_allz and zinput are provided, create a new D_allz array and insert D_zinput to the correct location
            D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filter_info=filter_info, conv=conv)
            D_allz_copy = D_allz.copy()
            D_allz = np.zeros((D_allz_copy.shape[0]+1, D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
            D_allz[:D_allz_copy.shape[0]] = D_allz_copy
            D_allz[-1] = D_zinput
            D_allz = D_allz[idx_zgrid]

    # after above steps, zgrid and D_allz are ready for any given situation
    # prepare arrays to store coefficients and chi2
    coefs_zgrid = np.zeros((zgrid.shape[0], D_allz.shape[1]))
    chi2_zgrid = np.zeros_like(zgrid) + np.inf

    # loop over zgrid to calculate best-fit and chi2
    for i in range(zgrid.shape[0]):
        D_thisz = D_allz[i]
        if not larslasso:
            coefs, model = fit_models_ols(D_thisz, spec_data, err_data)
        else:
            coefs, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, 
                                                positive=lars_positive, unit_X=True, unit_y=True, 
                                                max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
        coefs_zgrid[i] = coefs
        chi2_zgrid[i] = np.sum((model - spec_data)**2/err_data**2)

    if not zinput:
        idx_zbest0 = np.argmin(chi2_zgrid)
        zbest0 = zgrid[idx_zbest0]

    if not local_finegrid:
        idx_zbest1 = idx_zbest0
        zbest1 = zbest0
    # local fine grid
    else:
        z_localmax = min((zbest0+local_finegrid_size, zmax))
        z_localmin = max((zbest0-local_finegrid_size, zmin))
        zgrid_local = np.arange(z_localmin, z_localmax, local_finegrid_dz)
        local_length = zgrid_local.shape[0]
        D_all_localz = apply_redshift_all(D_rest, zgrid_local, lamb_D, lamb_data, filter_info=filter_info, conv=conv_finegrid)
        coefs_zgrid_local = np.zeros((local_length, D_allz.shape[1]))
        chi2_zgrid_local = np.zeros_like(zgrid_local) + np.inf

        for j in range(local_length):
            D_this_localz = D_all_localz[j]
            if not larslasso:
                coefs, model = fit_models_ols(D_this_localz, spec_data, err_data)
            else:
                coefs, model = fit_model_larslasso(D_this_localz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, 
                                                    positive=lars_positive, unit_X=True, unit_y=True, 
                                                    max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                    LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
            coefs_zgrid_local[j] = coefs
            chi2_zgrid_local[j] = np.sum((model - spec_data)**2/err_data**2)

        # combine coefs, chi2, D_allz, zgrid with local arrays
        zgrid_copy = zgrid.copy()
        zgrid = np.hstack((zgrid_copy, zgrid_local))
        idx_zgrid_new = np.argsort(zgrid)
        zgrid = zgrid[idx_zgrid_new]

        coefs_zgrid_copy = coefs_zgrid.copy()
        chi2_zgrid_copy = chi2_zgrid.copy()
        coefs_zgrid = np.vstack((coefs_zgrid_copy, coefs_zgrid_local))
        chi2_zgrid = np.hstack((chi2_zgrid_copy, chi2_zgrid_local))
        D_allz_copy = D_allz.copy()
        D_allz = np.zeros((zgrid.shape[0], D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
        D_allz[:D_allz_copy.shape[0]] = D_allz_copy
        D_allz[D_allz_copy.shape[0]:] = D_all_localz
        # sort all the combine arrays
        coefs_zgrid = coefs_zgrid[idx_zgrid_new]
        chi2_zgrid = chi2_zgrid[idx_zgrid_new]
        D_allz = D_allz[idx_zgrid_new]

        if zinput:
            idx_zbest1 = np.where(idx_zgrid_new==idx_zbest0)[0][0]  # figure where zinput is
            zbest1 = zinput_float
        else:
            idx_zbest1 = np.argmin(chi2_zgrid)
            zbest1 = zgrid[idx_zbest1]

    # Now we have zbest1, idx_zbest1, zgrid, coefs_zgrid, chi2_zgrid
    if not error:
        zlower = zbest1
        zupper = zbest1
    else:
        zlower, zupper = error_estimation(zgrid=zgrid, chi2=chi2_zgrid, probline=probline)
        # if zlower > zbest1:
        #     zlower = zbest1
        # if zupper < zbest1:
        #     zupper = zbest1
    D_zbest = D_allz[idx_zbest1]
    coef_zbest = coefs_zgrid[idx_zbest1]
    # chi2_zbest = chi2_zgrid[idx_zbest1]
    model_zbest = D_zbest.T @ coef_zbest

    return zbest1, zlower, zupper, coef_zbest, model_zbest, zgrid, chi2_zgrid


@njit(fastmath=True)
def error_estimation(zgrid, chi2, probline=0.317/2):
    min_chi2 = np.min(chi2)
    likelihood = np.exp(-(chi2-min_chi2)/2)
    dz = np.diff(zgrid)
    d_likelihood = np.diff(likelihood)
    d_area = dz * (likelihood[:-1] + 0.5*d_likelihood)  # area by treating each connecting points as triangle
    total_area = np.sum(d_area)
    likelihood = likelihood / total_area
    d_area = d_area / total_area

    cum_d_area_L = np.cumsum(d_area)    # cumulative sum of area from left
    idx_lower1 = np.argwhere((cum_d_area_L - probline)>0)[0][0] - 1
    idx_lower2 = idx_lower1 + 1
    zgrid_area_L = zgrid[1:]    # cumulative area correspond to zgrid[1:]
    zlower1 = zgrid_area_L[idx_lower1]
    zlower2 = zgrid_area_L[idx_lower2]
    dz_lower = zlower2 - zlower1
    factorL = (probline - cum_d_area_L[idx_lower1]) / (cum_d_area_L[idx_lower2] - cum_d_area_L[idx_lower1])
    zlower = zlower1 + factorL * dz_lower

    cum_d_area_R = np.cumsum(d_area[::-1])    # cumulative sum of area from right
    idx_upper1 = np.argwhere((cum_d_area_R - probline)>0)[0][0] - 1
    idx_upper2 = idx_upper1 + 1
    zgrid_area_R = zgrid[:-1][::-1]     # start counting from 2nd to last of zgrid, then reverse
    zupper1 = zgrid_area_R[idx_upper1]
    zupper2 = zgrid_area_R[idx_upper2]
    dz_upper = zupper2 - zupper1
    factorR = (probline - cum_d_area_R[idx_upper1]) / (cum_d_area_R[idx_upper2] - cum_d_area_R[idx_upper1])
    zupper = zupper1 + factorR * dz_upper

    return zlower, zupper

@jit(nopython=True, fastmath=True)
def fit_models_ols(D_thisz, spec_data, err_data):
    X = (D_thisz/err_data).T
    X = np.ascontiguousarray(X)
    y = spec_data/err_data
    coefs = ols(X, y)
    model = (D_thisz*err_data).T @ coefs
    return coefs, model

# OLS function with simple X, y input format
@jit(nopython=True, fastmath=True)
def ols(X, y):
    coefs = np.linalg.inv(X.T @ X) @ X.T @ y
    # model = X @ params
    return coefs


@jit(nopython=True, fastmath=True)
def fit_model_larslasso(D_thisz, spec_data, err_data, alpha=0.0, alpha_sig=0.0, positive=False, unit_X=True, unit_y=True, 
                        max_iter=200, decimals=10, alpha_ns_scaling=False, LARSlasso_alpha_selection_only=False):
    X = (D_thisz/err_data).T    # currently D_thisz is X.T in usual LARSlasso convention
    X = np.ascontiguousarray(X)
    y = spec_data/err_data
    ynorm = np.linalg.norm(y)
    coefs = _larslasso(X, y, alpha=alpha+alpha_sig*1/ynorm, positive=positive, unit_X=unit_X, unit_y=unit_y, 
                        max_iter=max_iter, decimals=decimals, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
    model = (D_thisz).T @ coefs
    return coefs, model

@jit(nopython=True, fastmath=True)
def _larslasso(X, y, alpha=0.0, positive=False, unit_X=True, unit_y=True, centering_X=False, centering_y=False, 
                 intercept=False, max_iter=200, decimals=14, alpha_ns_scaling=False, LARSlasso_alpha_selection_only=False):
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
            if not LARSlasso_alpha_selection_only:
                if n_iter > 0:  # TEMP
                    ss = (prev_alphas - alpha) / (prev_alphas - alphas)
                    coef = prev_coef + ss * (coef - prev_coef)
                    alphas[0] = alpha
            else:
                coef_A = ols(X[:, idx_A_tf], y)
                coef[idx_A_tf] = coef_A
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


def nmad_eta(zs, zp, eta_method=0):
    dz = zp - zs
    dz_1pz = dz/(1+zs)
    nmad = 1.48 * np.median(np.abs((dz - np.median(dz))/(1+zs)))
    if eta_method == 0:
        eta = np.mean(np.abs(dz_1pz) > 0.15)
    elif eta_method == 1:
        std_eta = np.std(dz_1pz)
        eta = np.mean(np.abs(dz_1pz) > 3*std_eta)
    return nmad, eta


class diagnostic_plots:
    def __init__(self, output_dirname):
        self.dpi = 300
        self.output_dirname = output_dirname
        self.fig_num = 0

    def template_plots(self, lamb_rest, D_rest, D_rest_initial):
        
        fig1, ax1 = plt.subplots(num=self.fig_num)
        ax1.plot(lamb_rest, D_rest.T, '-', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel(r"Wavelength [$\mu m$]")
        ax1.set_ylabel('Flux [arb]')
        ax1.set_title('Learned Templates')
        ax1.grid()
        fig1.tight_layout()
        plt.savefig(self.output_dirname+'trained_templates.png', dpi=self.dpi)
        self.fig_num += 1
        templates_figsize = (12, 10)
        tick_fontsize = 6
        
        Ndict = D_rest.shape[0]
        D_diff = D_rest - D_rest_initial
        fig2, ax2 = plt.subplots(Ndict, 2, num=self.fig_num, sharex=True, figsize=(templates_figsize))            
        for i in range(Ndict):
            ax2[i,0].plot(lamb_rest, D_rest_initial[i], alpha=0.8, linewidth=1)
            ax2[i,0].plot(lamb_rest, D_rest[i], alpha=0.8, linewidth=1)
            ax2[i,1].plot(lamb_rest, D_diff[i], linewidth=1)
            ax2[i,1].plot([-1,6], [0,0], color='k', linewidth=0.5, linestyle='--', alpha=0.8)
            ax2[i,0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax2[i,1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax2[i,0].set_ylabel(i, fontsize=tick_fontsize)

        ax2[0,1].set_ylim(np.min(D_diff), np.max(D_diff))
        for i in range(1, Ndict):
            ax2[i, 1].sharey(ax2[0,1])
        ax2[0,0].set_xlim(0,5.5)

        ax2[0,0].set_title('Initial/Trained Dictionaries')
        ax2[0,1].set_title('Trained-Initial')
        fig2.supxlabel(r"Wavelength [$\mu m$]")
        # fig.supylabel()
        fig2.tight_layout()
        plt.savefig(self.output_dirname+f'trained_template_multiplot.png', dpi=self.dpi)
        self.fig_num += 1

    def zp_zs_plots(self, ztrue, zbest_initial, zbest_trained, zmin=0.0, zmax=3.0, catalog='fitting', gridsize=150, gridsize2=(150,30)):
        zpzs_figsize = (12,7)
        lim_offset = 0.05
        bottom_ylim = 0.25

        labelfontsize = 12
        tickfontsize = 10
        legendfontsize = 12
        m0 = 'o'
        m1 = 'o'
        m0size = 2
        m1size = 3
        markeredgewidth = 0.2
        m0alpha = 0.3
        m1alpha = 0.4
        # m0edgec = 'tab:blue'
        # m1edgec = 'tab:orange'
        m0edgec = 'grey'
        m1edgec = 'k'
        # gridsize = 100
        bins = 'log'

        nmad_i, eta_i = nmad_eta(zs=ztrue, zp=zbest_initial)
        nmad_f, eta_f = nmad_eta(zs=ztrue, zp=zbest_trained)

        fig, ax = plt.subplots(2, 2, num=self.fig_num, sharex=True, figsize=zpzs_figsize, gridspec_kw={'height_ratios': [3,1], 'width_ratios': [1,1.25]})
        ax[0,0].set_xlim(zmin-lim_offset,zmax+lim_offset)
        ax[0,0].set_ylim(zmin-lim_offset,zmax+lim_offset)
        ax[1,0].set_ylim(-bottom_ylim, bottom_ylim)
        ax[1,0].set_xlim(zmin-lim_offset,zmax+lim_offset)
        ax[0,1].sharey(ax[0,0])
        ax[1,1].sharey(ax[1,0])
        ax[0,0].grid()
        ax[1,0].grid()
        ax[0,1].grid()
        ax[1,1].grid()

        ax[0,0].set_ylabel('Estimated Redshift', fontsize=labelfontsize)
        ax[1,0].set_xlabel('True Redshift', fontsize=labelfontsize)
        ax[1,1].set_xlabel('True Redshift', fontsize=labelfontsize)

        ax[1,0].set_ylabel(r'$\Delta z/(1+z_{True})$', fontsize=labelfontsize)
        ax[0,0].plot(ztrue, zbest_initial, m0, markersize=m0size, markeredgecolor=m0edgec, markeredgewidth=markeredgewidth, alpha=m0alpha,
                    label=f'Initial, $\eta={eta_i*100:.3f}$%, $\sigma_{{NMAD}}={nmad_i*100:.3f}$%')
        ax[0,0].plot(ztrue, zbest_trained, m1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth, alpha=m1alpha,
                    label=f'Trained, $\eta={eta_f*100:.3f}$%, $\sigma_{{NMAD}}={nmad_f*100:.3f}$%')
        ax[1,0].plot(ztrue, (zbest_initial-ztrue)/(1+ztrue), m0, markersize=m0size, markeredgecolor=m0edgec, \
                     markeredgewidth=markeredgewidth, alpha=m0alpha)
        ax[1,0].plot(ztrue, (zbest_trained-ztrue)/(1+ztrue), m1, markersize=m1size, markeredgecolor=m1edgec, \
                     markeredgewidth=markeredgewidth, alpha=m1alpha)

        hb1 = ax[0,1].hexbin(ztrue, zbest_trained, gridsize=gridsize, extent=[zmin, zmax, zmin, zmax], bins=bins)
        cb1 = fig.colorbar(hb1, ax=ax[0,1])
        hb2 = ax[1,1].hexbin(ztrue, (zbest_trained-ztrue)/(1+ztrue), gridsize=gridsize2, extent=[zmin, zmax, -bottom_ylim, bottom_ylim], bins=bins)
        cb2 = fig.colorbar(hb2, ax=ax[1,1])

        ax[0,0].plot([zmin-lim_offset, zmax+lim_offset], [zmin-lim_offset, zmax+lim_offset], '-', alpha=0.8, color='g', linewidth=2)
        ax[0,1].plot([zmin-lim_offset, zmax+lim_offset], [zmin-lim_offset, zmax+lim_offset], linewidth=0.7, color='salmon', alpha=0.8)
        ax[1,0].plot([zmin-lim_offset, zmax+lim_offset], [0,0],'-', alpha=0.8, color='g', linewidth=2)
        ax[1,1].plot([zmin-lim_offset, zmax+lim_offset], [0,0],'-', linewidth=0.7, color='salmon', alpha=0.8)

        ax[0,0].tick_params(axis='both', which='major', labelsize=tickfontsize)
        ax[1,0].tick_params(axis='both', which='major', labelsize=tickfontsize)
        ax[0,0].legend(fontsize=legendfontsize, framealpha=0.9, loc='upper left')
        # axs[1].legend(fontsize=20, loc='lower right')
        # fig.suptitle('Fitting Catalog Redshift Estimation')
        fig.tight_layout()
        plt.savefig(self.output_dirname+f'redshift_estimation_performance_{catalog}_catalog.png', dpi=self.dpi)
        self.fig_num += 1

    def sparsity_report(self, coefs_trained):
        Ngal = coefs_trained.shape[0]
        Ndict = coefs_trained.shape[1]

        figsize = (6,8)
        fig, ax = plt.subplots(3, 1, figsize=figsize, num=self.fig_num)
        p = coefs_trained
        param_nonzeros = np.zeros(Ndict+1, dtype=int)
        for i in range(Ngal):
            p1 = p[i]
            num_nonzeros = np.sum(p1!=0)
            param_nonzeros[num_nonzeros] += 1
        ax[0].bar(np.arange(Ndict+1), param_nonzeros)
        ax[0].set_title('Number of coefficients')
        p_counts = np.zeros(Ndict, dtype=int)
        p_pos_sums = np.zeros(Ndict)
        p_sums = np.zeros(Ndict)
        for i in range(Ndict):
            pi = p[:,i]
            p_counts[i] = np.sum(pi!=0)
            p_pos_sums[i] = np.sum(np.fabs(pi))
            p_sums[i] = np.sum((pi))

        ax[1].bar(np.arange(Ndict), p_counts, color='teal')
        ax[1].set_title('# of times dictionary have been used')
        ax[2].bar(np.arange(Ndict), p_pos_sums, color='brown', alpha=0.6)
        ax[2].bar(np.arange(Ndict), p_sums, color='salmon', alpha=1.0)

        ax[2].set_title('Sum of coefficients for each dictionary')

        fig.tight_layout()
        plt.savefig(self.output_dirname+'sparsity_reports.png')
        self.fig_num += 1

    def example_seds(self, cat, lamb_rest, D_rest, D_rest_initial, zgrid, validation_fit_kws, idx=None, filter_info=None):
        
        lamb_obs = cat.lamb_obs
        ztrue = cat.ztrue
        spec_obs = cat.spec_obs
        spec_obs_original = cat.spec_obs_original
        err_obs = cat.err_obs
        snr = cat.snr

        if idx is None:
            snr_max = np.max(snr, axis=1)
            qs = [99.9, 95, 85, 75, 50, 10] # target SNR percentile for example SEDs
            idx = []
            for i in range(len(qs)):
                snr_target = np.percentile(snr_max, qs[i])
                idx_target = np.argmin(np.abs(snr_max - snr_target))
                idx.append(idx_target)        

        figsize=(12,8)
        legendfontsize = 8
        ticklabelsize = 6
        labelfontsize = 10
        ms0 = 3
        mec0 = 'cadetblue'
        mew0 = 0.5
        cs0 = 2
        ew0 = 0.5
        alpha0 = 0.6
        c1 = 'green'
        ms1 = 2
        alpha1 = 0.7
        lw2 = 1.2
        c2 = 'k'
        alpha2 = 0.5
        lw3 = 1
        c3 = 'salmon'
        alpha3 = 0.9

        nrows = 2
        ncols = 3
        row_idx = [0,0,0,1,1,1]
        col_idx = [0,1,2,0,1,2]

        fig, ax = plt.subplots(nrows, ncols, sharex=True, num=self.fig_num, figsize=figsize)

        for i in range(len(idx)):
            row = row_idx[i]
            col = col_idx[i]
            idx_i = idx[i]

            # refit with the initial dictionary
            zbest_initial_ex,zlow_initial,zhigh_initial, \
            coefs_initial,best_model_initial,_,_ = fit_zgrid(lamb_obs, spec_obs[idx_i], err_obs[idx_i], lamb_rest, 
                                                            D_rest=D_rest_initial, zinput=False, zgrid=zgrid, 
                                                            filter_info=filter_info, **validation_fit_kws)
            # refit with the trained dictionary
            zbest_trained_ex,zlow_trained,zhigh_trained, \
            coefs_trained,best_model,_,_ = fit_zgrid(lamb_obs, spec_obs[idx_i], err_obs[idx_i], lamb_rest, 
                                                    D_rest=D_rest, zinput=False, zgrid=zgrid, 
                                                    filter_info=filter_info, **validation_fit_kws)
            nAi = np.sum(coefs_initial!=0)
            nAf = np.sum(coefs_trained!=0)
            ymax = np.max(best_model) * 2.0
            ymin = (np.min(best_model) - (np.max(best_model)-np.min(best_model))) * 1.5
            # data_max = spec_obs[i]+err_obs[i]
            # data_min = spec_obs[i]-err_obs[i]
            # ymax = np.max(data_max[:66])*1.2
            # ymin = np.min(data_min[:66])*1.2

            ax[row, col].errorbar(lamb_obs, spec_obs[idx_i], err_obs[idx_i], fmt='o', markersize=ms0, markerfacecolor='none', 
                                markeredgecolor=mec0, markeredgewidth=mew0, capsize=cs0, elinewidth=ew0, alpha=alpha0, label="Photometry")
            ax[row, col].plot(lamb_obs, spec_obs_original[idx_i], '.', color=c1, markersize=ms1, alpha=alpha1, label="Ground Truth")
            ax[row, col].plot(lamb_obs, best_model_initial, '-', linewidth=lw2, color=c2, alpha=alpha2, \
                            label=fr"Initial Template, $z_{{est}}={zbest_initial_ex:.3f}^{{+{zhigh_initial-zbest_initial_ex:.3f}}}_{{-{zbest_initial_ex-zlow_initial:.3f}}}$")
            ax[row, col].plot(lamb_obs, best_model, linewidth=lw3, color=c3, alpha=alpha3, \
                            label=fr"Trained Template, $z_{{est}}={zbest_trained_ex:.3f}^{{+{zhigh_trained-zbest_trained_ex:.3f}}}_{{-{zbest_trained_ex-zlow_trained:.3f}}}$")
            # ax[row, col].set_xlabel('Observed Wavelength [$\mu$m]', fontsize=8)
            # ax[row, col].set_ylabel('Flux [mJy]', fontsize=8)
            ax[row, col].set_title(fr"Idx={idx_i}, $z_{{true}}$={ztrue[idx_i]:.5f}, $n_A$={nAf}", fontsize=8)
            ax[row, col].legend(fontsize=legendfontsize)
            ax[row, col].grid()
            ax[row, col].tick_params(axis='both', which='major', labelsize=ticklabelsize)
            ax[row, col].set_ylim(ymin, ymax)
        fig.supxlabel('Observed Wavelength [$\mu$m]', fontsize=labelfontsize)
        fig.supylabel('Flux [mJy]', fontsize=labelfontsize)
        fig.tight_layout()
        plt.savefig(self.output_dirname+'example_fitted_SEDs.png', dpi=self.dpi)
        self.fig_num += 1

    # Plot fractional uncertaintyt and z-score in 6 uncertainty bins
    def uncertainty_binplots(self, zs, zp, zl, zh, zp0, zl0, zh0, dat=None, nbins=50):
        if dat is not None:
            zs = dat['ztrue']
            zp = dat['zest']
            zl = dat['zlow']
            zh = dat['zhigh']

            zp0 = dat['zest_initial']
            zl0 = dat['zlow_initial']
            zh0 = dat['zhigh_initial']
        
        color_args = {
            'hist1_color': 'salmon',
            'hist1_alpha': 0.9,
            'g1_color': 'red',
            'g1_alpha': 0.9,
            'g1_lwidth': 1,

            'hist1_color0': 'tab:blue',
            'hist1_alpha0': 0.5,
            'g1_color0': 'k',
            'g1_alpha0': 0.4,
            'hist1_lwidth0': 1,
            'g1_lwidth0': 0.8,

            'hist2_color': 'tan',
            'hist2_alpha': 0.9,
            'g2_color': 'k',
            'g2_alpha': 0.7,
            'hist2_lwidth0': 1,
            'g2_lwidth': 0.8,
            
            'hist2_color0': 'tab:blue',    
            'hist2_alpha0': 0.5,
            'g2_color0': 'grey',
            'g2_alpha0': 0.5,
            'g2_lwidth0': 0.8,
            }

        sigma_ranges = [
                (0, 0.003),
                (0.003, 0.01),
                (0.01, 0.03),
                (0.03, 0.1),
                (0.1, 0.2),
                (0.2, 0.5)
            ]

        hist1_ranges = [
                [-0.01, 0.01],
                [-0.05, 0.05],
                [-0.1, 0.1],
                [-0.2, 0.2],
                [-0.50, 0.50],
                [-1.0, 1.0]
            ]
        
        hist2_range = [-5, 5]
        figsize = (12,12)

        sig = (zh-zl)/2
        sig0 = (zh0-zl0)/2
        frac_uncertainty = sig/(1+zp)
        frac_uncertainty0 = sig0/(1+zp0)

        fig, ax = plt.subplots(4, 3, figsize=figsize, num=self.fig_num)
        # ax[1,2].axis('off')
        # ax[3,2].axis('off')

        # hist1_cols = [0,1,2,0,1]
        # hist1_rows = [0,0,0,1,1]
        # hist2_cols = [0,1,2,0,1]
        # hist2_rows = [2,2,2,3,3]

        hist1_cols = [0,1,2,0,1,2]
        hist1_rows = [0,0,0,1,1,1]
        hist2_cols = [0,1,2,0,1,2]
        hist2_rows = [2,2,2,3,3,3]

        for i, (low, high) in enumerate(sigma_ranges):
            h = (frac_uncertainty<high) & (frac_uncertainty>low)
            h0 = (frac_uncertainty0<high) & (frac_uncertainty0>low)

            frac_uncertainty_i = frac_uncertainty[h]
            frac_uncertainty0_i = frac_uncertainty0[h0]

            zsi = zs[h]
            zpi = zp[h]     
            sigi = sig[h]

            zs0i = zs[h0]
            zp0i = zp0[h0]     
            sig0i = sig0[h0]

            dzi = zpi - zsi
            dz0i = zp0i - zs0i
            err_dist_i = dzi/(1 + zsi)
            err_dist0_i = dz0i/(1 + zs0i)

            nmad_i, eta_i = nmad_eta(zs=zsi, zp=zpi)
            nmad0_i, eta0_i = nmad_eta(zs=zs0i, zp=zp0i)

            bias_i = np.mean(err_dist_i)
            bias0_i = np.mean(err_dist0_i)

            med_frac_i = np.median(frac_uncertainty_i)
            med_frac0_i = np.median(frac_uncertainty0_i)
            ngals_i = len(zpi)
            ngals0_i = len(zp0i)

            row1i = hist1_rows[i]
            col1i = hist1_cols[i]

            cts, bins_i, _ = ax[row1i, col1i].hist(err_dist_i, bins=nbins, range=hist1_ranges[i], color=color_args['hist1_color'], alpha=color_args['hist1_alpha'])
            cts0, bins0_i, _ = ax[row1i, col1i].hist(err_dist0_i, bins=nbins, range=hist1_ranges[i], histtype='step', 
                                                    linewidth=color_args['hist1_lwidth0'], color=color_args['hist1_color0'], alpha=color_args['hist1_alpha0'])

            g_pts = 100
            gaussian_xi = np.linspace(hist1_ranges[i][0], hist1_ranges[i][1], g_pts)
            gaussian_i = np.max(cts) * np.exp(-gaussian_xi**2/(2*med_frac_i**2))
            gaussian0_i = np.max(cts0) * np.exp(-gaussian_xi**2/(2*med_frac0_i**2))

            ax[row1i, col1i].plot(gaussian_xi, gaussian_i, linewidth=color_args['g1_lwidth'], color=color_args['g1_color'], alpha=color_args['g1_alpha'])
            ax[row1i, col1i].plot(gaussian_xi, gaussian0_i, '--', linewidth=color_args['g1_lwidth0'], color=color_args['g1_color0'], alpha=color_args['g1_alpha0'])

            stats_text = (
                fr"NMAD: {nmad0_i:.5f}$\rightarrow${nmad_i:.5f}"+'\n'
                fr"Bias: {bias0_i:.5f}$\rightarrow${bias_i:.5f}"+'\n'
                fr"$\sigma_{{z/(1+z)}}$: {med_frac0_i:.4f}$\rightarrow${med_frac_i:.4f}"+'\n'
                fr"$\eta$: {eta0_i*100:.2f}%$\rightarrow${eta_i*100:.2f}%"+'\n'
                fr"$N_g$: {ngals0_i}$\rightarrow${ngals_i}"
            )
            
            ax[row1i, col1i].text(0.02, 0.98, stats_text, ha='left', va='top', transform=ax[row1i, col1i].transAxes, fontsize=7,
                    bbox=dict(facecolor='none', alpha=0.7, linewidth=0.0, edgecolor='black', boxstyle='square,pad=0.5'))

            ax[row1i, col1i].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
            ax[row1i, col1i].set_xlabel(r'($z_p$-$z_s$)/(1+$z_s$)')
            ax[row1i, col1i].set_ylabel(r"$N_g$")
            ax[row1i, col1i].grid(alpha=0.4)

            row2i = hist2_rows[i]
            col2i = hist2_cols[i]

            zscore_i = (zpi - zsi)/sigi
            zscore0_i = (zp0i - zs0i)/sig0i

            mean_zscore_i = np.mean(zscore_i)
            mean_zscore0_i = np.mean(zscore0_i)
            sig_zscore_i = np.std(zscore_i)
            sig_zscore0_i = np.std(zscore0_i)

            cts2, bins2, _ = ax[row2i, col2i].hist(zscore_i, bins=nbins, range=hist2_range, color=color_args['hist2_color'], alpha=color_args['hist2_alpha'])
            cts0_2, bins0_2, _ = ax[row2i, col2i].hist(zscore0_i, bins=nbins, range=hist2_range, histtype='step', 
                                                    linewidth=color_args['hist2_lwidth0'], color=color_args['hist2_color0'], alpha=color_args['hist2_alpha0'])

            gaussian_xi = np.linspace(hist2_range[0], hist2_range[1], g_pts)
            gaussian_zscore_i = np.max(cts2) * np.exp(-gaussian_xi**2/(2))
            gaussian_zscore0_i = np.max(cts0_2) * np.exp(-gaussian_xi**2/(2))
            ax[row2i, col2i].plot(gaussian_xi, gaussian_zscore_i, linewidth=color_args['g2_lwidth'], color=color_args['g2_color'], alpha=color_args['g2_alpha'])
            ax[row2i, col2i].plot(gaussian_xi, gaussian_zscore0_i, '--', linewidth=color_args['g2_lwidth0'], color=color_args['g2_color0'], alpha=color_args['g2_alpha0'])
            ax[row2i, col2i].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
            ax[row2i, col2i].set_xlabel(r'($z_p$-$z_s$)/$\sigma_z$')
            ax[row2i, col2i].set_ylabel(r"$N_g$")
            ax[row2i, col2i].grid(alpha=0.4)

            stats_text2 = (
                fr"($\mu$, $\sigma$)=({mean_zscore0_i:.2f}, {sig_zscore0_i:.2f})"+'\n'+fr"$\rightarrow$({mean_zscore_i:.2f}, {sig_zscore_i:.2f})"
            )
            ax[row2i, col2i].text(0.35, 0.98, stats_text2, ha='right', va='top', transform=ax[row2i, col2i].transAxes, fontsize=7,
                    bbox=dict(facecolor='none', alpha=0.7, linewidth=0.0, edgecolor='black', boxstyle='square,pad=0.5'))
        fig.suptitle('Dictionary learning')
        fig.text(0.02, 0.73, 'Redshift error distribution', va='center', ha='center', rotation='vertical', fontsize=10)
        fig.text(0.02, 0.28, 'Z-Score', va='center', ha='center', rotation='vertical', fontsize=10)
        fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.98])
        plt.savefig(self.output_dirname+'binplot_error_dist_zscore.png', dpi=self.dpi)
        self.fig_num += 1



    def hexbin_binplot(self, zs, zp, zl, zh, zp0, zl0, zh0, dat=None, gridsize=100):
        figsize = (12,12)
        cmap1 = 'viridis'
        cmap2 = 'cividis'

        if dat is not None:
            zs = dat['ztrue']
            zp = dat['zest']
            zl = dat['zlow']
            zh = dat['zhigh']

            zp0 = dat['zest_initial']
            zl0 = dat['zlow_initial']
            zh0 = dat['zhigh_initial']

        fig1, ax1 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=figsize, num=self.fig_num)
        zmin = 0.0
        zmax = 3.0
        # gridsize = (80,100)
        hexbins_scale = 'log'

        sigma_ranges = [
                (0, 0.003),
                (0.003, 0.01),
                (0.01, 0.03),
                (0.03, 0.1),
                (0.1, 0.2),
                (0.2, 0.5)
            ]

        p1_cols = [0,1,2,0,1,2]
        p1_rows = [0,0,0,1,1,1]
        p2_cols = [0,1,2,0,1,2]
        p2_rows = [2,2,2,3,3,3]

        sig = (zh-zl)/2
        sig0 = (zh0-zl0)/2
        frac_uncertainty = sig/(1+zp)
        frac_uncertainty0 = sig0/(1+zp0)

        for i, (low, high) in enumerate(sigma_ranges):
            h = (frac_uncertainty<high) & (frac_uncertainty>low)
            h0 = (frac_uncertainty0<high) & (frac_uncertainty0>low)

            zsi = zs[h]
            zpi = zp[h]
            zs0i = zs[h0]
            zp0i = zp0[h0]

            dzi = zpi - zsi
            dz0i = zp0i - zs0i
            err_dist_i = dzi/(1 + zsi)
            err_dist0_i = dz0i/(1 + zs0i)

            nmad_i, eta_i = nmad_eta(zs=zsi, zp=zpi)
            nmad0_i, eta0_i = nmad_eta(zs=zs0i, zp=zp0i)
            bias_i = np.mean(err_dist_i)
            bias0_i = np.mean(err_dist0_i)
            ngals_i = len(zpi)
            ngals0_i = len(zp0i)

            row1i = p1_rows[i]
            col1i = p1_cols[i]

            if sum(h) > 0:
                hb1 = ax1[row1i, col1i].hexbin(zsi, zpi, gridsize=gridsize, extent=[zmin, zmax, zmin, zmax], bins=hexbins_scale, cmap=cmap1)
                cb1 = fig1.colorbar(hb1, ax=ax1[row1i, col1i])
            else:
                hb1 = ax1[row1i, col1i].hexbin([0], [0], gridsize=gridsize, extent=[zmin, zmax, zmin, zmax], bins=hexbins_scale, cmap=cmap1)
                cb1 = fig1.colorbar(hb1, ax=ax1[row1i, col1i])
            ax1[row1i, col1i].plot([zmin, zmax], [zmin, zmax], linewidth=0.7, color='salmon', alpha=0.8)

            stats_text = (
                fr"NMAD: {nmad_i:.5f}"+'\n'
                fr"Bias: {bias_i:.5f}"+'\n'
                fr"$\eta$: {eta_i*100:.2f}%"+'\n'
                fr"$N_g$: {ngals_i}"
            )            
            ax1[row1i, col1i].text(0.62, 0.23, stats_text, ha='left', va='top', transform=ax1[row1i, col1i].transAxes, fontsize=7,
                    bbox=dict(facecolor='w', alpha=0.7, linewidth=0.0, edgecolor='black', boxstyle='square,pad=0.5'))

            ax1[row1i, col1i].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
            ax1[row1i, col1i].set_xlabel(r'$z_{true}$')
            ax1[row1i, col1i].set_ylabel(r"$z_{est}$")
            # ax1[row1i, col1i].grid(alpha=0.4)
            ax1[row1i, col1i].set_xlim(zmin, zmax)
            ax1[row1i, col1i].set_ylim(zmin, zmax)

            row2i = p2_rows[i]
            col2i = p2_cols[i]

            if sum(h0)>0:
                hb2 = ax1[row2i, col2i].hexbin(zs0i, zp0i, gridsize=gridsize, extent=[zmin, zmax, zmin, zmax], bins=hexbins_scale, cmap=cmap2)
                cb2 = fig1.colorbar(hb2, ax=ax1[row2i, col2i])
            else:
                hb2 = ax1[row2i, col2i].hexbin([0], [0], gridsize=gridsize, extent=[zmin, zmax, zmin, zmax], bins=hexbins_scale, cmap=cmap2)
                cb2 = fig1.colorbar(hb2, ax=ax1[row2i, col2i])
            ax1[row2i, col2i].plot([zmin, zmax], [zmin, zmax], linewidth=0.7, color='salmon', alpha=0.8)

            ax1[row2i, col2i].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
            ax1[row2i, col2i].set_xlabel(r'$z_{true}$')
            ax1[row2i, col2i].set_ylabel(r"$z_{est}$")
            # ax1[row2i, col2i].grid(alpha=0.4)
            ax1[row2i, col2i].set_xlim(zmin, zmax)
            ax1[row2i, col2i].set_ylim(zmin, zmax)

            stats_text2 = (
                fr"NMAD: {nmad0_i:.5f}"+'\n'
                fr"Bias: {bias0_i:.5f}"+'\n'
                fr"$\eta$: {eta0_i*100:.2f}%"+'\n'
                fr"$N_g$: {ngals0_i}"
            )
            ax1[row2i, col2i].text(0.62, 0.23, stats_text2, ha='left', va='top', transform=ax1[row2i, col2i].transAxes, fontsize=7,
                    bbox=dict(facecolor='w', alpha=0.7, linewidth=0.0, edgecolor='black', boxstyle='square,pad=0.5'))

        # fig.suptitle('Dictionary learning')
        fig1.text(0.02, 0.73, 'Trained dictionary performance', va='center', ha='center', rotation='vertical', fontsize=10)
        fig1.text(0.02, 0.28, 'Initial dictionary performance', va='center', ha='center', rotation='vertical', fontsize=10)
        fig1.tight_layout(rect=[0.02, 0.0, 1.0, 0.98])
        plt.savefig(self.output_dirname+'binplot_hexbin.png', dpi=self.dpi)
        self.fig_num += 1


    def fit_eazy_plots(self, lamb_rest, D_rest, templates_EAZY):

        # lamb_rest_resolution = np.mean(np.diff(lamb_rest))
        lamb_rest_resolution = 0.01
        lamb_um = np.arange(0.3, 4.8, lamb_rest_resolution)
        h_lamb_um = (lamb_rest>=min(lamb_um)-0.000001) & (lamb_rest<max(lamb_um))
        templates_EAZY = templates_EAZY[:,h_lamb_um]
        templates_EAZY = templates_EAZY/np.linalg.norm(templates_EAZY, axis=1)[:,None]

        D_rest_interpolated = multiInterp1(lamb_um, lamb_rest, D_rest)
        templates_figsize = (8, 8)
        tick_fontsize = 6
        fig, ax = plt.subplots(7, 1, figsize=templates_figsize, num=self.fig_num)

        for i in range(7):
            # reconstruct this ground-truth template item with the learned template
            coefs, this_model = fit_models_ols(D_rest_interpolated, templates_EAZY[i], np.ones_like(templates_EAZY[i]))
            ax[i].plot(lamb_um, templates_EAZY[i,:],'-', linewidth=1.5, label='EAZY templates')
            ax[i].plot(lamb_um, this_model, linewidth=1, label='Fitted with dictionaries', alpha=0.8)
            ax[i].set_ylabel('T'+str(i+1), fontsize=7)
            ax[i].grid(alpha=0.7)
            ax[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
            if i == 0:
                ax[i].legend(fontsize=8)

        fig.supxlabel(r'[$\mu m$]', fontsize=7)
        fig.suptitle('Reconstructing EAZY Template with dictionaries')
        fig.tight_layout()
        plt.savefig(self.output_dirname+'EAZY_reconstruction.png', dpi=self.dpi)
        self.fig_num += 1


# Diagnostic functions that works just as fit_zgrid but output all coefs instead of just best fit coefficient
def fit_zgrid_coefs(lamb_data, spec_data, err_data, lamb_D, D_rest=None, D_allz=None, zgrid=None, zinput=False, filter_info=None,
                larslasso=False, alpha=0, alpha_sig=0.0, lars_positive=False, alpha_ns_scaling=False, LARSlasso_alpha_selection_only=False,
                dz=0.002, zmin=0.0, zmax=3.0, scale_1plusz=True, error=False, probline=0.317/2, 
                conv=False, conv_finegrid=False, local_finegrid=False, local_finegrid_size=0.03, local_finegrid_dz=0.001):
    # Check if D_rest is necessary but not given
    if (zinput or local_finegrid) and D_rest is None:
        raise ValueError("D_rest is required but not given for zinput=True")
    if D_rest is None and D_allz is None:
        raise ValueError("Both D_rest and D_allz are not given")
    if D_rest is None and D_allz is not None:
        D_rest = np.zeros((8,lamb_D.shape[0])) # set D_rest to some float array so numba can fix D_rest type

    # if redshift is fixed and doesn't need error, have only single point in the redshift grid and turn off local_finegrid
    if zinput and not error:
        zinput_float = np.float64(zinput)
        zgrid = np.array([zinput_float])
        local_finegrid = False
        zbest0 = zinput_float
        D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filter_info=filter_info, conv=conv)
        D_allz = D_zinput.reshape((1, D_zinput.shape[0], D_zinput.shape[1]))
    
    else:
        # if zgrid is not given, create a default zgrid
        zmin = np.float64(zmin)
        zmax = np.float64(zmax)
        if zgrid is None:
            zgridlist = [] 
            zgi = zmin
            while zgi < zmax:
                zgridlist.append(zgi)
                if scale_1plusz:
                    zgi += dz * (1+zgi)
                else:
                    zgi += dz
            zgrid = np.array(zgridlist)    
        else:
            zgrid_limit = (zgrid<=zmax) & (zgrid>=zmin)
            zgrid = zgrid[zgrid_limit]

        # if zinput is still given, add z=zinput to zgrid
        if zinput:
            # unfortunately numba doesn't support np.insert
            zinput_float = np.float64(zinput)
            zgrid_copy = zgrid.copy()
            zgrid = np.hstack((zgrid_copy, np.array([zinput_float]))) # input of hstack in numba have to be all arrays
            idx_zgrid = np.argsort(zgrid)
            zgrid = zgrid[idx_zgrid]
            zbest0 = zinput_float
            idx_zbest0 = np.argmax(idx_zgrid)

        # check if D_allz is given
        if D_allz is None:
            D_allz = apply_redshift_all(D_rest, zgrid, lamb_D, lamb_data, filter_info=filter_info, conv=conv)
        elif zinput:
            # if D_allz and zinput are provided, create a new D_allz array and insert D_zinput to the correct location
            D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filter_info=filter_info, conv=conv)
            D_allz_copy = D_allz.copy()
            D_allz = np.zeros((D_allz_copy.shape[0]+1, D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
            D_allz[:D_allz_copy.shape[0]] = D_allz_copy
            D_allz[-1] = D_zinput
            D_allz = D_allz[idx_zgrid]

    # after above steps, zgrid and D_allz are ready for any given situation
    # prepare arrays to store coefficients and chi2
    coefs_zgrid = np.zeros((zgrid.shape[0], D_allz.shape[1]))
    chi2_zgrid = np.zeros_like(zgrid) + np.inf

    # loop over zgrid to calculate best-fit and chi2
    for i in range(zgrid.shape[0]):
        D_thisz = D_allz[i]
        if not larslasso:
            coefs, model = fit_models_ols(D_thisz, spec_data, err_data)
        else:
            coefs, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, 
                                                positive=lars_positive, unit_X=True, unit_y=True, 
                                                max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
        coefs_zgrid[i] = coefs
        chi2_zgrid[i] = np.sum((model - spec_data)**2/err_data**2)

    if not zinput:
        idx_zbest0 = np.argmin(chi2_zgrid)
        zbest0 = zgrid[idx_zbest0]

    if not local_finegrid:
        idx_zbest1 = idx_zbest0
        zbest1 = zbest0
    # local fine grid
    else:
        z_localmax = min((zbest0+local_finegrid_size, zmax))
        z_localmin = max((zbest0-local_finegrid_size, zmin))
        zgrid_local = np.arange(z_localmin, z_localmax, local_finegrid_dz)
        local_length = zgrid_local.shape[0]
        D_all_localz = apply_redshift_all(D_rest, zgrid_local, lamb_D, lamb_data, filter_info=filter_info, conv=conv_finegrid)
        coefs_zgrid_local = np.zeros((local_length, D_allz.shape[1]))
        chi2_zgrid_local = np.zeros_like(zgrid_local) + np.inf

        for j in range(local_length):
            D_this_localz = D_all_localz[j]
            if not larslasso:
                coefs, model = fit_models_ols(D_this_localz, spec_data, err_data)
            else:
                coefs, model = fit_model_larslasso(D_this_localz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, 
                                                    positive=lars_positive, unit_X=True, unit_y=True, 
                                                    max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                    LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
            coefs_zgrid_local[j] = coefs
            chi2_zgrid_local[j] = np.sum((model - spec_data)**2/err_data**2)

        # combine coefs, chi2, D_allz, zgrid with local arrays
        zgrid_copy = zgrid.copy()
        zgrid = np.hstack((zgrid_copy, zgrid_local))
        idx_zgrid_new = np.argsort(zgrid)
        zgrid = zgrid[idx_zgrid_new]

        coefs_zgrid_copy = coefs_zgrid.copy()
        chi2_zgrid_copy = chi2_zgrid.copy()
        coefs_zgrid = np.vstack((coefs_zgrid_copy, coefs_zgrid_local))
        chi2_zgrid = np.hstack((chi2_zgrid_copy, chi2_zgrid_local))
        D_allz_copy = D_allz.copy()
        D_allz = np.zeros((zgrid.shape[0], D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
        D_allz[:D_allz_copy.shape[0]] = D_allz_copy
        D_allz[D_allz_copy.shape[0]:] = D_all_localz
        # sort all the combine arrays
        coefs_zgrid = coefs_zgrid[idx_zgrid_new]
        chi2_zgrid = chi2_zgrid[idx_zgrid_new]
        D_allz = D_allz[idx_zgrid_new]

        if zinput:
            idx_zbest1 = np.where(idx_zgrid_new==idx_zbest0)[0][0]  # figure where zinput is
            zbest1 = zinput_float
        else:
            idx_zbest1 = np.argmin(chi2_zgrid)
            zbest1 = zgrid[idx_zbest1]

    # Now we have zbest1, idx_zbest1, zgrid, coefs_zgrid, chi2_zgrid
    if not error:
        zlower = zbest1
        zupper = zbest1
    else:
        zlower, zupper = error_estimation(zgrid=zgrid, chi2=chi2_zgrid, probline=probline)

    D_zbest = D_allz[idx_zbest1]
    coef_zbest = coefs_zgrid[idx_zbest1]
    # chi2_zbest = chi2_zgrid[idx_zbest1]
    model_zbest = D_zbest.T @ coef_zbest

    return zbest1, zlower, zupper, coefs_zgrid, model_zbest, zgrid, chi2_zgrid




# --------------------------------- OUTDATED ---------------------------------


# function to generate zgrid in steps
def generate_zgrid1(zgrid_seps, zgrid_stepsizes, z_fitting_max):
    zgrid = []
    for i in range(len(zgrid_seps)-1):
        zgrid1 = np.arange(zgrid_seps[i], zgrid_seps[i+1], zgrid_stepsizes[i])
        zgrid.extend(zgrid1)
    zgrid.append(z_fitting_max)
    zgrid = np.array(zgrid)
    return zgrid


# OUTDATED
# Read the input file and prepare them for dictionary learning
# def read_file(pathfile, Ndat, centering=False, error_method=0, SNR=np.inf, f_lambda_mode=False, 
#               add_fluctuations=False, flux_fluctuation_scaling=1.0):
#     """
#     Return: ztrue, lamb_obs, spec_obs, spec_obs_original, err_obs
#     """
#     data = np.load(pathfile)
#     # If the input file denotes whether it's flambda or fnu, read it
#     try:
#         data_is_flambda = data['save_flambda']
#     except:
#         data_is_flambda = True     # if input data doesn't contain this parameter, assume input catalog is f_lambda

#     ztrue = data['z']
#     ztrue = ztrue.flatten() # legacy feature, flatten in case input format is (Ngal, 1)
#     lamb_obs = data['wavelengths']
#     spec_obs = data['spectra']
#     try:
#         desi_flag = data['desi_flag']
#     except:
#         desi_flag = np.zeros(len(spec_obs))

#     if error_method == 0:
#         try:
#             data['error']
#             err_obs = data['error'] * flux_fluctuation_scaling
#             # SNR = 1
#         except:
#             err_obs = np.full(spec_obs.shape, fill_value=np.median(spec_obs))/SNR
#     elif error_method == 1:
#         err_obs = np.full(spec_obs.shape, fill_value=np.median(spec_obs))/SNR
#     elif error_method == 2:
#         err_obs = spec_obs/SNR

#     # If the dictionary learning code is running in f_lambda mode but input data is f_nu, convert data to f_lambda
#     if f_lambda_mode and not data_is_flambda:
#         print("Convert input from f_nu to f_lambda")
#         spec_obs = fnu2flambda(lamb_obs*10000, spec_obs)
#         err_obs = fnu2flambda(lamb_obs*10000, err_obs)
#     # if in f_nu mode but data is f_lambda, convert to f_nu
#     elif not f_lambda_mode and data_is_flambda:
#         print("Convert input from f_lambda to f_nu")
#         spec_obs = flambda2fnu(lamb_obs*10000, spec_obs)
#         err_obs = flambda2fnu(lamb_obs*10000, err_obs)

#     # Read the original spectra without noise
#     try:
#         spec_obs_original = data['spectra_original']
#     except: 
#         spec_obs_original = spec_obs.copy()

#     if add_fluctuations:
#         spec_obs = np.random.normal(spec_obs, err_obs)

#     ztrue = ztrue[:Ndat]
#     spec_obs = spec_obs[:Ndat]
#     spec_obs_original = spec_obs_original[:Ndat]
#     err_obs = err_obs[:Ndat]
#     desi_flag = desi_flag[:Ndat]

#     if centering:
#         spec_obs = spec_obs - np.mean(spec_obs, axis=1)[:,None]
#         spec_obs_original = spec_obs_original - np.mean(spec_obs_original, axis=1)[:,None]

#     return ztrue, lamb_obs, spec_obs, spec_obs_original, err_obs, desi_flag




# Adaptive grid search with increasing step size toward higher z
@jit(nopython=True, fastmath=True)
def fit_zgrid1(lamb_data, spec_data, err_data, lamb_D, D, zgrid, zinput=False, filter_info=None,
                 larslasso=False, alpha=0, alpha_sig=0.0, lars_positive=False, alpha_ns_scaling=False, LARSlasso_alpha_selection_only=False,
                 zgrid_searchsize=0.02, zgrid_errsearchsize=0.03, z_fitting_max=3.0, probline=0.317/2, 
                 conv_first=False, conv_last=False, error=False, local_finegrid=True):

    if not zinput:
        # consider each redshift from 0-2
        ztrial0 = zgrid.copy()
        # calculate residual at each redshift
        residual_vs_z0 = np.inf + np.zeros_like(ztrial0) # initialize to infinity
        # loop over trial redshifts
        for k in range(ztrial0.shape[0]):
            # make this redshifted template
            D_thisz = apply_redshift1(D,ztrial0[k],lamb_D,lamb_data, filter_info, conv=conv_first)

            if not larslasso:   # use OLS fitting
                params, model = fit_models_ols(D_thisz, spec_data, err_data)
            else:
                params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
                                                    unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)

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
        if local_finegrid:
            # calculate residual at each redshift
            # loop over trial redshifts
            for k in range(ztrial.shape[0]):
                # make this redshifted template
                D_thisz = apply_redshift1(D,ztrial[k],lamb_D,lamb_data, filter_info, conv=conv_last)

                if not larslasso:   # use OLS fitting
                    params, model = fit_models_ols(D_thisz, spec_data, err_data)
                else:
                    params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
                                                        unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
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
    D_thisz = apply_redshift1(D,z,lamb_D,lamb_data, filter_info, conv=conv_last)

    if not larslasso:   # use OLS fitting
        params, model = fit_models_ols(D_thisz, spec_data, err_data)
    else:
        params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
                                            unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
    # calculate the model for these parameters and this template
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
            D_thisz = apply_redshift1(D,ztrial1_low[k],lamb_D,lamb_data, filter_info, conv=conv_last)

            if not larslasso:   # use OLS fitting
                params, model = fit_models_ols(D_thisz, spec_data, err_data)
            else:
                params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
                                                    unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)

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
            D_thisz = apply_redshift1(D,ztrial1_high[k],lamb_D,lamb_data, filter_info, conv=conv_last)

            if not larslasso:   # use OLS fitting
                params, model = fit_models_ols(D_thisz, spec_data, err_data)
            else:
                params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
                                                    unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)                

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
def igplots(ig, lamb_obs, spec_obs, spec_obs_original, err_obs, lamb_rest, D_rest, zgrid, filter_info, ztrue, figsize=(10,12)):
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
        z,zl,zh,params,model,ztrials,residues = fit_zgrid(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
                                                            full_zgrid, filter_info, conv_first=conv, conv_last=conv, error=True, local_finegrid=False)
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
        z,zl,zh,params,model,ztrials,residues = fit_zgrid(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
                                                            zgrid, filter_info, conv_first=conv, conv_last=conv, error=True)
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
def igplots2(ig, lamb_obs, spec_obs, spec_obs_original, err_obs, lamb_rest, D_rest, zgrid, filter_info, ztrue, figsize=(10,12)):
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
    local_finegrids = [False, True]
    sub_titles = ['Full Grid', 'Optimized Grid']
    j_end = [1,2]

    model0 = None
    total0 = None

    for i in range(2):

        grid = grids[i]
        local_finegrid = local_finegrids[i]
        # full grid
        t1 = time.time()
        z,zl,zh,params,model,ztrials,residues = fit_zgrid(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
                                                            grid, filter_info, error=True, local_finegrid=local_finegrid)
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

