import numpy as np
import pandas as pd
from numba import jit, njit
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
import yaml
from pathlib import Path

c = 3e18

class Configs:
    def __init__(self, config=''):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        self.Catalog = config['Catalog']
        self.Dictionary = config['Dictionary']
        self.Algorithm = config['Algorithm']
        self.LARSlasso = config['LARSlasso']
        self.Fitting = config['Fitting']
        self.Zgrid = config['Zgrid']
        self.Directory_locations = config['Directory_locations']

        # Catalog configurations
        self.training_catalog = self.Catalog['training_catalog']
        self.validation_catalog = self.Catalog['validation_catalog']
        self.filter_central_wavelengths = self.Catalog['filter_central_wavelengths']
        self.Ndat = self.Catalog['Ndat_training']
        self.Ndat_validation = self.Catalog['Ndat_validation']
        self.Ncalibrators = self.Catalog['Ncalibrators']
        self.use_DESI_flag = self.Catalog['use_DESI_flag']
        self.calibrator_SNR = self.Catalog['calibrator_SNR']
        self.f_lambda_mode = self.Catalog['f_lambda_mode']

        # Dictionary input configurations
        self.dict_read_from_file = self.Dictionary['read_from_file']
        self.add_constant = self.Dictionary['add_constant']
        self.fix_dicts = self.Dictionary['Fix_dicts']
        self.Ndict = self.Dictionary['Ndict']
        self.num_EAZY_as_dict = self.Dictionary['num_EAZY_as_dict']
        self.dicts_fluctuation_scaling_const = self.Dictionary['dict_fluctuation_scaling_start']
        if type(self.dicts_fluctuation_scaling_const) == str:
            self.dicts_fluctuation_scaling_const = float(self.dicts_fluctuation_scaling_const)
        self.dict_fluctuation_scaling_base = self.Dictionary['dict_fluctuation_scaling_base']

        # Algorithm configurations
        self.training = self.Algorithm['Training']
        self.Nepoch = self.Algorithm['Nepoch']
        self.algorithm = self.Algorithm['update_algorithm']
        self.fix_z = self.Algorithm['fix_z']
        self.centering = self.Algorithm['Centering']
        self.AB_update_tolerance = self.Algorithm['AB_update_tolerance']
        self.max_AB_loops = self.Algorithm['max_update_loops']
        self.remove_old_ab_info = self.Algorithm['remove_old_ab_info']
        self.epochs_to_keep = self.Algorithm['epochs_to_keep']
        self.scale_past_data = self.Algorithm['scale_past_data']
        self.separate_training_weights = self.Algorithm['separate_training_weights']
        self.weights1 = self.Algorithm['weights1']
        self.weights2 = self.Algorithm['weights2']
        self.learning_rate0 = self.Algorithm['learning_rate0']
        self.learning_rate_cali = self.Algorithm['learning_rate_cali']
        # self.residue_factor = self.Algorithm['residue_factor']
        # self.AB_factor = self.Algorithm['AB_factor']

        # LARSlasso configurations
        self.larslasso = self.LARSlasso['LARSlasso']
        self.larslasso_alpha_train = self.LARSlasso['alpha_train']
        self.larslasso_alpha_sigma_train = self.LARSlasso['alpha_sigma_train']
        self.larslasso_alpha_fit = self.LARSlasso['alpha_fit']
        self.larslasso_alpha_sigma_fit = self.LARSlasso['alpha_sigma_fit']
        self.larslasso_positive = self.LARSlasso['positive']
        self.train_best_estimator = self.LARSlasso['train_best_estimator']
        self.fit_best_estimator = self.LARSlasso['fit_best_estimator']
        self.max_feature = self.LARSlasso['max_feature']
        self.active_OLS_training = self.LARSlasso['active_OLS_training']
        self.active_OLS_fitting = self.LARSlasso['active_OLS_fitting']
        self.center_Xy = self.LARSlasso['center_Xy']
        self.unit_X = self.LARSlasso['unit_X']
        self.unit_y = self.LARSlasso['unit_y']
        self.larslasso_alpha_scaling = self.LARSlasso['alpha_scaling']

        # Fitting configurations
        self.probline = self.Fitting['probline']
        self.fit_training_catalog = self.Fitting['fit_training_catalog']
        self.fit_initial_dicts = self.Fitting['fit_initial_dicts']
        self.convolve_filters = self.Fitting['convolve_filters']
        self.convolve_filter = self.convolve_filters[0] # choose to convolve templates with filter or not in the first stage of optimized grid search
        self.last_stage_convolve_filter = self.convolve_filters[1]   # whether to colvolve with filters in the last stage of grid search 
        self.fitting_convolve_filter = self.convolve_filters[2] # Whether to convolve with filters in the end when fitting final redshifts
        self.multiprocess = self.Fitting['multiprocess']
        self.mp_threads = self.Fitting['mp_threads']

        # Zgrid configurations
        self.zmax = self.Zgrid['zmax']
        self.zmin = self.Zgrid['zmin']
        self.dz = self.Zgrid['dz']
        self.scale_1plusz = self.Zgrid['scale_1plusz']
        self.local_finegrid = self.Zgrid['local_finegrid']
        self.local_finegrid_size = self.Zgrid['local_finegrid_size']
        self.local_finegrid_dz = self.Zgrid['local_finegrid_dz']
        self.testing_zgrid = self.Zgrid['testing_zgrid']
        if self.testing_zgrid:
            self.local_finegrid = True

        # Directory locations
        self.eazy_templates_location = self.Directory_locations['eazy_templates_location']
        self.filter_list = self.Directory_locations['filter_list']
        self.output_dirname = self.Directory_locations['OUTPUT']
        self.Plots_subfolder = self.Directory_locations['Plots_subfolder']
        self.parameters_report = self.Directory_locations['parameters_report']

        if self.filter_list is None:
            self.convolve_filter = False
            self.last_stage_convolve_filter = False
            self.fitting_convolve_filter = False

        # keywords for fit_zgrid function (training) has to match fit_zgrid input keywords
        self.fit_zgrid_training_kws = {
                            'larslasso': self.larslasso,
                            'alpha': self.larslasso_alpha_train,
                            'alpha_sig': self.larslasso_alpha_sigma_train,
                            'positive': self.larslasso_positive,
                            'best_estimator': self.train_best_estimator,
                            'max_feature': self.max_feature,
                            'active_ols': self.active_OLS_training,
                            'alpha_ns_scaling': self.larslasso_alpha_scaling,
                            'center_Xy': self.center_Xy,
                            'unit_X': self.unit_X,
                            'unit_y': self.unit_y,
                            'zmax': self.zmax,
                            'zmin': self.zmin,
                            'dz': self.dz,
                            'scale_1plusz': self.scale_1plusz,
                            'probline': self.probline,
                            'conv': self.convolve_filter,
                            'conv_finegrid': self.last_stage_convolve_filter,
                            'error': False,
                            'local_finegrid': self.local_finegrid,
                            'local_finegrid_size': self.local_finegrid_size,
                            'local_finegrid_dz': self.local_finegrid_dz
                            }

        self.fit_zgrid_validation_kws = self.fit_zgrid_training_kws.copy()
        self.fit_zgrid_validation_kws['error'] = True
        self.fit_zgrid_validation_kws['conv'] = self.fitting_convolve_filter
        self.fit_zgrid_validation_kws['conv_finegrid'] = self.fitting_convolve_filter
        self.fit_zgrid_validation_kws['active_ols'] = self.active_OLS_fitting
        self.fit_zgrid_validation_kws['best_estimator'] = self.fit_best_estimator
        self.fit_zgrid_validation_kws['alpha'] = self.larslasso_alpha_fit
        self.fit_zgrid_validation_kws['alpha_sig'] = self.larslasso_alpha_sigma_fit



class Catalog:
    def __init__(self, pathfile='', Ndat=None, istart=0, centering=False, max_SNR=np.inf, min_SNR=-np.inf, SNR_channel=51):
        # print('Read from file...')
        data = np.load(pathfile)

        ztrue = data['z'][istart:]
        lamb_obs = data['wavelengths']
        spec_obs = data['spectra'][istart:]
        err_obs = data['error'][istart:]
        snr = spec_obs/err_obs

        try:
            desi_flag = data['desi_flag'][istart:]
        except:
            desi_flag = np.zeros(len(spec_obs))
        try:
            spec_obs_original = data['spectra_original'][istart:]
        except: 
            spec_obs_original = spec_obs.copy()

        snr[np.isnan(snr)] = 0.0    # TEMP
        snr_mask = (snr[:,SNR_channel]>min_SNR) & (snr[:,SNR_channel]<max_SNR)
        ztrue = ztrue[snr_mask][:Ndat]
        spec_obs = spec_obs[snr_mask][:Ndat]
        spec_obs_original = spec_obs_original[snr_mask][:Ndat]
        err_obs = err_obs[snr_mask][:Ndat]
        desi_flag = desi_flag[snr_mask][:Ndat]
        snr = snr[snr_mask][:Ndat]

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
        self.snr_norm = np.linalg.norm(snr, axis=1)


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

# TODO
class Dictionary:
    def __init__(self, configfile='', dictfile='', lamb_rest=None, D_rest=None, Ndict=20, add_constant=False, centering=False,
                 fix_dicts=0, f_lambda_mode=False, max_AB_loops=10, AB_update_tolerance=1e-3):
        if configfile != '':
            if type(configfile) == str:
                config = Configs(config=config)
            elif isinstance(configfile, Configs):
                config = configfile
            dict_read_from_file = config.dict_read_from_file
            f_lambda_mode = config.f_lambda_mode
            fix_dicts = config.fix_dicts
            max_AB_loops = config.max_AB_loops
            AB_update_tolerance = config.AB_update_tolerance
            add_constant = config.add_constant
            centering = config.centering
            Ndict = config.Ndict
        elif configfile == '' and dictfile != '':
            dict_read_from_file = dictfile
            # use default values for all other keywords

        # Read dictionaries from file
        if dict_read_from_file:
            D_rest_input = np.load(dict_read_from_file)
            lamb_rest = D_rest_input['lamb_rest']
            D_rest = D_rest_input['D_rest']
            save_flambda = D_rest_input['save_flambda']
            del D_rest_input
            if Ndict >= len(D_rest):
                Ndict = len(D_rest)
            else:
                D_rest = D_rest[:Ndict]
            D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
            if f_lambda_mode and not save_flambda:
                D_rest = fnu2flambda(lamb_rest*1e5, D_rest)
                D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
            elif not f_lambda_mode and save_flambda:
                D_rest = flambda2fnu(lamb_rest*1e5, D_rest)
                D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
            if centering:
                D_rest = D_rest - np.mean(D_rest, axis=1)[:,None]
                D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
            if add_constant:
                D_rest = np.vstack((D_rest, np.ones_like(lamb_rest)/np.linalg.norm(np.ones_like(lamb_rest))))

        # save important variables to self; if not read from config or dictfile, save it directly from input args
        self.lamb_rest = lamb_rest
        self.D_rest = D_rest
        self.D_rest_initial = D_rest.copy()
        self.Ndict = D_rest.shape[0]
        # prepare A and B so they exist
        self.A = np.zeros((D_rest.shape[0], D_rest.shape[0]))
        self.B = np.zeros((len(lamb_rest), D_rest.shape[0]))
        self.A_history = None
        self.B_history = None
    
    # def learn_prepare(self, Ngal):
    #     self.A = 

    # def learn(self, i, )

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



# Initialize dictionary
def initialize_dicts(Ndict, dictionary_fluctuation_scaling, 
                     templates_EAZY=None, num_EAZY_as_dict=1, lamb_rest=np.arange(0.2,6.0,0.01), add_constant=False):
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
        # rand_dict_i = np.random.randn(len(lamb_rest))*dictionary_fluctuation_scaling[i]
        rand_dict_i = np.random.randn(len(lamb_rest))
        rand_dict_i += np.fabs(np.min(rand_dict_i))
        D_rest_list.append(rand_dict_i/np.linalg.norm(rand_dict_i))
    if add_constant:
        constant_dict = np.ones_like(lamb_rest)
        constant_dict = constant_dict / np.linalg.norm(constant_dict)
        D_rest_list.append(constant_dict)
    D_rest = np.vstack(tuple(D_rest_list))
    # note that the last row is a constant DC value to enable the code to fit out the average value
    # D_rest_initial = D_rest.copy()
    return D_rest


# Read SPHEREx filters
def read_filters(filter_list, half_length=105):
    filter_dir = Path(filter_list).parent
    filter_names = np.loadtxt(filter_list, dtype=str)
    Nf = filter_names.shape[0]
    filters = np.zeros((Nf, 2, half_length*2))

    for i in range(Nf):
        filter_path_name = filter_dir / Path(filter_names[i])
        filt_i = np.loadtxt(filter_path_name)
        wavelength_i = filt_i[:,0] * 1e-4   # convert from AA to micron
        response_i = filt_i[:,1]
        arg_peak = np.argmax(response_i)
        if arg_peak - half_length < 0:
            istart = 0
            ifinish = half_length*2
        elif arg_peak + half_length > len(wavelength_i):
            ifinish = len(wavelength_i)
            istart = ifinish - half_length*2
        else:
            istart = arg_peak - half_length
            ifinish = arg_peak + half_length
        wavelength_i1 = wavelength_i[istart:ifinish]
        response_i1 = response_i[istart:ifinish]
        tot_response_i1 = np.trapezoid(response_i1, wavelength_i1)
        response_i1 = response_i1 / tot_response_i1    # divide by total response now so that when convolving SED with filters no normalization is needed
        filters[i][0] = wavelength_i1
        filters[i][1] = response_i1
        # filters.append((wavelength_i1, response_i1))
    return filters

# # Read SPHEREx filters old
# def read_filters(filter_location):
#     """
#     Return: Nfilt, filt_length, filt_norm, filt_all_lams, filt_all_lams_reshaped
#     """
#     # if convolve_filter:
#     if filter_location is not None:
#         filter_dir = Path(filter_location)   
#     filtlams = []  # for storing filter wavelengths
#     for i in range(1,103):
#         if len(str(i)) == 1:
#             idx_str = '00'+str(i)
#         elif len(str(i)) == 2:
#             idx_str = '0'+str(i)
#         elif len(str(i)) == 3:
#             idx_str = str(i)
#         filt = pd.read_csv(filter_dir / f'spherex_paoyu_{idx_str}.txt', sep='\s+')
#         lams_i = filt['lambda']
#         filtlams.append(lams_i/10000) # convert to um
#     ftrans = filt['transmission'].to_numpy()   # All transmission looks the same, so just take the last one
#     filtlams = np.array(filtlams)
#     filt_norm = np.zeros(len(filtlams))

#     # Calculate filter normalization
#     # Also save all filters' wavelength points in a single 1-dimensional array
#     filt_all_lams = []
#     for i in range(len(filtlams)):
#         lb = filtlams[i]
#         filt_norm[i] = max(lb[ftrans>0])-min(lb[ftrans>0])
#         filt_all_lams.extend(lb[ftrans>0])
#     filt_all_lams = np.array(filt_all_lams)
#     filt_length = len(lb[ftrans>0])  # each filters length
#     Nfilt = len(filtlams)
#     filt_all_lams_reshaped = filt_all_lams.reshape((Nfilt, filt_length))

#     return Nfilt, filt_length, filt_norm, filt_all_lams, filt_all_lams_reshaped


# Load EAZY templates
def load_EAZY(lamb_um, eazy_templates_location):
    eazy_templates_dir = Path(eazy_templates_location)
    dfs = []
    for i in range(1,8):
        dfi = pd.read_csv(eazy_templates_dir / f'eazy_v1.1_sed{i}.dat', names=['lambda_ang','flux'], sep=r'\s+')
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
def f_convolve_filter(wl, flux, filters=None):
    if filters != None:
        Nf = len(filters)
        flux_conv = np.zeros(Nf)
        for i in range(Nf):
            lb = filters[i][0]
            ftrans = filters[i][1]
            f_interp = np.interp(lb, wl, flux)
            fnu_i = np.trapezoid(f_interp*ftrans, lb)
            flux_conv[i] = fnu_i
    else:
        flux_conv = flux    # TODO remove this part
    return flux_conv

# # function to integrate fluxes convolved with filters and return all 102 values, old
# @njit(fastmath=True)
# def f_convolve_filter(wl, flux, filters=None):
#     if filters != None:
#         Nfilt, filt_length, filt_norm, filt_all_lams, filt_all_lams_reshaped = filters
#         # flux_conv = np.zeros(102)
#         # for i in range(102):
#         #     lb = filtlams[i]
#         #     f_interp = np.interp(lb, wl, flux)
#         #     fnu_i = np.trapz(f_interp*ftrans, lb)/filt_norm[i]
#         #     flux_conv[i] = fnu_i

#         # manually calculate area below curves
#         f_interps = np.interp(filt_all_lams, wl, flux).reshape((Nfilt, filt_length))
#         # flux_conv = np.sum(f_interps, axis=1) / filt_length   # Numba doesn't support np.mean with arguments
#         df_interps = np.diff(f_interps)
#         dl = np.diff(filt_all_lams_reshaped)
#         f_ints = dl * (0.5*df_interps + f_interps[:,:-1])
#         flux_conv = np.sum(f_ints, axis=1)/filt_norm
#     else:
#         flux_conv = flux
#     return flux_conv


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
def apply_redshift1(D,z,lamb_in,lamb_out, filters=None, conv=False):
    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in * (1+z)
    if conv and (filters is not None):
        D_out = np.zeros((D.shape[0],len(lamb_out)))
        for i in range(D.shape[0]):
            D_conv = f_convolve_filter(lamb_inx1pz, D[i,:], filters)
            D_out[i,:] = D_conv
        # else:
        #     D_out[i,:] = np.interp(lamb_out,lamb_inx1pz,D[i,:])
    else:
        D_out = multiInterp1(lamb_out, lamb_inx1pz, D)
    return D_out

# apply redshift to all zgrid points and output 3-d array
@njit(fastmath=True)
def apply_redshift_all(D, zgrid, lamb_in, lamb_out, filters=None, conv=False):
    nz = zgrid.shape[0]
    D_all = np.zeros((nz, D.shape[0], lamb_out.shape[0]))
    for i in range(nz):
        D_thisz = apply_redshift1(D, zgrid[i], lamb_in, lamb_out, filters=filters, conv=conv)
        D_all[i] = D_thisz
    return D_all


    
# Fit data with dictionary across the whole redshift grid
@jit(nopython=True, fastmath=True)
def fit_zgrid(lamb_data, spec_data, err_data, lamb_D, D_rest=None, D_allz=None, zgrid=None, zinput=False, filters=None,
                larslasso=False, alpha=0, alpha_sig=0.0, positive=False, center_Xy=False, unit_X=True, unit_y=True, 
                path=False, best_estimator=None, max_feature=None, alpha_ns_scaling=False, active_ols=False,
                dz=0.002, zmin=0.0, zmax=3.0, scale_1plusz=True, error=False, probline=0.317/2, 
                conv=False, conv_finegrid=False, local_finegrid=False, local_finegrid_size=0.03, local_finegrid_dz=0.001):
    # Check if D_rest is necessary but not given
    if (zinput or local_finegrid) and D_rest is None:
        raise ValueError("D_rest is required but not given for zinput=True")
    if D_rest is None and D_allz is None:
        raise ValueError("Both D_rest and D_allz are not given")
    if D_rest is None and D_allz is not None:
        D_rest = np.zeros((8, lamb_D.shape[0])) # set D_rest to some float array so numba can fix D_rest type

    # if redshift is fixed and doesn't need error, have only single point in the redshift grid and turn off local_finegrid
    if zinput and not error:
        zinput_float = np.float64(zinput)
        zgrid = np.array([zinput_float])
        local_finegrid = False
        zpeak0 = zinput_float
        D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filters=filters, conv=conv)
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
            zpeak0 = zinput_float
            idx_zpeak0 = np.argmax(idx_zgrid)

        # check if D_allz is given
        if D_allz is None:
            D_allz = apply_redshift_all(D_rest, zgrid, lamb_D, lamb_data, filters=filters, conv=conv)
        elif zinput:
            # if D_allz and zinput are provided, create a new D_allz array and insert D_zinput to the correct location
            D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filters=filters, conv=conv)
            D_allz_copy = D_allz.copy()
            D_allz = np.zeros((D_allz_copy.shape[0]+1, D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
            D_allz[:D_allz_copy.shape[0]] = D_allz_copy
            D_allz[-1] = D_zinput
            D_allz = D_allz[idx_zgrid]

    # after above steps, zgrid and D_allz are ready for any given situation
    # prepare arrays to store coefficients and chi2
    coefs_zgrid = np.zeros((zgrid.shape[0], D_allz.shape[1]))
    bs_zgrid = np.zeros_like(zgrid)
    chi2_zgrid = np.zeros_like(zgrid) + np.inf

    # loop over zgrid to calculate best-fit and chi2
    for i in range(zgrid.shape[0]):
        D_thisz = D_allz[i]
        if not larslasso:
            coefs, model = fit_models_ols(D_thisz, spec_data, err_data)
            b = np.array(0.0)
            cost = np.sum((model - spec_data)**2/err_data**2)
        else:
            # D_thisz = np.ascontiguousarray(D_thisz)
            # spec_data = np.ascontiguousarray(spec_data)
            # err_data = np.ascontiguousarray(err_data)
            coefs, b, model, cost = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=positive, 
                                                center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator, 
                                                max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                max_feature=max_feature, active_ols=active_ols)
        coefs_zgrid[i] = coefs
        bs_zgrid[i] = b
        chi2_zgrid[i] = cost

    # if not zinput:
    idx_zpeak0 = np.argmin(chi2_zgrid)
    zpeak0 = zgrid[idx_zpeak0]

    if not local_finegrid:
        idx_zpeak1 = idx_zpeak0
        zpeak1 = zpeak0
    # local fine grid
    else:
        z_localmax = min((zpeak0+local_finegrid_size, zmax))
        z_localmin = max((zpeak0-local_finegrid_size, zmin))
        zgrid_local = np.arange(z_localmin, z_localmax, local_finegrid_dz)
        local_length = zgrid_local.shape[0]
        D_all_localz = apply_redshift_all(D_rest, zgrid_local, lamb_D, lamb_data, filters=filters, conv=conv_finegrid)
        coefs_zgrid_local = np.zeros((local_length, D_allz.shape[1]))
        bs_zgrid_local = np.zeros_like(zgrid_local)
        chi2_zgrid_local = np.zeros_like(zgrid_local) + np.inf

        for j in range(local_length):
            D_this_localz = D_all_localz[j]
            if not larslasso:
                coefs, model = fit_models_ols(D_this_localz, spec_data, err_data)
                b = np.array(0.0)
                cost = np.sum((model - spec_data)**2/err_data**2)
            else:
                # spec_data = np.ascontiguousarray(spec_data)
                # err_data = np.ascontiguousarray(err_data)
                coefs, b, model, cost = fit_model_larslasso(D_this_localz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=positive, 
                                                    center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator, 
                                                    max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                    max_feature=max_feature, active_ols=active_ols)
            coefs_zgrid_local[j] = coefs
            bs_zgrid_local[j] = b
            chi2_zgrid_local[j] = cost

        # combine coefs, chi2, D_allz, zgrid with local arrays
        zgrid_copy = zgrid.copy()
        zgrid = np.hstack((zgrid_copy, zgrid_local))
        idx_zgrid_new = np.argsort(zgrid)
        zgrid = zgrid[idx_zgrid_new]

        coefs_zgrid_copy = coefs_zgrid.copy()
        bs_zgrid_copy = bs_zgrid.copy()
        chi2_zgrid_copy = chi2_zgrid.copy()
        coefs_zgrid = np.vstack((coefs_zgrid_copy, coefs_zgrid_local))
        bs_zgrid = np.hstack((bs_zgrid_copy, bs_zgrid_local))
        chi2_zgrid = np.hstack((chi2_zgrid_copy, chi2_zgrid_local))
        D_allz_copy = D_allz.copy()
        D_allz = np.zeros((zgrid.shape[0], D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
        D_allz[:D_allz_copy.shape[0]] = D_allz_copy
        D_allz[D_allz_copy.shape[0]:] = D_all_localz
        # sort all the combine arrays
        coefs_zgrid = coefs_zgrid[idx_zgrid_new]
        bs_zgrid = bs_zgrid[idx_zgrid_new]
        chi2_zgrid = chi2_zgrid[idx_zgrid_new]
        D_allz = D_allz[idx_zgrid_new]

        if zinput:
            idx_zpeak1 = np.where(idx_zgrid_new==idx_zpeak0)[0][0]  # figure where zinput is
            zpeak1 = zinput_float
        else:
            idx_zpeak1 = np.argmin(chi2_zgrid)
            zpeak1 = zgrid[idx_zpeak1]

    # Now we have zpeak1, idx_zpeak1, zgrid, coefs_zgrid, chi2_zgrid
    if not error:
        zlower = zpeak1
        zupper = zpeak1
    else:
        zlower, zupper = error_estimation(zgrid=zgrid, chi2=chi2_zgrid, probline=probline)
        # if zlower > zpeak1:
        #     zlower = zpeak1
        # if zupper < zpeak1:
        #     zupper = zpeak1
    D_zpeak = D_allz[idx_zpeak1]
    coef_zpeak = coefs_zgrid[idx_zpeak1]
    b_zpeak = np.array(bs_zgrid[idx_zpeak1])
    # chi2_zpeak = chi2_zgrid[idx_zpeak1]
    model_zpeak = D_zpeak.T @ coef_zpeak

    # calculate averaged z_best
    if zinput:
        zbest = zinput_float
    else:
        min_chi2 = np.min(chi2_zgrid)
        likelihood = np.exp(-(chi2_zgrid-min_chi2)/2)
        area = np.trapezoid(likelihood, zgrid)
        # likelihood = likelihood/area
        zbest = np.trapezoid((likelihood*zgrid), zgrid)/area

    return zpeak1, zbest, zlower, zupper, coef_zpeak, b_zpeak, model_zpeak, zgrid, chi2_zgrid


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
    if idx_lower1 == -1:
        zlower = 0.0
    else:
        zlower1 = zgrid_area_L[idx_lower1]
        zlower2 = zgrid_area_L[idx_lower2]
        dz_lower = zlower2 - zlower1
        factorL = (probline - cum_d_area_L[idx_lower1]) / (cum_d_area_L[idx_lower2] - cum_d_area_L[idx_lower1])
        zlower = zlower1 + factorL * dz_lower


    cum_d_area_R = np.cumsum(d_area[::-1])    # cumulative sum of area from right
    idx_upper1 = np.argwhere((cum_d_area_R - probline)>0)[0][0] - 1
    idx_upper2 = idx_upper1 + 1
    if idx_upper1 == -1:
        zupper = zgrid[-1]
    else:
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
def fit_model_larslasso(D_thisz, spec_data, err_data, alpha=0.0, alpha_sig=0.0, positive=False,  
                        center_Xy=False, unit_X=True, unit_y=True, path=False, best_estimator=None,
                        max_iter=200, decimals=10, max_feature=None, alpha_ns_scaling=False, active_ols=False):
    X = (D_thisz/err_data).T    # currently D_thisz is X.T in usual LARSlasso convention
    X = np.ascontiguousarray(X)
    y = spec_data/err_data
    # y = np.ascontiguousarray(y)

    if center_Xy:    # calculate ynorm with centered y even if y is not centered because this determines alpha value during larslasso
        ynorm = np.linalg.norm(y-np.mean(y))
        Xmeans = np.sum(X, axis=0)/X.shape[0]
        Xnorms = np.sqrt(np.sum((X-Xmeans)**2, axis=0))    # numba doesn't support np.mean(X, axis=0)
    else:
        ynorm = np.linalg.norm(y)
        Xnorms = np.sqrt(np.sum(X**2, axis=0))
    if unit_X:
        Xnorms_inside = np.ones_like(Xnorms)
    else:
        Xnorms_inside = Xnorms
    if unit_y:
        ynorm_inside = 1.0
    else:
        ynorm_inside = ynorm
    d_corr = Xnorms_inside * 1.0 * ynorm_inside/ynorm
    
    alpha_in = alpha_sig*d_corr
    alpha_in[alpha_in<alpha] = alpha

    coefs, b, _, alpha_path, est_path = larslasso(X, y, alpha=alpha_in, positive=positive, 
                        center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator,
                        max_iter=max_iter, decimals=decimals, max_feature=max_feature, 
                        alpha_ns_scaling=alpha_ns_scaling, active_ols=active_ols)
    coefs = np.ascontiguousarray(coefs)
    
    ymodel = X @ coefs + b
    model = ymodel * err_data
    # TESTING
    if best_estimator is not None:
        alpha_out = np.full_like(alpha_in, alpha_path[np.argmin(est_path)])
    else:
        alpha_out = alpha_in
    # alpha_out = alpha_in
    cost = np.sum((y - ymodel)**2) + 2 * (ynorm/ynorm_inside) * (np.sum(np.fabs(alpha_out*Xnorms/Xnorms_inside*coefs)))
    # cost = np.sum((y - ymodel)**2)
    return coefs, b, model, cost

@jit(nopython=True, fastmath=True)
def larslasso(X, y, alpha=0.0, positive=False, center_Xy=False, unit_X=True, unit_y=True, path=False, best_estimator=None,
                max_iter=200, decimals=10, alpha_ns_scaling=False, max_feature=None, active_ols=False):
    '''
    Return LARS-lasso fitting coefficients. X is the atoms array, y is the target vector.

    Parameters
    ----------
    X : ndarray of shape (ns, nf)
        Atom array with ns samples and nf features

    y : ndarray of shape (ns, )
        Target vector with ns sample points to fit

    alpha : float or ndarray of shape (nf, ), default=0.0
        Regularization parameter for LARS-lasso algorithm. alpha>=0. Alpha=0 will yield OLS solution.
        The actual mathematical meaning of this parameter is the lowest required dot product value between atoms and target.
        If an ndarray is given, use each elemeht of alpha as minimum required correlation for each feature in X
    
    positive : bool, default=False
        If True, force all coefficients to be positive

    center_Xy : bool, default=False
        If True, center all the atoms in X and y

    unit_X : bool, default=True
        If False, do not normalize the input atoms
    
    unit_y : bool, default=True
        If False, do not normalize the input target vector

    path : bool, default=False
        If True, return coef, coef_path and alphas

    best_estimator : str, default=None
        Use best estimator fit as output, can be 'Cp' or 'BIC'

    best_bic : bool, default=False
        If True, output the best fit based on BIC before early stopping due to input alpha. Has priority over best_cp
        
    max_iter : int, default=200
        Max allowed iteration for the LARS-lasso algorithm

    decimals : int, default=10
        Number of decimals to round coefficients for stability

    alpha_ns_scaling: bool, default=False
        If True, alpha_i in each step will be alpha_i/ns 
        This is so that alpha is more consistent with lasso L1-regularization definition.

    max_feature: int, default=None
        Max allowed activated features before ending algorithm early
        If None, max_feature = nf

    active_ols: bool, default=False
        If True, only use alpha as active set selection and calculate ordinary least-square fit with active set when selction ends

    Returns
    -------
    coef : ndarray of shape (nf, )
        Fitted coefficients
    
    b : float
        Fitted incercept; the recovered model should be X @ coef + b
    
    coef_path : ndarray of shape (n_iters, nf)
        All coefficients along fitting path, will return np.array([[0.0]]) if path=False
    
    alpha_path : ndarray of shape(n_iters, )
        All alpha values along fitting path, will return np.array([0.0]) if path=False

    '''
    # X = np.ascontiguousarray(X)
    # y = np.ascontiguousarray(y)

    nf = X.shape[1]
    ns = X.shape[0]
    if max_feature is None:
        max_feature = nf

    if best_estimator is not None:
        path = True

    Xnorms = np.ones(nf)
    ynorm = 1.0
    Xmeans = np.zeros(nf)
    ymean = 0.0

    if center_Xy:
        Xmeans = np.sum(X, axis=0)/X.shape[0]    # numba doesn't support np.mean(X, axis=0)
        X = X - Xmeans
        ymean = np.mean(y)
        y = y - ymean
    if unit_X:
        Xnorms = np.sqrt(np.sum(X**2, axis=0))  # norm need to be re-calculated if centered
        X = X / Xnorms
    if unit_y:
        ynorm = np.linalg.norm(y)
        y = y / ynorm

    Cov = X.T @ y
    G = X.T @ X

    coef = np.zeros(nf)
    prev_coef = np.zeros(nf)
    alphas = np.array([0.0])
    prev_alphas = np.array([0.0])
    coef_path = coef.copy()[None,:]
    # alpha_path = alphas.copy()
    alpha_path = np.empty(shape=0)

    # check if input alpha is a single float number
    # set alpha_array for each feature, even if they are all the same
    # if best_cp:
        # alpha_array = np.zeros(nf)
    # else:
    alpha_array = alpha - np.zeros(nf)

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
        if path:
            alpha_path = np.hstack((alpha_path, alphas))

        if alphas < alpha_array[c_idx] and n_iter > 0:
            # print('End early')
            if not active_ols:
                ss = (prev_alphas - alpha_array) / (prev_alphas - alphas)
                coef = prev_coef + ss * (coef - prev_coef)
                alphas[0] = alpha_array[c_idx]
            else:
                coef_A = ols(X[:, idx_A_tf], y)
                coef[idx_A_tf] = coef_A
            if path:
                # coef_path = np.vstack((coef_path, coef))
                # coef_path_copy = coef_path.copy()
                # coef_path = np.zeros((coef_path_copy.shape[0]+1, coef_path_copy.shape[1]))
                # coef_path[:-1] = coef_path_copy
                # coef_path[-1] = coef
                coef_path[-1] = coef
                alpha_path[-1] = alpha_array[c_idx]
                # coef_path = np.append(coef_path, coef).reshape((n_iter+2, nf))
            # else:
                # coef_path = coef_path[None,:]   # add an axis so that dimension and type is compatible with coef_path when path=True
            break

        if n_iter >= max_iter or nA >= nf or nA >= max_feature:
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

        if path:
            # coef_path = np.vstack((coef_path, coef))
            # coef_path = np.append(coef_path, coef).reshape((n_iter+1, nf))
            coef_path_copy = coef_path.copy()
            coef_path = np.zeros((coef_path_copy.shape[0]+1, coef_path_copy.shape[1]))
            coef_path[:-1] = coef_path_copy
            coef_path[-1] = coef
        if drop:
            nA -= 1            
            temp = Cov_copy[idx_drop] - G_copy[idx_drop] @ coef
            Cov[idx_drop] = temp
            sign_active[idx_drop] = 0
            idx_A_tf[idx_drop] = False
            idx_rest_tf[idx_drop] = True

    if best_estimator:
        est_path = np.zeros_like(alpha_path)
        for i in range(est_path.shape[0]):
            coef_i = coef_path[i]
            coef_i = np.ascontiguousarray(coef_i)
            k = i   # LARS
            # k = np.sum(coef_i!=0)
            if best_estimator == 'Cp':
                est_path[i] = np.sum((y - X @ coef_i)**2/(1.0/ynorm)**2) - ns + 2*k
            elif best_estimator == 'BIC':
                est_path[i] = np.sum((y - X @ coef_i)**2/(1.0/ynorm)**2) + np.log(ns) * k

        best_est_idx = np.argmin(est_path[1:])+1    # in some case the 1st fit was bad, best_est will choose the coefs=0 case
        coef = coef_path[best_est_idx]
        # estimator_path = est_path

    coef = np.around(coef, decimals=decimals)
    # Calculate the intercept and recover the coef & b for original X and y
    b = np.array(ymean - (Xmeans/Xnorms) @ coef * ynorm)
    coef = coef / Xnorms * ynorm
    if path:
        coef_path = coef_path / Xnorms[None,:] * ynorm
        coef_path = np.around(coef_path, decimals=decimals)

    # model = X @ coef + b

    # return coef, b, coef_path, alpha_path
    return coef, b, coef_path, alpha_path, est_path  # TEMP
    # return coef, b
    # return coef


def nmad_eta(zs, zp, eta_method=0):
    nanmask = np.isnan(zp)
    zp = zp[~nanmask]
    zs = zs[~nanmask]
    dz = zp - zs
    dz_1pz = dz/(1+zs)
    nmad = 1.48 * np.median(np.abs((dz - np.median(dz))/(1+zs)))
    if eta_method == 0:
        eta = np.mean(np.abs(dz_1pz) > 0.15)
    elif eta_method == 1:
        std_eta = np.std(dz_1pz)
        eta = np.mean(np.abs(dz_1pz) > 3*std_eta)
    elif eta_method == 2:
        eta = np.mean(np.abs(dz_1pz) > 3*nmad)
    return nmad, eta




# Diagnostic functions that works just as fit_zgrid but output all coefs instead of just best fit coefficient
def fit_zgrid_coefs(lamb_data, spec_data, err_data, lamb_D, D_rest=None, D_allz=None, zgrid=None, zinput=False, filters=None,
                larslasso=False, alpha=0, alpha_sig=0.0, lars_positive=False, center_Xy=False, unit_X=True, unit_y=True,
                path=False, best_estimator=None, max_feature=None, alpha_ns_scaling=False, active_ols=False,
                dz=0.002, zmin=0.0, zmax=3.0, scale_1plusz=True, error=False, probline=0.317/2, 
                conv=False, conv_finegrid=False, local_finegrid=False, local_finegrid_size=0.03, local_finegrid_dz=0.001):
    # Check if D_rest is necessary but not given
    if (zinput or local_finegrid) and D_rest is None:
        raise ValueError("D_rest is required but not given for zinput=True")
    if D_rest is None and D_allz is None:
        raise ValueError("Both D_rest and D_allz are not given")
    if D_rest is None and D_allz is not None:
        D_rest = np.zeros((8, lamb_D.shape[0])) # set D_rest to some float array so numba can fix D_rest type

    # if redshift is fixed and doesn't need error, have only single point in the redshift grid and turn off local_finegrid
    if zinput and not error:
        zinput_float = np.float64(zinput)
        zgrid = np.array([zinput_float])
        local_finegrid = False
        zpeak0 = zinput_float
        D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filters=filters, conv=conv)
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
            zpeak0 = zinput_float
            idx_zpeak0 = np.argmax(idx_zgrid)

        # check if D_allz is given
        if D_allz is None:
            D_allz = apply_redshift_all(D_rest, zgrid, lamb_D, lamb_data, filters=filters, conv=conv)
        elif zinput:
            # if D_allz and zinput are provided, create a new D_allz array and insert D_zinput to the correct location
            D_zinput = apply_redshift1(D_rest, zinput_float, lamb_D, lamb_data, filters=filters, conv=conv)
            D_allz_copy = D_allz.copy()
            D_allz = np.zeros((D_allz_copy.shape[0]+1, D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
            D_allz[:D_allz_copy.shape[0]] = D_allz_copy
            D_allz[-1] = D_zinput
            D_allz = D_allz[idx_zgrid]

    # after above steps, zgrid and D_allz are ready for any given situation
    # prepare arrays to store coefficients and chi2
    coefs_zgrid = np.zeros((zgrid.shape[0], D_allz.shape[1]))
    bs_zgrid = np.zeros_like(zgrid)
    chi2_zgrid = np.zeros_like(zgrid) + np.inf

    # loop over zgrid to calculate best-fit and chi2
    for i in range(zgrid.shape[0]):
        D_thisz = D_allz[i]
        if not larslasso:
            coefs, model = fit_models_ols(D_thisz, spec_data, err_data)
            b = 0.0
            cost = np.sum((model - spec_data)**2/err_data**2)
        else:
            coefs, b, model, cost = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, 
                                                center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator,  
                                                max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                max_feature=max_feature, active_ols=active_ols)
        coefs_zgrid[i] = coefs
        bs_zgrid[i] = b
        chi2_zgrid[i] = cost

    # if not zinput:
    idx_zpeak0 = np.argmin(chi2_zgrid)
    zpeak0 = zgrid[idx_zpeak0]

    if not local_finegrid:
        idx_zpeak1 = idx_zpeak0
        zpeak1 = zpeak0
    # local fine grid
    else:
        z_localmax = min((zpeak0+local_finegrid_size, zmax))
        z_localmin = max((zpeak0-local_finegrid_size, zmin))
        zgrid_local = np.arange(z_localmin, z_localmax, local_finegrid_dz)
        local_length = zgrid_local.shape[0]
        D_all_localz = apply_redshift_all(D_rest, zgrid_local, lamb_D, lamb_data, filters=filters, conv=conv_finegrid)
        coefs_zgrid_local = np.zeros((local_length, D_allz.shape[1]))
        bs_zgrid_local = np.zeros_like(zgrid_local)
        chi2_zgrid_local = np.zeros_like(zgrid_local) + np.inf

        for j in range(local_length):
            D_this_localz = D_all_localz[j]
            if not larslasso:
                coefs, model = fit_models_ols(D_this_localz, spec_data, err_data)
                b = 0.0
                cost = np.sum((model - spec_data)**2/err_data**2)
            else:
                coefs, b, model, cost = fit_model_larslasso(D_this_localz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, 
                                                    center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator,  
                                                    max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, 
                                                    max_feature=max_feature, active_ols=active_ols)
            coefs_zgrid_local[j] = coefs
            bs_zgrid_local[j] = b
            chi2_zgrid_local[j] = cost

        # combine coefs, chi2, D_allz, zgrid with local arrays
        zgrid_copy = zgrid.copy()
        zgrid = np.hstack((zgrid_copy, zgrid_local))
        idx_zgrid_new = np.argsort(zgrid)
        zgrid = zgrid[idx_zgrid_new]

        coefs_zgrid_copy = coefs_zgrid.copy()
        bs_zgrid_copy = bs_zgrid.copy()
        chi2_zgrid_copy = chi2_zgrid.copy()
        coefs_zgrid = np.vstack((coefs_zgrid_copy, coefs_zgrid_local))
        bs_zgrid = np.hstack((bs_zgrid_copy, bs_zgrid_local))
        chi2_zgrid = np.hstack((chi2_zgrid_copy, chi2_zgrid_local))
        D_allz_copy = D_allz.copy()
        D_allz = np.zeros((zgrid.shape[0], D_allz_copy.shape[1], D_allz_copy.shape[2]), dtype=D_allz_copy.dtype)
        D_allz[:D_allz_copy.shape[0]] = D_allz_copy
        D_allz[D_allz_copy.shape[0]:] = D_all_localz
        # sort all the combine arrays
        coefs_zgrid = coefs_zgrid[idx_zgrid_new]
        bs_zgrid = bs_zgrid[idx_zgrid_new]
        chi2_zgrid = chi2_zgrid[idx_zgrid_new]
        D_allz = D_allz[idx_zgrid_new]

        if zinput:
            idx_zpeak1 = np.where(idx_zgrid_new==idx_zpeak0)[0][0]  # figure where zinput is
            zpeak1 = zinput_float
        else:
            idx_zpeak1 = np.argmin(chi2_zgrid)
            zpeak1 = zgrid[idx_zpeak1]

    # Now we have zpeak1, idx_zpeak1, zgrid, coefs_zgrid, chi2_zgrid
    if not error:
        zlower = zpeak1
        zupper = zpeak1
    else:
        zlower, zupper = error_estimation(zgrid=zgrid, chi2=chi2_zgrid, probline=probline)

    D_zpeak = D_allz[idx_zpeak1]
    coef_zpeak = coefs_zgrid[idx_zpeak1]
    b_zpeak = bs_zgrid[idx_zpeak1]
    # chi2_zpeak = chi2_zgrid[idx_zpeak1]
    model_zpeak = D_zpeak.T @ coef_zpeak

    # calculate averaged z_best
    if zinput:
        zbest = zinput_float
    else:
        min_chi2 = np.min(chi2_zgrid)
        likelihood = np.exp(-(chi2_zgrid-min_chi2)/2)
        area = np.trapezoid(likelihood, zgrid)
        # likelihood = likelihood/area
        zbest = np.trapezoid((likelihood*zgrid), zgrid)/area

    return zpeak1, zbest, zlower, zupper, coefs_zgrid, b_zpeak, model_zpeak, zgrid, chi2_zgrid




# --------------------------------- OUTDATED ---------------------------------



# class keywords:
#     def __init__(self, config=''):
#         with open(config, 'r') as file:
#             config = yaml.safe_load(file)
#         self.config = config
#         larslasso_kws = config['LARSlasso']
#         zgrid_kws = config['Zgrid']
#         self.larslasso_kws = larslasso_kws
#         self.zgrid_kws = zgrid_kws

#         zmax = zgrid_kws['zmax']
#         zmin = zgrid_kws['zmin']
#         dz = zgrid_kws['dz']
#         scale_1plusz = zgrid_kws['scale_1plusz']
#         local_finegrid = zgrid_kws['local_finegrid']
#         local_finegrid_size = zgrid_kws['local_finegrid_size']
#         local_finegrid_dz = zgrid_kws['local_finegrid_dz']
#         testing_zgrid = zgrid_kws['testing_zgrid']
#         if testing_zgrid:
#             local_finegrid = True

#         filter_location = config['Directory_locations']['filter_location']
#         if filter_location != 'None':
#             convolve_filters = config['Fitting']['convolve_filters']
#             convolve_filter = convolve_filters[0] # choose to convolve templates with filter or not in the first stage of optimized grid search
#             last_stage_convolve_filter = convolve_filters[1]   # whether to colvolve with filters in the last stage of grid search 
#             fitting_convolve_filter = convolve_filters[2]
#             self.fitting_convolve_filter = fitting_convolve_filter
#         else:
#             convolve_filter = False
#             last_stage_convolve_filter = False
#             fitting_convolve_filter = False
#             self.fitting_convolve_filter = fitting_convolve_filter
#         # baseline keywords
#         self.fit_spectrum_kws = {
#                             'larslasso': larslasso_kws['LARSlasso'],
#                             'alpha': larslasso_kws['alpha'],
#                             'alpha_sig': larslasso_kws['alpha_sigma'],
#                             'lars_positive': larslasso_kws['positive'],
#                             'max_feature': larslasso_kws['max_feature'],
#                             'active_ols': larslasso_kws['active_OLS_training'],
#                             'alpha_ns_scaling': larslasso_kws['alpha_scaling'],
#                             'center_Xy': larslasso_kws['center_Xy'],
#                             'zmax': zmax,
#                             'zmin': zmin,
#                             'dz': dz,
#                             'scale_1plusz': scale_1plusz,
#                             'probline': config['Fitting']['probline'],
#                             'conv': convolve_filter,
#                             'conv_finegrid': last_stage_convolve_filter,
#                             'error': False,
#                             'local_finegrid': local_finegrid,
#                             'local_finegrid_size': local_finegrid_size,
#                             'local_finegrid_dz': local_finegrid_dz
#                             }

#     def train_fit(self):
#         train_fit_kws = self.fit_spectrum_kws.copy()
#         return train_fit_kws
#     def validation_fit(self):
#         validation_fit_kws = self.fit_spectrum_kws.copy()
#         validation_fit_kws['error'] = True
#         validation_fit_kws['conv_finegrid'] = self.fitting_convolve_filter
#         validation_fit_kws['active_ols'] = self.larslasso_kws['active_OLS_fitting']
#         return validation_fit_kws
    

# function to generate zgrid in steps
# def generate_zgrid1(zgrid_seps, zgrid_stepsizes, z_fitting_max):
#     zgrid = []
#     for i in range(len(zgrid_seps)-1):
#         zgrid1 = np.arange(zgrid_seps[i], zgrid_seps[i+1], zgrid_stepsizes[i])
#         zgrid.extend(zgrid1)
#     zgrid.append(z_fitting_max)
#     zgrid = np.array(zgrid)
#     return zgrid


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



# # outdated
# # Adaptive grid search with increasing step size toward higher z
# @jit(nopython=True, fastmath=True)
# def fit_zgrid1(lamb_data, spec_data, err_data, lamb_D, D, zgrid, zinput=False, filters=None,
#                  larslasso=False, alpha=0, alpha_sig=0.0, lars_positive=False, alpha_ns_scaling=False, LARSlasso_alpha_selection_only=False,
#                  zgrid_searchsize=0.02, zgrid_errsearchsize=0.03, z_fitting_max=3.0, probline=0.317/2, 
#                  conv_first=False, conv_last=False, error=False, local_finegrid=True):

#     if not zinput:
#         # consider each redshift from 0-2
#         ztrial0 = zgrid.copy()
#         # calculate residual at each redshift
#         residual_vs_z0 = np.inf + np.zeros_like(ztrial0) # initialize to infinity
#         # loop over trial redshifts
#         for k in range(ztrial0.shape[0]):
#             # make this redshifted template
#             D_thisz = apply_redshift1(D,ztrial0[k],lamb_D,lamb_data, filters, conv=conv_first)

#             if not larslasso:   # use OLS fitting
#                 params, model = fit_models_ols(D_thisz, spec_data, err_data)
#             else:
#                 params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
#                                                     unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)

#             # calculate the RMS residual
#             residual_vs_z0[k] = np.sum((model - spec_data)**2/err_data**2)
        
#         # find the trial redshift with the lowest residual
#         kbest = int(np.where(residual_vs_z0 == np.min(residual_vs_z0))[0][0])
#         residues0 = residual_vs_z0.copy()    # save this residue for error estimation later
#         # note the redshift with the lowest residual
#         z = ztrial0[kbest]
#         # if kbest > 0:
#         #     zmin = ztrial0[kbest-1]

#         # create second round ztrial and residues regardless for output format purpose
#         zmin = z - zgrid_searchsize
#         zmax = z + zgrid_searchsize
#         if zmin < 0:
#             zmin = 0.0
#         if zmax > z_fitting_max:
#             zmax = z_fitting_max
#         ztrial = np.arange(zmin, zmax, 0.001)
#         residual_vs_z = np.inf + np.zeros_like(ztrial) # initialize to infinity
#         # second round
#         if local_finegrid:
#             # calculate residual at each redshift
#             # loop over trial redshifts
#             for k in range(ztrial.shape[0]):
#                 # make this redshifted template
#                 D_thisz = apply_redshift1(D,ztrial[k],lamb_D,lamb_data, filters, conv=conv_last)

#                 if not larslasso:   # use OLS fitting
#                     params, model = fit_models_ols(D_thisz, spec_data, err_data)
#                 else:
#                     params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
#                                                         unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
#                 # calculate the RMS residual
#                 residual_vs_z[k] = np.sum((model - spec_data)**2/err_data**2)
            
#             # find the trial redshift with the lowest residual
#             kbest = int(np.where(residual_vs_z == np.min(residual_vs_z))[0][0])
#             # note the redshift with the lowest residual
#             z = ztrial[kbest]        

#     else:
#         z = zinput
#         error = False
#     # redo the fit at this redshift
#     # make this redshifted template
#     D_thisz = apply_redshift1(D,z,lamb_D,lamb_data, filters, conv=conv_last)

#     if not larslasso:   # use OLS fitting
#         params, model = fit_models_ols(D_thisz, spec_data, err_data)
#     else:
#         params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
#                                             unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)
#     # calculate the model for these parameters and this template
#     model0 = model.copy()
#     if zinput:
#         residues0 = np.array([np.sum((model - spec_data)**2/err_data**2)])
#         residual_vs_z = np.array([np.sum((model - spec_data)**2/err_data**2)])

#     params0 = params.copy()
#     if not error:
#         return z,0.0,0.0,params0,model0,(ztrial0,ztrial),(residues0,residual_vs_z)

#     else:
#         min_res = np.min(residues0)
#         prob0 = np.exp(-(residues0-min_res)/2)
#         # calculate integration and normalize prob0
#         dz0 = np.diff(ztrial0)
#         dprob0 = np.diff(prob0)
#         d_area0 = dz0 * (prob0[:-1] + 0.5*dprob0)
#         total_prob0 = np.sum(d_area0)
#         prob0 = prob0/total_prob0
#         d_area0 = d_area0/total_prob0

#         c_d_area0 = np.cumsum(d_area0)  # cumulative area from z=0
#         # zlow_0 = ztrial0[np.argmin(np.abs(c_d_area0 - probline))]
#         zlow_0 = ztrial0[np.argwhere((c_d_area0 - probline)>0)[0][0]]
#         # reverse cumulative area
#         ztrial0_r = ztrial0[::-1]
#         c_d_area0_r = np.cumsum(d_area0[::-1])
#         # zhigh_0 = ztrial0_r[np.argmin(np.abs(c_d_area0_r - probline))+1]    # because it is reverse the index had to be added by 1
#         zhigh_0 = ztrial0_r[np.argwhere((c_d_area0_r - probline)>0)[0][0]-1]

#         # second round with more precision
#         zlow_zmin = zlow_0 - zgrid_errsearchsize
#         zlow_zmax = zlow_0 + zgrid_errsearchsize
#         if zlow_zmin < 0:
#             zlow_zmin = 0.0
#         if zlow_zmax > z_fitting_max:
#             zlow_zmax = z_fitting_max
#         ztrial1_low = np.arange(zlow_zmin, zlow_zmax, 0.001)
#         residual_vs_z_low = np.inf + np.zeros_like(ztrial1_low)
#         for k in range(ztrial1_low.shape[0]):
#             # make this redshifted template
#             D_thisz = apply_redshift1(D,ztrial1_low[k],lamb_D,lamb_data, filters, conv=conv_last)

#             if not larslasso:   # use OLS fitting
#                 params, model = fit_models_ols(D_thisz, spec_data, err_data)
#             else:
#                 params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
#                                                     unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)

#             # calculate the RMS residual
#             residual_vs_z_low[k] = np.sum((model - spec_data)**2/err_data**2)

#         zhigh_zmin = zhigh_0 - zgrid_errsearchsize
#         zhigh_zmax = zhigh_0 + zgrid_errsearchsize
#         if zhigh_zmin < 0:
#             zhigh_zmin = 0.0
#         if zhigh_zmax > z_fitting_max:
#             zhigh_zmax = z_fitting_max
#         ztrial1_high = np.arange(zhigh_zmin, zhigh_zmax, 0.001)
#         residual_vs_z_high = np.inf + np.zeros_like(ztrial1_high)
#         for k in range(ztrial1_high.shape[0]):
#             # make this redshifted template
#             D_thisz = apply_redshift1(D,ztrial1_high[k],lamb_D,lamb_data, filters, conv=conv_last)

#             if not larslasso:   # use OLS fitting
#                 params, model = fit_models_ols(D_thisz, spec_data, err_data)
#             else:
#                 params, model = fit_model_larslasso(D_thisz, spec_data, err_data, alpha=alpha, alpha_sig=alpha_sig, positive=lars_positive, unit_X=True, 
#                                                     unit_y=True, max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, LARSlasso_alpha_selection_only=LARSlasso_alpha_selection_only)                

#             # calculate the RMS residual
#             residual_vs_z_high[k] = np.sum((model - spec_data)**2/err_data**2)

#         # combine residues[0] with two more residue arrays
#         ztrial1_unsorted = np.concatenate((ztrial0, ztrial1_low, ztrial1_high))

#         sort_idx = np.argsort(ztrial1_unsorted)
#         ztrial1 = ztrial1_unsorted[sort_idx]
#         residues1 = np.concatenate((residues0, residual_vs_z_low, residual_vs_z_high))
#         min_res = np.min(residues1)
#         prob1 = np.exp(-(residues1[sort_idx]-min_res)/2)
#         dz1 = np.diff(ztrial1)
#         dprob1 = np.diff(prob1)
#         d_area1 = dz1 * (prob1[:-1] + 0.5*dprob1)
#         total_prob1 = np.sum(d_area1)
#         # total_prob1 = np.trapz(prob1, x=ztrial1)
#         prob1 = prob1/total_prob1
#         d_area1 = d_area1/total_prob1

#         c_d_area1 = np.cumsum(d_area1)
#         # zlow_1 = ztrial1[np.argmin(np.abs(c_d_area1 - probline))]
#         zlow_1 = ztrial1[np.argwhere((c_d_area1 - probline)>0)[0][0]]

#         # reverse cumulative area
#         ztrial1_r = ztrial1[::-1]
#         c_d_area1_r = np.cumsum(d_area1[::-1])
#         # zhigh_1 = ztrial1_r[np.argmin(np.abs(c_d_area1_r - probline))+1]
#         zhigh_1 = ztrial1_r[np.argwhere((c_d_area1_r - probline)>0)[0][0]-1]

#         return z,zlow_1,zhigh_1,params0,model0,(ztrial0,ztrial),(residues0,residual_vs_z)


# # residue and likelihood plots for an individual galaxy
# def igplots(ig, lamb_obs, spec_obs, spec_obs_original, err_obs, lamb_rest, D_rest, zgrid, filters, ztrue, figsize=(10,12)):
#     fig, ax = plt.subplots(4,2, figsize=figsize, num=0)

#     # delete ax[0,1]
#     # fig.delaxes(ax[0,1])
#     # adjust ax[0,0] to center
#     # ax[0,0].set_position([0.125, 0.75, 0.775, 0.15])
#     ax[0,0].errorbar(lamb_obs, spec_obs_original[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1)
#     ax[0,0].errorbar(lamb_obs, spec_obs[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1, alpha=0.8)

#     # share x axis
#     rows = [1,2,3]
#     cols = [0,1]
#     for ri in range(len(rows)):
#         for ci in range(len(cols)):
#             ax[rows[ri], cols[ci]].sharex(ax[1,0])

#     # no convolution runs
#     convs = [False, True]
#     full_zgrid = np.arange(min(zgrid,), max(zgrid), 0.001)
#     model0 = None

#     for i in range(2):

#         conv = convs[i]
#         # full grid
#         t1 = time.time()
#         z,zl,zh,params,model,ztrials,residues = fit_zgrid(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
#                                                             full_zgrid, filters, conv_first=conv, conv_last=conv, error=True, local_finegrid=False)
#         if i == 0:
#             model0 = model.copy()

#         t2 = time.time()
#         min_res = np.max((np.min(residues[0]),np.min(residues[1])))
#         ymin = (min(residues[0])-max(residues[0]))/10
#         ymax = max(residues[0])*1.2

#         ax[1,i].plot(ztrials[0],residues[0],'.-',linewidth=1,markersize=2)
#         ax[1,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
#         ax[1,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
#         ax[1,i].set_ylim(ymin,ymax)
#         ax[1,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s, conv={conv}")
#         ax[1,i].grid()
#         ax[1,i].set_ylabel(r'Full grid $\chi^2$')    

#         # convert to likelihood
#         prob0 = np.exp(-(residues[0]-min_res)/2)
#         # normalization
#         total = np.trapz(prob0, x=ztrials[0])
#         # total = 1.0
#         prob0 = prob0/total
#         ymin=(min(prob0)-max(prob0))/10
#         ymax=max(prob0)*1.2

#         ax[2,i].plot(ztrials[0],prob0,'.-',linewidth=1,markersize=2)
#         ax[2,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
#         ax[2,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
#         ax[2,i].vlines(x=zl,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
#         ax[2,i].vlines(x=zh,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
#         ax[2,i].set_ylim(ymin,ymax)
#         ax[2,i].set_title(rf"$zp={z:.3f}^{{+{zh-z:.3f}}}_{{-{z-zl:.3f}}}$, zh-zl={zh-zl:.3f}")
#         ax[2,i].grid()
#         ax[2,i].set_ylabel('Full grid likelihood')

#         # optimized zgrid
#         t1 = time.time()
#         z,zl,zh,params,model,ztrials,residues = fit_zgrid(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
#                                                             zgrid, filters, conv_first=conv, conv_last=conv, error=True)
#         t2 = time.time()

#         for j in range(len(ztrials)):
#             ztrial1 = ztrials[j]
#             prob1 = np.exp(-(residues[j]-min_res)/2)/total
#             ax[3,i].plot(ztrial1,prob1,'.-',linewidth=1,markersize=2)
#         ax[3,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
#         ax[3,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
#         ax[3,i].vlines(x=zl,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
#         ax[3,i].vlines(x=zh,ymin=ymin,ymax=ymax,alpha=0.5,color='teal', linestyle='--')
#         ax[3,i].set_ylim(ymin,ymax)
#         ax[3,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s, conv={conv}")
#         ax[3,i].grid()
#         ax[3,i].set_ylabel('Optimized grid likelihood')

#     ax[0,0].plot(lamb_obs, model0, '-', markersize=5, linewidth=1, marker='s', fillstyle='none', c='teal')

#     fig.supxlabel('ztrials')
#     fig.suptitle(f"i={ig}, ztrue={ztrue[ig][0]:.5f}")
    
#     fig.tight_layout()

#     pos00 = ax[0,0].get_position()
#     pos01 = ax[0,1].get_position()

#     fig.delaxes(ax[0,1])
#     ax[0,0].set_position([pos00.x0, pos00.y0, pos01.x1-pos00.x0, pos00.y1-pos00.y0])
#     # plt.show()
#     # plt.close(fig)




# # residue and likelihood plots for an individual galaxy
# # this one doesn't output convolved plots
# def igplots2(ig, lamb_obs, spec_obs, spec_obs_original, err_obs, lamb_rest, D_rest, zgrid, filters, ztrue, figsize=(10,12)):
#     fig, ax = plt.subplots(3,2, figsize=figsize, num=0)

#     # delete ax[0,1]
#     # fig.delaxes(ax[0,1])
#     # adjust ax[0,0] to center
#     # ax[0,0].set_position([0.125, 0.75, 0.775, 0.15])
#     ax[0,0].errorbar(lamb_obs, spec_obs_original[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1)
#     ax[0,0].errorbar(lamb_obs, spec_obs[ig], err_obs[ig], capsize=3,fmt='.-',markersize=5,elinewidth=0.5,linewidth=1, alpha=0.8)

#     # share x axis
#     rows = [1,2]
#     cols = [0,1]
#     for ri in range(len(rows)):
#         for ci in range(len(cols)):
#             ax[rows[ri], cols[ci]].sharex(ax[1,0])
#     ax[1,0].sharey(ax[1,1])
#     ax[2,0].sharey(ax[2,1])

#     full_zgrid = np.arange(min(zgrid,), max(zgrid), 0.001)
#     grids = [full_zgrid, zgrid]
#     local_finegrids = [False, True]
#     sub_titles = ['Full Grid', 'Optimized Grid']
#     j_end = [1,2]

#     model0 = None
#     total0 = None

#     for i in range(2):

#         grid = grids[i]
#         local_finegrid = local_finegrids[i]
#         # full grid
#         t1 = time.time()
#         z,zl,zh,params,model,ztrials,residues = fit_zgrid(lamb_obs, spec_obs[ig], err_obs[ig], lamb_rest, D_rest, 
#                                                             grid, filters, error=True, local_finegrid=local_finegrid)
#         if i == 0:
#             model0 = model.copy()

#         t2 = time.time()
#         if i == 0:
#             min_res = np.min(residues[0])
#         elif i == 1:
#             min_res = np.min((np.min(residues[0]),np.min(residues[1])))

#         ymin = (min(residues[0])-max(residues[0]))/10
#         ymax = max(residues[0])*1.2

#         for j in range(j_end[i]):
#             ax[1,i].plot(ztrials[j],residues[j],'.-',linewidth=1,markersize=2)
#         ax[1,i].vlines(x=ztrue[ig][0],ymin=ymin,ymax=ymax,alpha=0.6)
#         ax[1,i].vlines(x=z,ymin=ymin,ymax=ymax,alpha=0.6,color='salmon')
#         ax[1,i].set_ylim(ymin,ymax)
#         ax[1,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s")
#         ax[1,i].grid()
#         ax[1,i].set_ylabel(rf'{sub_titles[i]} $\chi^2$')    


#         if i == 0:
#             prob0 = np.exp(-(residues[0]-min_res)/2)
#             # normalization
#             total0 = np.trapz(prob0, x=ztrials[0])
#             prob0 = prob0/total0
#             pmin = (min(prob0)-max(prob0))/10
#             pmax = max(prob0)*1.2


#         for j in range(j_end[i]):
#             ztrial1 = ztrials[j]
#             prob1 = np.exp(-(residues[j]-min_res)/2)
#             prob1 = prob1/total0
#             ax[2,i].plot(ztrial1,prob1,'.-',linewidth=1,markersize=2)
#         ax[2,i].vlines(x=ztrue[ig][0],ymin=pmin,ymax=pmax,alpha=0.6)
#         ax[2,i].vlines(x=z,ymin=pmin,ymax=pmax,alpha=0.6,color='salmon')
#         ax[2,i].vlines(x=zl,ymin=pmin,ymax=pmax,alpha=0.5,color='teal', linestyle='--')
#         ax[2,i].vlines(x=zh,ymin=pmin,ymax=pmax,alpha=0.5,color='teal', linestyle='--')
#         ax[2,i].set_ylim(pmin,pmax)
#         ax[2,i].set_title(f"zp={z:.3f}, dz={z-ztrue[ig][0]:.4f}, time={t2-t1:.4f}s")
#         ax[2,i].grid()
#         ax[2,i].set_ylabel(f'{sub_titles[i]} likelihood')


#     ax[0,0].plot(lamb_obs, model0, '-', markersize=5, linewidth=1, marker='s', fillstyle='none', c='teal')

#     fig.supxlabel('ztrials')
#     fig.suptitle(f"i={ig}, ztrue={ztrue[ig][0]:.5f}")
    
#     fig.tight_layout()

#     pos00 = ax[0,0].get_position()
#     pos01 = ax[0,1].get_position()

#     fig.delaxes(ax[0,1])
#     ax[0,0].set_position([pos00.x0, pos00.y0, pos01.x1-pos00.x0, pos00.y1-pos00.y0])
#     # plt.show()
#     # plt.close(fig)

