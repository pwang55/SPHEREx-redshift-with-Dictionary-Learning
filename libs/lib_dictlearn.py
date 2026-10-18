import numpy as np
import pandas as pd
from numba import jit, njit
# from numba.typed import List
# import matplotlib.pyplot as plt
# import time
from types import SimpleNamespace
import yaml
from pathlib import Path
import h5py
from scipy.ndimage import gaussian_filter1d, median_filter
from . import functions as func
import multiprocessing as mp

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
        self.calibrator_ynorm = self.Catalog['calibrator_ynorm']
        self.f_lambda_mode = self.Catalog['f_lambda_mode']

        # Dictionary input configurations
        self.dict_read_from_file = self.Dictionary['read_from_file']
        self.dict_sys_from_file = self.Dictionary['sys_from_file']
        self.lamb_rest_step = self.Dictionary['initialize_dicts_steps']
        self.add_constant = self.Dictionary['add_constant']
        self.Ndict_sys = self.Dictionary['Ndict_sys']
        self.fix_dicts_sys = self.Dictionary['Fix_dicts_sys']
        self.fix_dicts = self.Dictionary['Fix_dicts']
        self.Ndict = self.Dictionary['Ndict']
        self.num_EAZY_as_dict = self.Dictionary['num_EAZY_as_dict']
        # self.dicts_fluctuation_scaling_const = self.Dictionary['dict_fluctuation_scaling_start']
        # if type(self.dicts_fluctuation_scaling_const) == str:
            # self.dicts_fluctuation_scaling_const = float(self.dicts_fluctuation_scaling_const)
        # self.dict_fluctuation_scaling_base = self.Dictionary['dict_fluctuation_scaling_base']

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
        self.correct_nodata_offset = self.Algorithm['correct_nodata_offset']
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
        # if type(self.larslasso_alpha_train) == list:
            # self.larslasso_alpha_train = np.array(self.larslasso_alpha_train)
        self.larslasso_alpha_sigma_train = self.LARSlasso['alpha_sigma_train']
        self.larslasso_alpha_fit = self.LARSlasso['alpha_fit']
        # if type(self.larslasso_alpha_fit) == list:
            # self.larslasso_alpha_fit = np.array(self.larslasso_alpha_fit)
        self.larslasso_alpha_sigma_fit = self.LARSlasso['alpha_sigma_fit']
        self.larslasso_positive = self.LARSlasso['positive']
        self.train_best_estimator = self.LARSlasso['train_best_estimator']
        self.fit_best_estimator = self.LARSlasso['fit_best_estimator']
        self.max_feature = self.LARSlasso['max_feature']
        self.active_alpha_0_training = self.LARSlasso['active_alpha_0_training']
        self.active_alpha_0_fitting = self.LARSlasso['active_alpha_0_fitting']
        self.active_alpha_0_fitting_penalty = self.LARSlasso['active_alpha_0_fitting_penalty']
        self.active_OLS_training = self.LARSlasso['active_OLS_training']
        self.active_OLS_fitting = self.LARSlasso['active_OLS_fitting']
        self.center_Xy = self.LARSlasso['center_Xy']
        self.unit_X = self.LARSlasso['unit_X']
        self.unit_y = self.LARSlasso['unit_y']
        self.larslasso_alpha_scaling = self.LARSlasso['alpha_scaling']

        # Fitting configurations
        self.fitting = self.Fitting['fitting']
        self.marginalization = self.Fitting['marginalization']
        self.marginalized_model = self.Fitting['marginalized_model']
        self.marginalization_min_alpha = self.Fitting['marginalization_min_alpha']
        self.marginalization_max_alpha = self.Fitting['marginalization_max_alpha']
        self.marginalize_active_only = self.Fitting['marginalize_active_only']
        self.partial_marginalization_min_alpha = self.Fitting['partial_marginalization_min_alpha']
        self.partial_marginalization_max_alpha = self.Fitting['partial_marginalization_max_alpha']
        self.cdf_tol = self.Fitting['cdf_tol']
        self.cdf_max_steps = self.Fitting['cdf_max_steps']
        self.probline = self.Fitting['probline']
        self.fit_training_catalog = self.Fitting['fit_training_catalog']
        self.fit_initial_dicts = self.Fitting['fit_initial_dicts']
        self.median_ynorm_train = self.Fitting['median_ynorm_train']
        self.median_ynorm_fit = self.Fitting['median_ynorm_fit']
        self.median_spectra_train = self.Fitting['median_spectra_train']
        self.median_spectra_fit = self.Fitting['median_spectra_fit']
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
                            # 'Ndict_sys': self.Ndict_sys,
                            'larslasso': self.larslasso,
                            'alpha': self.larslasso_alpha_train,
                            'alpha_sig': self.larslasso_alpha_sigma_train,
                            'positive': self.larslasso_positive,
                            'best_estimator': self.train_best_estimator,
                            'max_feature': self.max_feature,
                            'active_ols': self.active_OLS_training,
                            'active_alpha_0': self.active_alpha_0_training,
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
                            'local_finegrid_dz': self.local_finegrid_dz,
                            'median_ynorm': self.median_ynorm_train,
                            'median_spectra': self.median_spectra_train
                            }

        self.fit_zgrid_validation_kws = self.fit_zgrid_training_kws.copy()
        self.fit_zgrid_validation_kws['error'] = True
        self.fit_zgrid_validation_kws['conv'] = self.fitting_convolve_filter
        self.fit_zgrid_validation_kws['conv_finegrid'] = self.fitting_convolve_filter
        self.fit_zgrid_validation_kws['active_ols'] = self.active_OLS_fitting
        self.fit_zgrid_validation_kws['active_alpha_0'] = self.active_alpha_0_fitting
        self.fit_zgrid_validation_kws['best_estimator'] = self.fit_best_estimator
        self.fit_zgrid_validation_kws['alpha'] = self.larslasso_alpha_fit
        self.fit_zgrid_validation_kws['alpha_sig'] = self.larslasso_alpha_sigma_fit
        self.fit_zgrid_validation_kws['median_ynorm'] = self.median_ynorm_fit
        self.fit_zgrid_validation_kws['median_spectra'] = self.median_spectra_fit
        self.fit_zgrid_validation_kws['active_alpha_0_fitting_penalty'] = self.active_alpha_0_fitting_penalty

        self.marginalize_zgrid_validation_kws = {
            'alpha': self.larslasso_alpha_fit,
            'alpha_sig': self.larslasso_alpha_sigma_fit,
            'median_ynorm': self.median_ynorm_fit,
            'median_spectra': self.median_spectra_fit,
            'zmax': self.zmax,
            'zmin': self.zmin,
            'dz': self.dz,
            'scale_1plusz': self.scale_1plusz,
            'probline': self.probline,
            'conv': self.convolve_filter,
            'zoutput': self.marginalized_model,
            'cdf_tol': self.cdf_tol,
            'cdf_max_steps': self.cdf_max_steps
        }

class Catalog:
    def __init__(self, pathfile='', lambdafile='', Ndat=None, istart=0, centering=False, min_ynorm=None, sort_ynorm=False):
        # print('Read from file...')
        filetype = pathfile.split('.')[-1]
        if filetype == 'npz':
            data = np.load(pathfile)
            ztrue = data['z'][istart:]
            lamb_obs = data['wavelengths']
            spec_obs = data['spectra'][istart:]
            err_obs = data['error'][istart:]
            # try:
            spherex_id = data['SPHERExRefID'][istart:]

        elif filetype == 'parq' or filetype == 'parquet':
            data = pd.read_parquet(pathfile)[istart:]
            try:
                ztrue = data['z'].to_numpy()
            except:
                ztrue = data['z_specz'].to_numpy()

            # lamb_obs
            try:    # TEMP
                spec_obs = np.stack(data['fiducial_flux'])
                err_obs = np.stack(data['fiducial_flux_err'])
            except:
                spec_obs = np.stack(data['flux_dered_fiducial'])
                err_obs = np.stack(data['flux_err_dered_fiducial'])                
            spherex_id = np.array(data['SPHERExRefID'])
            try:
                lamb_data = np.load(lambdafile)
                lamb_obs = lamb_data['lambda_fiducial']
            except:
                try:
                    lamb_obs = data['lambda_fiducial'].iloc[0]
                except:
                    lamb_obs = None

        snr = spec_obs/err_obs

        try:
            desi_flag = data['desi_flag'][istart:]
        except:
            desi_flag = np.zeros(len(spec_obs))
        try:
            spec_obs_original = data['spectra_original'][istart:]
        except: 
            spec_obs_original = spec_obs.copy()

        if Ndat is None:
            Ndat = len(ztrue)
        snr[np.isnan(snr)] = 0.0
        # snr_mask = (snr[:,SNR_channel]>min_SNR) & (snr[:,SNR_channel]<max_SNR)
        snr_norm = np.linalg.norm(snr, axis=1)
        if min_ynorm is not None:
            snr_mask = snr_norm > min_ynorm
        else:
            snr_mask = np.full(shape=len(snr_norm), fill_value=True)

        ztrue = ztrue[snr_mask][:Ndat]
        spec_obs = spec_obs[snr_mask][:Ndat]
        spec_obs_original = spec_obs_original[snr_mask][:Ndat]
        err_obs = err_obs[snr_mask][:Ndat]
        desi_flag = desi_flag[snr_mask][:Ndat]
        snr = snr[snr_mask][:Ndat]
        snr_norm = snr_norm[snr_mask][:Ndat]
        spherex_id = spherex_id[snr_mask][:Ndat]

        if centering:
            spec_obs = spec_obs - np.mean(spec_obs, axis=1)[:,None]
            spec_obs_original = spec_obs_original - np.mean(spec_obs_original, axis=1)[:,None]

        if sort_ynorm:
            idx_sort = np.argsort(snr_norm)[::-1]
        else:
            idx_sort = np.arange(len(ztrue))

        self.ztrue = ztrue[idx_sort]
        self.z = ztrue[idx_sort]
        self.lamb_obs = lamb_obs
        self.spec_obs = spec_obs[idx_sort]
        self.spec_obs_original = spec_obs_original[idx_sort]
        self.err_obs = err_obs[idx_sort]
        self.desi_flag = desi_flag[idx_sort]
        self.snr = snr[idx_sort]
        self.spherex_id = spherex_id[idx_sort]
        self.idx_sort = idx_sort

        self.pathfile = pathfile
        self.centering = centering
        self.Ndat = Ndat
        # self.max_SNR = max_SNR
        # self.min_SNR = min_SNR
        # self.SNR_channel = SNR_channel
        # self.snr_i = snr[:,51]
        self.snr_norm = snr_norm[idx_sort]
        self.ynorm = snr_norm[idx_sort]

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


class Dictionary:
    def __init__(self,
                 dictfile=None,
                 lamb_rest=None,
                 lamb_sys=None,
                 lamb_sys_size=306,
                 lamb_rest_start=0.2,
                 lamb_rest_end=5.5,
                 lamb_rest_step=0.002,
                 Ndict=None,
                 Ndict_sys=None,
                 alpha_train=None,
                 centering=False):

        # These settings are ONLY used if lamb_rest or lamb_sys_size is not found and need to be initialized
        self.lamb_sys = lamb_sys
        if self.lamb_sys is not None:
            self.lamb_sys_size = self.lamb_sys.shape[0]
        else:
            self.lamb_sys_size = lamb_sys_size
        self.lamb_rest_start = lamb_rest_start
        self.lamb_rest_end = lamb_rest_end
        self.lamb_rest_step = lamb_rest_step
        self.D_allz = None
        self._D_allz_filtered = False
        # if dictfile is provided, read from file and set up variables
        if dictfile is not None:
            self.read_dict(dictfile)
            # if direct input of Ndict or Ndict_sys is provided, modify the dictionary
            if (Ndict is not None) or (Ndict_sys is not None):
                self.modify_dict(Ndict=Ndict, Ndict_sys=Ndict_sys, alpha_train=alpha_train)
        # if dictfile is not provided, initialize from input settings
        else:
            self.initialize_dict(Ndict=Ndict,
                                 Ndict_sys=Ndict_sys,
                                 lamb_rest=lamb_rest,
                                 lamb_rest_start=lamb_rest_start,
                                 lamb_rest_end=lamb_rest_end,
                                 lamb_rest_step=lamb_rest_step,
                                 lamb_sys_size=lamb_sys_size,
                                 alpha_train=alpha_train)

        # TEMP this doesn't take care of A and B if they already exist
        if centering:
            self.D_rest = self.D_rest - np.mean(self.D_rest, axis=1)[:,None]
            self.D_rest = self.D_rest / np.linalg.norm(self.D_rest, axis=1)[:,None]
            if self.D_sys is not None:
                self.D_sys = self.D_sys - np.mean(self.D_sys, axis=1)[:,None]
                self.D_sys = self.D_sys / np.linalg.norm(self.D_sys, axis=1)[:,None]         

        if self.D_rest is not None:
            self.D_rest_initial = self.D_rest.copy()
        else:
            self.D_rest_initial = None
        if self.D_sys is not None:
            self.D_sys_initial = self.D_sys.copy()
        else:
            self.D_sys_initial = None

    def read_dict(self, dictfile):
        # reading npz format
        if dictfile.split('.')[-1] == 'npz':
            dicts = np.load(dictfile)
            keys = dicts.keys()
            if 'lamb_rest' in keys:
                self.lamb_rest = dicts['lamb_rest']
            else:
                self.lamb_rest = np.arange(self.lamb_rest_start, self.lamb_rest_end, self.lamb_rest_step)

            # determine if D_rest, Ndict exist in file
            if 'D_rest' in keys and dicts['D_rest'] is not None:
                self.D_rest = dicts['D_rest']
                self.Ndict = self.D_rest.shape[0]
                # further determine if A and B matrices exist
                try:
                    self.A_rest = dicts['A']
                    self.B_rest = dicts['B']
                except:
                    self.A_rest = np.zeros((self.Ndict, self.Ndict))
                    self.B_rest = np.zeros((len(self.lamb_rest), self.Ndict))
            else:
                self.D_rest = None
                self.Ndict = 0
                self.A_rest = None
                self.B_rest = None

            # determine if D_sys, Ndict_sys exist in file
            if 'D_sys' in keys and dicts['D_sys'] is not None:
                self.D_sys = dicts['D_sys']
                self.Ndict_sys = self.D_sys.shape[0]
                self.lamb_sys_size = self.D_sys.shape[1]
                # further determine if A_sys and B_sys exist
                try:
                    self.A_sys = dicts['A_sys']
                    self.B_sys = dicts['B_sys']
                except:
                    self.A_sys = np.zeros((self.Ndict_sys, self.Ndict_sys))
                    self.B_sys = np.zeros((self.lamb_sys_size, self.Ndict_sys))
            else:
                self.D_sys = None
                self.Ndict_sys = 0
                self.A_sys = None
                self.B_sys = None

            # determine if D_rest_alphas and D_rest_alphas_ndict exist in file
            try:
                self.D_rest_alphas = dicts['D_rest_alphas']
                self.D_rest_alphas_ndict = dicts['D_rest_alphas_ndict']
            except:
                self.D_rest_alphas = None
                self.D_rest_alphas_ndict = None

            # determine if D_sys_alphas and its ndict setting exist in file
            try:
                self.D_sys_alphas = dicts['D_sys_alphas']
                self.D_sys_alphas_ndict = dicts['D_sys_alphas_ndict']
            except:
                self.D_sys_alphas = None
                self.D_sys_alphas_ndict = None

        # reading h5 format
        elif dictfile.split('.')[-1] == 'h5':
            with h5py.File(dictfile, 'r') as f:
                if 'lamb_rest' in f:
                    self.lamb_rest = f['lamb_rest'][()]
                else:
                    self.lamb_rest = np.arange(self.lamb_rest_start, self.lamb_rest_end, self.lamb_rest_step)

                if 'lamb_sys' in f:
                    self.lamb_sys = f['lamb_sys'][()]
                    self.lamb_sys_size = self.lamb_sys.shape[0]
                elif 'lamb_obs' in f:
                    self.lamb_sys = f['lamb_obs'][()]
                    self.lamb_sys_size = self.lamb_sys.shape[0]

                # determine if D_rest and Ndict exist
                if 'D_rest' in f and f['D_rest'][()] is not None:
                    self.D_rest = f['D_rest'][()]
                    self.Ndict = self.D_rest.shape[0]              
                    # check if A, B exist
                    try:
                        self.A_rest = f['A'][()]
                        self.B_rest = f['B'][()]
                    except:
                        self.A_rest = np.zeros((self.Ndict, self.Ndict))
                        self.B_rest = np.zeros((len(self.lamb_rest), self.Ndict))
                else:
                    self.D_rest = None
                    self.Ndict = 0
                    self.A_rest = None
                    self.B_rest = None

                # determie if D_sys and Ndict_sys exist
                if 'D_sys' in f and f['D_sys'][()] is not None:
                    self.D_sys = f['D_sys'][()]
                    self.Ndict_sys = len(self.D_sys)
                    self.lamb_sys_size = self.D_sys.shape[1]
                    # check if A_sys and B_sys exist
                    try:
                        self.A_sys = f['A_sys'][()]
                        self.B_sys = f['B_sys'][()]
                    except:
                        self.A_sys = np.zeros((self.Ndict_sys, self.Ndict_sys))
                        self.B_sys = np.zeros((self.lamb_sys_size, self.Ndict_sys))        
                else:
                    self.D_sys = None
                    self.Ndict_sys = 0
                    self.A_sys = None
                    self.B_sys = None

                # determine if D_rest_alphas settings exist
                try:
                    self.D_rest_alphas = f['D_rest_alphas'][()]
                    self.D_rest_alphas_ndict = f['D_rest_alphas_ndict'][()]
                except:
                    self.D_rest_alphas = None
                    self.D_rest_alphas_ndict = None

                # determine if D_sys_alphas settings exist
                try:
                    self.D_sys_alphas = f['D_sys_alphas'][()]
                    self.D_sys_alphas_ndict = f['D_sys_alphas_ndict'][()]
                except:
                    self.D_sys_alphas = None
                    self.D_sys_alphas_ndict = None


    def modify_dict(self, Ndict=None, Ndict_sys=None, alpha_train=None):

        # if input Ndict is not None then try to modify D_rest, Ndict, A, B
        if Ndict is not None:
            if self.D_rest is not None:
                # if in rare case Ndict is set to zero, remove all information regarding existing D_rest
                if Ndict == 0:
                    self.Ndict = 0
                    self.D_rest = None
                    self.A_rest = None
                    self.B_rest = None
                # add or subtract dictionaries from existing D_rest
                else:
                    self.Ndict = Ndict

                    # modify D_rest based on new Ndict
                    # if new Ndict is smaller, remove extra templates
                    if self.Ndict <= len(self.D_rest):
                        self.D_rest = self.D_rest[:self.Ndict]
                        self.A_rest = self.A_rest[:self.Ndict, :self.Ndict]
                        self.B_rest = self.B_rest[:, :self.Ndict]
                    # if new Ndict is larger, add noise templates
                    else:
                        old_Ndict = len(self.D_rest)
                        D_rest_additional = initialize_dicts(self.Ndict-old_Ndict, lamb_rest=self.lamb_rest)
                        self.D_rest = np.vstack([self.D_rest, D_rest_additional])
                        Anew = np.zeros((self.Ndict, self.Ndict))
                        Anew[:old_Ndict, :old_Ndict] = self.A_rest
                        self.A_rest = Anew
                        self.B_rest = np.hstack([self.B_rest, np.zeros((len(self.lamb_rest), self.Ndict-old_Ndict))])

                        # if alpha_train is given as input, try to modify D_rest_alphas settings if they also exist
                        if alpha_train is not None and self.D_rest_alphas is not None:
                            self.D_rest_alphas = np.append(self.D_rest_alphas, alpha_train)
                            self.D_rest_alphas_ndict = np.append(self.D_rest_alphas_ndict, Ndict)

            # if D_rest didn't exist and input Ndict > 0, add noise templates
            elif Ndict > 0:
                self.Ndict = Ndict
                self.D_rest = initialize_dicts(self.Ndict, lamb_rest=self.lamb_rest)
                self.A_rest = np.zeros((self.Ndict, self.Ndict))
                self.B_rest = np.zeros((len(self.lamb_rest), self.Ndict))
                if alpha_train is not None:
                    # since D_rest doesn't exist, D_rest_alphas settings definitely don't exist, so create new when alpha_train is given
                    self.D_rest_alphas = np.array([alpha_train])
                    self.D_rest_alphas_ndict = np.array([Ndict])


        # if input Ndict_sys is not None, then try to modify D_sys, Ndict_sys, A_sys, B_sys
        if Ndict_sys is not None:
            if self.D_sys is not None:
                # if in rare case Ndict_sys is set to zero, remove all information regarding existing D_sys
                if Ndict_sys == 0:
                    self.Ndict_sys = 0
                    self.D_sys = None
                    self.A_sys = None
                    self.B_sys = None
                # add or subtract dictionaries from existing D_sys
                else:
                    self.Ndict_sys = Ndict_sys

                    # modify D_sys based on new Ndict
                    # if new Ndict_sys is smaller, remove extra templates
                    if self.Ndict_sys <= len(self.D_sys):
                        self.D_sys = self.D_sys[:self.Ndict_sys]
                        self.A_sys = self.A_sys[:self.Ndict_sys, :self.Ndict_sys]
                        self.B_sys = self.B_sys[:, :self.Ndict_sys]
                    # if new Ndict_sys is larger, add noise templates
                    else:
                        old_Ndict_sys = len(self.D_sys)
                        D_sys_additional = initialize_dicts(self.Ndict_sys-old_Ndict_sys, lamb_rest=np.zeros(self.lamb_sys_size))
                        self.D_sys = np.vstack([self.D_sys, D_sys_additional])
                        Anew_sys = np.zeros((self.Ndict_sys, self.Ndict_sys))
                        Anew_sys[:old_Ndict_sys, :old_Ndict_sys] = self.A_sys
                        self.A_sys = Anew_sys
                        self.B_sys = np.hstack([self.B_sys, np.zeros((self.lamb_sys_size, self.Ndict_sys-old_Ndict_sys))])

                        # if alpha_train is given as input, try to modify D_sys_alphas settings if they also exist
                        if alpha_train is not None and self.D_sys_alphas is not None:
                            self.D_sys_alphas = np.append(self.D_sys_alphas, alpha_train)
                            self.D_sys_alphas_ndict = np.append(self.D_sys_alphas_ndict, Ndict_sys)

            # if D_sys didn't exist and input Ndict_sys > 0, add noise templates
            elif Ndict_sys > 0:
                self.Ndict_sys = Ndict_sys
                self.D_sys = initialize_dicts(self.Ndict_sys, lamb_rest=np.zeros(self.lamb_sys_size))
                self.A_sys = np.zeros((self.Ndict_sys, self.Ndict_sys))
                self.B_sys = np.zeros((self.lamb_sys_size, self.Ndict_sys))
                if alpha_train is not None:
                    # since D_rest doesn't exist, D_rest_alphas settings definitely don't exist, so create new when alpha_train is given
                    self.D_sys_alphas = np.array([alpha_train])
                    self.D_sys_alphas_ndict = np.array([Ndict_sys])


    def initialize_dict(self,
                        Ndict=None,
                        Ndict_sys=None,
                        lamb_rest=None,
                        lamb_rest_start=0.2,
                        lamb_rest_end=5.5,
                        lamb_rest_step=0.002,
                        lamb_sys=None,
                        lamb_sys_size=306,
                        alpha_train=None):

        if lamb_rest is not None:
            self.lamb_rest = lamb_rest
        else:
            self.lamb_rest = np.arange(lamb_rest_start, lamb_rest_end, lamb_rest_step)

        if lamb_sys is not None:
            lamb_sys_size = lamb_sys.shape[0]

        self.Ndict = Ndict
        self.Ndict_sys = Ndict_sys

        self.D_rest_alphas = None
        self.D_rest_alphas_ndict = None
        self.D_sys_alphas = None
        self.D_sys_alphas_ndict = None
        # self.D_allz = None

        # Create D_rest and related stuffs
        if self.Ndict is not None and self.Ndict > 0:
            self.D_rest = initialize_dicts(self.Ndict, lamb_rest=self.lamb_rest)
            self.A_rest = np.zeros((self.Ndict, self.Ndict))
            self.B_rest = np.zeros((len(self.lamb_rest), self.Ndict))
            if alpha_train is not None:
                self.D_rest_alphas = np.array([alpha_train])
                self.D_rest_alphas_ndict = np.array([self.Ndict])
        else:
            self.D_rest = None
            self.A_rest = None
            self.B_rest = None

        # Create D_sys and related stuffs
        if self.Ndict_sys is not None and self.Ndict_sys > 0:
            self.D_sys = initialize_dicts(self.Ndict_sys, lamb_rest=np.zeros(lamb_sys_size))
            self.A_sys = np.zeros((self.Ndict_sys, self.Ndict_sys))
            self.B_sys = np.zeros((lamb_sys_size, self.Ndict_sys))
            if alpha_train is not None:
                self.D_sys_alphas = np.array([alpha_train])
                self.D_sys_alphas_ndict = np.array([self.Ndict])
        else:
            self.D_sys = None
            self.A_sys = None
            self.B_sys = None

    def ndict_from_alpha(self, alpha=None, alpha_sys=None):
        # if alpha_sys is None:
            # alpha_sys = alpha
        if self.D_rest_alphas is not None and alpha is not None:
            idx_Ndicts_now = max(np.searchsorted(-self.D_rest_alphas, -alpha, side='right')-1, 0)
            Ndicts_now = self.D_rest_alphas_ndict[idx_Ndicts_now]
        else:
            Ndicts_now = self.Ndict
        if self.D_sys_alphas is not None and alpha_sys is not None:
            idx_Ndicts_sys_now = max(np.searchsorted(-self.D_sys_alphas, -alpha_sys, side='right')-1, 0)
            Ndicts_sys_now = self.D_sys_alphas_ndict[idx_Ndicts_sys_now]
        else:
            Ndicts_sys_now = self.Ndict_sys
        return Ndicts_now, Ndicts_sys_now

    def at_z(self, z, lamb_obs, alpha=None, alpha_sys=None, filters=None, conv=False):
        # if alpha_sys is None:
            # alpha_sys = alpha
        Ndicts_now, Ndicts_sys_now = self.ndict_from_alpha(alpha=alpha, alpha_sys=alpha_sys)
        D_thisz = apply_redshift(self.D_rest[:Ndicts_now], z, self.lamb_rest, lamb_obs, filters=filters, conv=conv)
        if self.D_sys is not None:
            if not np.allclose(lamb_obs, self.lamb_sys):
                # raise ValueError('lamb_obs has different size than D_sys size!')
                D_sys0 = self.D_sys[:Ndicts_sys_now]
                D_sys_interp = np.array([np.interp(lamb_obs, self.lamb_sys, D_sys0[i]) for i in range(D_sys0.shape[0])])
            else:
                D_sys_interp = self.D_sys[:Ndicts_sys_now]
            # D_sys = D_sys_interp[:Ndicts_sys_now]
            D_zout = np.vstack((D_sys_interp, D_thisz))
        else:
            D_zout = D_thisz
        return D_zout

    def make_D_allz(self, lamb_obs, zgrid=None, alpha=None, alpha_sys=None, filters=None, conv=False):
        if zgrid is None:
            zgrid = generate_zgrid()
        Ndicts_now, _ = self.ndict_from_alpha(alpha=alpha, alpha_sys=alpha_sys)
        D_allz = apply_redshift_all(self.D_rest[:Ndicts_now], zgrid, self.lamb_rest, lamb_obs, filters=filters, conv=conv)
        if conv:
            self._D_allz_filtered = True
        else:
            self._D_allz_filtered = False
        self.D_allz = D_allz
        return D_allz

    def make_D_zgrid(self, lamb_obs, zgrid=None, alpha=None, alpha_sys=None, filters=None, conv=False):
        '''
        Similar to D_allz, but includes D_sys and sort them correctly
        '''
        if zgrid is None:
            zgrid = generate_zgrid()
        Ndicts_now, Ndicts_sys_now = self.ndict_from_alpha(alpha=alpha, alpha_sys=alpha_sys)
        if self.D_allz is None:
            D_allz = apply_redshift_all(self.D_rest[:Ndicts_now], zgrid, self.lamb_rest, lamb_obs, filters=filters, conv=conv)
        else:
            D_allz = self.D_allz
        if conv:
            self._D_allz_filtered = True
        if self.D_sys is not None:
            D_zgrid = np.concatenate((np.broadcast_to(self.D_sys[:Ndicts_sys_now], (D_allz.shape[0], *self.D_sys[:Ndicts_sys_now].shape)), D_allz), axis=1)
        self.D_zgrid =D_zgrid
        return D_zgrid
    

# def get_D_rest_list_idx(alpha, D_rest_list_alphas):
#     idx = np.searchsorted(-D_rest_list_alphas, -alpha, side="left") - 1
#     idx = np.clip(idx, 0, len(D_rest_list_alphas) - 2)
#     return int(idx)

def get_D_rest_list_idx(alpha, D_rest_list_tab):
    edges = np.array([row[0] for row in D_rest_list_tab] + [D_rest_list_tab[-1][1]])
    idx = np.searchsorted(-edges, -alpha, side="left") - 1
    return np.clip(idx, 0, len(D_rest_list_tab) - 1)

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

def pbpz(zgrid=None, alpha=1.27, beta=0.73, gamma=2.27):
    if zgrid is None:
        zgrid = generate_zgrid()
    return zgrid**alpha * np.exp(-((zgrid/beta)**gamma))

# Initialize dictionary
def initialize_dicts(Ndict, # dictionary_fluctuation_scaling=None, 
                     templates_EAZY=None, 
                     num_EAZY_as_dict=0, 
                     lamb_rest=np.arange(0.2,5.5,0.002), 
                     add_constant=False):
    """
    Return: D_rest
    """
    if Ndict is not None and Ndict > 0:
        D_rest_list = []
        # Load EAZY as initialized dictionary
        if num_EAZY_as_dict > 0:
            if templates_EAZY is not None:
                for i in range(num_EAZY_as_dict):
                    D_rest_list.append(templates_EAZY[i]*0.1)
            else:
                num_EAZY_as_dict = 0
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
    else:
        return None


# Read SPHEREx filters
def read_filters(filter_list, half_length=105, return_lamb_obs=False, response_threshold=0.1, return_unique_inverse=False):
    '''
    Read SPHEREx official filters and return ndarray of (nfilt, 2, 2*half_length)
    
    Parameters
    ----------
    filter_list : str
        Full path to the filter_list.txt that contains all the filter names in order
    half_length : int, default=105
        Half size of each filter's length.
        Currently, official SPHEREx filters don't have the same size; half_length of 105 is a good middle ground for 306 bands
        to cut off unimportant outside parts while preserving response to <3% accuracy.
        If half_length*2 > existing wavelength grid, fill longer wavelength side with zeros
    return_unique_inverse : bool, default=False
        If True, also return unique_wavelength, inverse_wavelength, conv_weights that will be
        used in fast_convolve_filters()
    return_lamb_obs : bool, default=False
        If True, also returns the lamb_obs by finding 'fiducial_filters_cent_waves.txt' at the same directory as the filter_list
        If the cent_wave file can't be found, calculate by weight average of normalized response>0.1
    
    Returns
    -------
    filters : ndarray of shape (nfilt, 2, 2*half_length)
        Full array of all the filters' wavelength coverage (micron) and response
        For example, n_filter=i has wavelength filters[i][0] and response filters[i][1]
    unique_wavelength : ndarray of shape (n_unique_wavs, )
        Only return when return_unique_inverse=True
        1-D array of unique wavelength points that are hstacked from all filters' wavelength grids
    inverse_wavelength : ndarray of shape (n_total_wavs, )
        Only return when return_unique_inverse=True
        1-D mapping array of total hstacked wavelength grids to unique_wavelength with:
        all_wavelength = unique_wavelength[inverse_wav]
    conv_weights : ndarray of shape (nfilt, 2*half_length)
        Only return when return_unique_inverse=True
        weight for trapezoid integration that includes both wavelength grid spacing weights
        and filter response weights, such that
        np.trapezoid(flux * response) = np.sum(flux * conv_weights)
    lamb_obs : ndarray of shape (nfilt, )
        Only return when return_lamb_obs=True
    '''
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
        if len(wavelength_i) < half_length*2:
            wavelength_i1 = np.zeros(half_length*2)
            response_i1 = np.zeros(half_length*2)
            wavelength_i1[:len(wavelength_i)] = wavelength_i
            dw = wavelength_i[-1] - wavelength_i[-2]
            wavelength_i1[len(wavelength_i):] = wavelength_i[-1] + dw * np.arange(1, len(wavelength_i1)-len(wavelength_i)+1)
            response_i1[:len(response_i)] = response_i
        else:
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

    # calculate convolution weights for filter convolution & integration
    wavelengths = filters[:,0]
    filter_responses = filters[:,1]
    dx = np.diff(wavelengths, axis=1)
    trap_weights = np.empty_like(wavelengths)

    trap_weights[:, 0] = dx[:, 0] / 2
    trap_weights[:, -1] = dx[:, -1] / 2
    trap_weights[:, 1:-1] = (dx[:, :-1] + dx[:, 1:]) / 2
    convolution_weights = trap_weights * filter_responses

    # calculate unique wavelength points and the inverse map
    all_wavelengths = np.concatenate([wavelengths[i] for i in range(Nf)])
    unique_wavelength, inverse_wavelength = np.unique(all_wavelengths, return_inverse=True)

    results = (filters,)

    if return_unique_inverse:
        results = results + (unique_wavelength, inverse_wavelength, convolution_weights,)

    if return_lamb_obs:
        try:
            filter_central_wavelengths = filter_list.replace('fiducial_filters.txt', 'fiducial_filters_cent_waves.txt')
            lamb_obs = np.genfromtxt(filter_central_wavelengths)[:,1]
        except:
            lamb_obs = np.zeros(Nf)
            for i in range(Nf):
                wav_i = filters[i][0]
                res_i = filters[i][1] / np.max(filters[i][1])
                mask = res_i > response_threshold
                lamb_obs[i] = np.sum(wav_i[mask] * res_i[mask]) / np.sum(res_i[mask])
        results = results + (lamb_obs,)

    if not return_unique_inverse and not return_lamb_obs:
        results = filters

    return results


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

# @njit(fastmath=True)
def fast_convolve_filter(wl, flux, filters=None):
    if filters is not None:
        # wavelength = filters["wavelength"]
        # response = filters["response"]
        # unique_wavelength = filters["unique_wavelength"]
        # inverse_wavelength = filters["inverse_wavelength"]
        # conv_weights = filters["convolution_weights"]
        wavelength = filters[0][:,0]
        # response = filters[0][:,1]
        unique_wls = filters[1]
        inverse_wls = filters[2]
        conv_weights = filters[3]
        unique_flux_interp = np.interp(unique_wls, wl, flux)
        flux_interps = unique_flux_interp[inverse_wls].reshape(wavelength.shape)
        flux_conv = np.sum(flux_interps*conv_weights, axis=1)
    else:
        flux_conv = flux
    return flux_conv


# function to integrate fluxes convolved with filters and return all 102 values
@njit(fastmath=True)
def convolve_filter(wl, flux, filters=None):
    '''
    Convolve spectra with filters read from read_filters()
    filters need to have shape (nfilt, 2, ns)

    Parameters
    ----------
    wl : ndarray of shape n
        original wavelength points in micron
    
    flux : ndarray of shape n
        original flux in f_nu

    filters : ndarray of shape (nfilt, 2, ns)
        filters array from read_filters()

    Returns
    -------
    flux_conv : ndarray of shape (nfilt, )
        Convolved fluxes in each passband
    '''
    if filters is not None:
        Nf = len(filters)
        flux_conv = np.zeros(Nf)
        for i in range(Nf):
            lb = filters[i][0]
            ftrans = filters[i][1]
            f_interp = np.interp(lb, wl, flux)
            fnu_i = np.trapezoid(f_interp*ftrans, lb)
            flux_conv[i] = fnu_i
    else:
        flux_conv = flux
    return flux_conv



# utility function to apply a redshift to spectral dictionary
@jit(nopython=True, fastmath=True)
def apply_redshift1(D, z, lamb_in, lamb_out):
    # initialize output dictionary
    D_out = np.zeros((D.shape[0],len(lamb_out)))
    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in*(1+z)
    D_out = func.multiInterp1(lamb_out, lamb_inx1pz, D)
    return D_out

# apply redshift function with additional option to convolve with filters
# @jit(nopython=True, fastmath=True)
def apply_redshift(D, z, lamb_in, lamb_out, Ndict_noshift=0, filters=None, conv=False):
    if isinstance(filters, tuple):
        convolving_func = fast_convolve_filter
    elif isinstance(filters, np.ndarray):
        convolving_func = convolve_filter
    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in * (1+z)
    if conv and (filters is not None):
        D_out = np.zeros((D.shape[0],len(lamb_out)))
        for i in range(D.shape[0]):
            if i < Ndict_noshift:    # if i is smaller than Ndict_noshift, don't shift it
                D_conv = convolving_func(lamb_in, D[i,:], filters)
            else:
                D_conv = convolving_func(lamb_inx1pz, D[i,:], filters)
            D_out[i,:] = D_conv
        # else:
        #     D_out[i,:] = np.interp(lamb_out,lamb_inx1pz,D[i,:])
    else:
        if Ndict_noshift > 0:
            D_out_sys = func.multiInterp1(lamb_out, lamb_in, D[:Ndict_noshift])
            D_out_redshifted = func.multiInterp1(lamb_out, lamb_inx1pz, D[Ndict_noshift:])
            D_out = np.vstack((D_out_sys, D_out_redshifted))
        else:
            D_out = func.multiInterp1(lamb_out, lamb_inx1pz, D)
    return D_out


# apply redshift to all zgrid points and output 3-d array
# @njit(fastmath=True)
def apply_redshift_all(D, zgrid, lamb_in, lamb_out, Ndict_noshift=0, filters=None, conv=False):
    nz = zgrid.shape[0]
    D_all = np.zeros((nz, D.shape[0], lamb_out.shape[0]))
    for i in range(nz):
        D_thisz = apply_redshift(D, zgrid[i], lamb_in, lamb_out, Ndict_noshift=Ndict_noshift, filters=filters, conv=conv)
        D_all[i] = D_thisz
    return D_all


# Fit data with dictionary across the whole redshift grid
# @jit(nopython=True, fastmath=True)
def fit_zgrid(lamb_obs, spec_obs, err_obs, dictionary=None,
              lamb_rest=None, D_rest=None, D_sys=None, D_allz=None, 
              D_rest_alphas=None, D_rest_alphas_ndict=None, D_sys_alphas=None, D_sys_alphas_ndict=None,
              D_rest_list=None, D_allz_list=None, D_rest_list_tab=None,
              zgrid=None, zinput=None, filters=None, median_ynorm=None, median_spectra=None,
              larslasso=True, alpha=0, alpha_sig=0.0, positive=True, center_Xy=False, unit_X=True, unit_y=True, 
              path=True, best_estimator=None, max_feature=None, alpha_ns_scaling=False, 
              active_alpha_0=False, active_alpha_0_fitting_penalty=False, active_ols=False,
              dz=0.002, zmin=0.0, zmax=3.0, scale_1plusz=True, error=False, probline=0.317/2, 
              conv=False, conv_finegrid=False, local_finegrid=False, local_finegrid_size=0.03, local_finegrid_dz=0.001,
              output_all_coefs=False, fix_at_zinput=False, pz=None):
    if (median_spectra is not None) and (median_spectra > 1):
        median_ynorm = None     # safeguard; if median_spectra, there is no reason to use median_ynorm
    # Check if D_rest is necessary but not given
    if ((zinput is not None) or local_finegrid) and D_rest is None and D_sys is None and dictionary is None and ((D_rest_list is None) or (D_allz_list is None)):
        raise ValueError("D_rest is required but not given for zinput != None")
    if D_rest is None and D_allz is None and D_sys is None and dictionary is None and D_rest_list is None and D_allz_list is None:
        raise ValueError("D_rest, D_allz, D_sys, D_rest_list, D_allz_list are all not given")

    use_D_allz_list = False
    if (D_rest_list is not None or D_allz_list is not None) and D_rest_list_tab is not None:
        use_D_allz_list = True
        if lamb_rest is None:
            raise ValueError("lamb_rest should be provided when using D_rest_list!")
        D_rest = None
        D_rest_alphas = None
        D_rest_alphas_ndict = None
        D_sys = None
        D_sys_alphas = None
        D_sys_alphas_ndict = None
        if D_allz_list is None:
            D_allz_list = []
            for i in range(len(D_rest_list)):
                D_allz_list.append(apply_redshift_all(D_rest_list[i], zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv))

    # if dictionary object is provided, read info from it
    elif dictionary is not None:
        lamb_rest = dictionary.lamb_rest
        D_rest = dictionary.D_rest
        D_rest_alphas = dictionary.D_rest_alphas
        D_rest_alphas_ndict = dictionary.D_rest_alphas_ndict
        D_sys = dictionary.D_sys
        D_sys_alphas = dictionary.D_sys_alphas
        D_sys_alphas_ndict = dictionary.D_sys_alphas_ndict
        D_allz = dictionary.D_allz
        if (D_allz is None) or (conv != dictionary._D_allz_filtered):
            # if conv and dictionary._D_allz_filtered do not match, rebuild D_allz
            if (conv != dictionary._D_allz_filtered):
                print(f'libs.fit_zgrid: conv={conv}, dictionary._D_allz_filtered={dictionary._D_allz_filtered}, rebuild D_allz')
            D_allz = dictionary.make_D_allz(lamb_obs, zgrid=zgrid, filters=filters, conv=conv)
    # determine original Ndict
    if use_D_allz_list:
    # if in D_rest_list mode, Ndicts_original is max number of dictionary
        Ndicts_original = D_allz_list[-1].shape[1]
    elif D_rest is not None:
        Ndicts_original = D_rest.shape[0]
    elif D_allz is not None:
        Ndicts_original = D_allz.shape[1]
    else:
        Ndicts_original = 0

    # determine original Ndict_sys
    if D_sys is not None:
        Ndicts_sys_original = D_sys.shape[0]
    else:
        Ndicts_sys_original = 0

    # if median_spectra is set to > 1, pre-process input signal with a median filter
    if (median_spectra is not None) and (median_spectra > 1):
        spec_obs = median_filter(spec_obs, size=median_spectra)

    # calculate ynorm first and use D_rest_alphas to determine how many templates to use
    if D_rest_alphas is not None or D_sys_alphas is not None or use_D_allz_list:
        if (median_ynorm is None) or (median_ynorm == 1):
            ynorm = np.linalg.norm(spec_obs/err_obs)
        else:
            ynorm = np.linalg.norm(median_filter(spec_obs, size=median_ynorm)/err_obs)
        if ynorm != 0:
            alpha_now = 1/ynorm
        else:
            # safeguard against some 0 flux objects having ynorm=0
            alpha_now = 0.2

    # if D_rest_alphas is given, calculate SNR and select the correct number of dictionaries to be used based on D_rest_alphas_ndict
    if D_rest_alphas is not None:
        idx_Ndicts_now = max(np.searchsorted(-D_rest_alphas, -alpha_now, side='right')-1, 0)
        Ndicts_now = D_rest_alphas_ndict[idx_Ndicts_now]
        if D_rest is not None:
            D_rest = D_rest[:Ndicts_now]
        if D_allz is not None:
            D_allz = D_allz[:, :Ndicts_now]
    elif use_D_allz_list:
        idx_D_list = get_D_rest_list_idx(alpha=alpha_now, D_rest_list_tab=D_rest_list_tab)
        if D_rest_list is not None:
            D_rest = D_rest_list[idx_D_list]
            Ndicts_now = D_rest.shape[0]
        if D_allz_list is not None:
            D_allz = D_allz_list[idx_D_list]
            Ndicts_now = D_allz.shape[1]
        # else:
            # D_allz = apply_redshift_all(D_rest, zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv)
    else:
        Ndicts_now = Ndicts_original

    # if D_sys_alphas is given, calculate SNR and select the correct number of sys dictionaries to be used based on D_sys_alphas_ndict
    if D_sys_alphas is not None:
        idx_Ndicts_sys_now = max(np.searchsorted(-D_sys_alphas, -alpha_now, side='right')-1, 0)
        Ndicts_sys_now = D_sys_alphas_ndict[idx_Ndicts_sys_now]
        if D_sys is not None:
            D_sys = D_sys[:Ndicts_sys_now]
    else:
        Ndicts_sys_now = Ndicts_sys_original

    # safeguard against case when only D_sys is provided (z=0 case) but forgot to input zinput
    if (D_rest is None) and (D_allz is None) and (D_sys is not None):
        zinput = 0.0
        error = False

    # if redshift is fixed and doesn't need error, have only single point in the redshift grid and turn off local_finegrid
    if (zinput is not None) and not error:
        zgrid = np.array([zinput])
        idx_zpeak0 = 0
        local_finegrid = False
        zpeak0 = zinput
        fix_at_zinput = True
        idx_zinput = None
        if D_rest is not None:
            D_zinput = apply_redshift(D_rest, zinput, lamb_rest, lamb_obs, filters=filters, conv=conv)
            if D_sys is not None:
                D_zinput = np.vstack((D_sys, D_zinput))
        else:
            # if only D_sys is provided, then all necessary dictionary is D_sys
            D_zinput = D_sys
        D_allz = D_zinput.reshape((1, D_zinput.shape[0], D_zinput.shape[1]))

    # if full zgrid need to be evaluated:
    else:
        # if zgrid is not given, create a default zgrid    
        if zgrid is None:
            zgrid = generate_zgrid(zmin=zmin, zmax=zmax, dz=dz, scale_1plusz=scale_1plusz) 
        else:
            zgrid_limit = (zgrid<=zmax) & (zgrid>=zmin)
            zgrid = zgrid[zgrid_limit]
        zgrid_original = zgrid.copy()   # keep a copy of original zgrid before inserting zinput for pz
        # if zinput is still given, add z=zinput to zgrid
        if zinput is not None:
            # idx_zpeak0 = np.searchsorted(zgrid, zinput)
            # zgrid = np.insert(zgrid, idx_zpeak0, zinput)
            idx_zinput = np.searchsorted(zgrid, zinput)
            zgrid = np.insert(zgrid, idx_zinput, zinput)
            if fix_at_zinput:
                idx_zpeak0 = idx_zinput
                zpeak0 = zinput
        else:
            idx_zinput = None

        # check if D_allz is given
        if D_allz is None:
            D_allz = apply_redshift_all(D_rest, zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv)
        # if D_allz and zinput are provided, create a new D_allz array and insert D_zinput to the correct location
        elif zinput is not None:
            D_zinput = apply_redshift(D_rest, zinput, lamb_rest, lamb_obs, filters=filters, conv=conv)
            D_allz = np.insert(D_allz, idx_zinput, D_zinput, axis=0)

        # insert D_sys to the beginning of each element of D_allz
        if D_sys is not None:
            D_allz = np.concatenate((np.broadcast_to(D_sys, (D_allz.shape[0], *D_sys.shape)), D_allz), axis=1)

    # after above steps, zgrid and D_allz are ready for any given situation
    # prepare arrays to store coefficients and chi2
    coefs_zgrid = np.zeros((zgrid.shape[0], D_allz.shape[1]))
    bs_zgrid = np.zeros_like(zgrid)
    cost_zgrid = np.zeros_like(zgrid) + np.inf
    # larslasso_step_zgrid = np.zeros_like(zgrid)
    # alpha_path_list = List()
    alpha_path_list = []

    if (median_ynorm is not None) and (median_ynorm > 1) and (median_spectra is None):
        # adjust alpha_sig
        original_1sig = 1/np.linalg.norm(spec_obs/err_obs)
        if np.isnan(original_1sig):
            original_1sig = 0.2
        new_1sig = 1/np.linalg.norm(median_filter(spec_obs, size=median_ynorm)/err_obs)
        if np.isnan(new_1sig):
            new_1sig = 0.2
        one_sig_ratio = new_1sig/original_1sig
        alpha_sig = alpha_sig * one_sig_ratio

    # loop over zgrid to calculate best-fit and chi2
    for i in range(zgrid.shape[0]):
        D_thisz = D_allz[i]
        if not larslasso:
            coef, model, _ = fit_models_ols(D_thisz, spec_obs, err_obs)
            b = np.array(0.0)
            cost = np.sum((model - spec_obs)**2/err_obs**2)
            alpha_path = np.array([0.0])
        else:
            # D_thisz = np.ascontiguousarray(D_thisz)
            # spec_obs = np.ascontiguousarray(spec_obs)
            # err_obs = np.ascontiguousarray(err_obs)
            coef, b, model, cost, _, alpha_path = fit_model_larslasso(D_thisz, spec_obs, err_obs, alpha=alpha, alpha_sig=alpha_sig, positive=positive, 
                                                center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator, 
                                                max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, max_feature=max_feature, 
                                                active_alpha_0=active_alpha_0, active_alpha_0_fitting_penalty=active_alpha_0_fitting_penalty, active_ols=active_ols)

        coefs_zgrid[i] = coef
        bs_zgrid[i] = b
        cost_zgrid[i] = cost
        # larslasso_step_zgrid[i] = larslasso_step
        alpha_path_list.append(alpha_path)

    # if zinput=True, error=False, idx_zpeak0 = 0
    # if zinput=True, err=True, idx_zpeak0 is the index where zinput is
    if (zinput is None) or (not fix_at_zinput):
        idx_zpeak0 = np.argmin(cost_zgrid)
        zpeak0 = zgrid[idx_zpeak0]
    # if local_finegrid is False, skip local part and advance idx_zpeak1 and zpeak1 with the 0 variables
    if not local_finegrid:
        idx_zpeak1 = idx_zpeak0
        zpeak1 = zpeak0

    # local fine grid
    else:
        z_localmax = min((zpeak0+local_finegrid_size, zmax))
        z_localmin = max((zpeak0-local_finegrid_size, zmin))
        zgrid_local = np.arange(z_localmin, z_localmax, local_finegrid_dz)
        local_length = zgrid_local.shape[0]
        D_all_localz = apply_redshift_all(D_rest, zgrid_local, lamb_rest, lamb_obs, filters=filters, conv=conv_finegrid)
        if D_sys is not None:
            D_all_localz = np.concatenate((np.broadcast_to(D_sys, (D_all_localz.shape[0], *D_sys.shape)), D_all_localz), axis=1)
        coefs_zgrid_local = np.zeros((local_length, D_allz.shape[1]))
        bs_zgrid_local = np.zeros_like(zgrid_local)
        cost_zgrid_local = np.zeros_like(zgrid_local) + np.inf
        # alpha_path_list_local = List()
        alpha_path_list_local = []
        # larslasso_step_zgrid_local = np.zeros_like(zgrid_local)

        for j in range(local_length):
            D_this_localz = D_all_localz[j]
            if not larslasso:
                coef, model, _ = fit_models_ols(D_this_localz, spec_obs, err_obs)
                b = np.array(0.0)
                cost = np.sum((model - spec_obs)**2/err_obs**2)
                alpha_path = np.array([0.0])
            else:
                # spec_obs = np.ascontiguousarray(spec_obs)
                # err_obs = np.ascontiguousarray(err_obs)
                coef, b, model, cost, _, alpha_path = fit_model_larslasso(D_this_localz, spec_obs, err_obs, alpha=alpha, alpha_sig=alpha_sig, positive=positive, 
                                                    center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator, 
                                                    max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, max_feature=max_feature, 
                                                    active_alpha_0=active_alpha_0, active_alpha_0_fitting_penalty=active_alpha_0_fitting_penalty, active_ols=active_ols)

            coefs_zgrid_local[j] = coef
            bs_zgrid_local[j] = b
            cost_zgrid_local[j] = cost
            alpha_path_list_local.append(alpha_path)
            # larslasso_step_zgrid_local[j] = larslasso_step

        # combine coefs, chi2, D_allz, zgrid with local arrays
        idx_zgrid_local = np.searchsorted(zgrid, zgrid_local)
        zgrid = np.insert(zgrid, idx_zgrid_local, zgrid_local)
        coefs_zgrid = np.insert(coefs_zgrid, idx_zgrid_local, coefs_zgrid_local, axis=0)
        bs_zgrid = np.insert(bs_zgrid, idx_zgrid_local, bs_zgrid_local)
        cost_zgrid = np.insert(cost_zgrid, idx_zgrid_local, cost_zgrid_local)
        # larslasso_step_zgrid = np.insert(larslasso_step_zgrid, larslasso_step_zgrid_local)
        D_allz = np.insert(D_allz, idx_zgrid_local, D_all_localz, axis=0)

        # insert alpha_list_local to alpha_path_list in reverse order so that index doesn't change
        for insert_idx in reversed(range(len(idx_zgrid_local))):
            alpha_path_list.insert(idx_zgrid_local[insert_idx], alpha_path_list_local[insert_idx])

        if zinput is not None:
            idx_zpeak0_shift = np.searchsorted(idx_zgrid_local, idx_zpeak0, side='right')   
            # get the index of idx_zpeak0 within idx_zgrid_local, use side='right' so that if there are same index in idx_zgrid_local,
            # they are still counted and the actual shift is accurate, since idx_zgrid_local is before actual shifts
            idx_zpeak1 = idx_zpeak0 + idx_zpeak0_shift
            zpeak1 = zinput
        else:
            idx_zpeak1 = np.argmin(cost_zgrid)
            zpeak1 = zgrid[idx_zpeak1]

    # Now we have zpeak1, idx_zpeak1, zgrid, coefs_zgrid, cost_zgrid
    if (zinput is not None) and (not error):
        zmean = zinput
        zmedian = zinput
        zlower = zinput
        zupper = zinput
        zstd = 0.0
        ztrue_pit = -99.0
        pdf = -99.0
    else:
        pdf = np.exp(-(cost_zgrid - min(cost_zgrid))/2)
        if pz is not None:
            if pz.shape[0] != zgrid.shape[0]:
                pz = np.interp(zgrid, zgrid_original, pz)
            pdf = pdf * pz
            pdf = pdf / np.max(pdf)
        zpdf_outputs = evaluate_zpdf(zgrid=zgrid, pdf=pdf, error=error, error_probline=probline, ztrue=zinput, ztrue_idx=idx_zinput)
        zmean, zmedian, _, zlower, zupper, zstd, _, _, ztrue_pit = zpdf_outputs

    # if not error:
    #     zlower = zpeak1
    #     zupper = zpeak1
    #     zstd = 0.0
    #     ztrue_pit = -99.0
    # else:
    #     zlower, zupper, ztrue_pit = error_estimation(zgrid=zgrid, chi2=cost_zgrid, probline=probline, ztrue=zinput, ztrue_idx=idx_zinput)

    D_zpeak = D_allz[idx_zpeak1]
    coef_zpeak = coefs_zgrid[idx_zpeak1]
    b_zpeak = np.array(bs_zgrid[idx_zpeak1])
    # chi2_zpeak = cost_zgrid[idx_zpeak1]
    model_zpeak = D_zpeak.T @ coef_zpeak
    alpha_path_zpeak = alpha_path_list[idx_zpeak1]
    # larslasso_step_peak = larslasso_step_zgrid[idx_zpeak1]

    # calculate averaged z_best
    if (zinput is not None) and (fix_at_zinput):
        zbest = zinput
        # zstd = 0.0
    else:
        # min_cost = np.min(cost_zgrid)
        # likelihood = np.exp(-(cost_zgrid-min_cost)/2)
        # area = np.trapezoid(likelihood, zgrid)
        # # likelihood = likelihood/area
        # zbest = np.trapezoid((likelihood*zgrid), zgrid)/area
        # zstd = np.sqrt(np.trapezoid((likelihood*zgrid**2), zgrid)/area - zbest**2)
        zbest = zmean
    
    if output_all_coefs:
        # coef_out = coefs_zgrid
        coef_rest_out = np.pad(coefs_zgrid[:, Ndicts_sys_now:], ((0,0), (0, Ndicts_original-Ndicts_now)), mode='constant', constant_values=0.0)
        coef_sys_out = np.pad(coefs_zgrid[:, :Ndicts_sys_now], ((0,0), (0, Ndicts_sys_original-Ndicts_sys_now)), mode='constant', constant_values=0.0)
        # coef_out = np.hstack([coef_rest_out, coef_sys_out])
        coef_out = np.hstack([coef_sys_out, coef_rest_out])

    else:
        # coef_out = coef_zpeak
        coef_rest_out = np.pad(coef_zpeak[Ndicts_sys_now:], (0, Ndicts_original-Ndicts_now), mode='constant', constant_values=0.0)
        coef_sys_out = np.pad(coef_zpeak[:Ndicts_sys_now], (0, Ndicts_sys_original-Ndicts_sys_now), mode='constant', constant_values=0.0)
        # coef_out = np.hstack([coef_rest_out, coef_sys_out])
        coef_out = np.hstack([coef_sys_out, coef_rest_out])

    logpdf = -cost_zgrid/2
    # pdf = np.exp(-(cost_zgrid - min(cost_zgrid))/2)

    outputs_dict = {
        'zpeak': zpeak1,
        'zbest': zbest,
        'zmed': zmedian,
        'zlower': zlower,
        'zupper': zupper,
        'zstd': zstd,
        'coef': coef_out,
        'b': b_zpeak,
        'model': model_zpeak,
        'zgrid': zgrid,
        'cost_zgrid': cost_zgrid,
        'alpha_path': alpha_path_zpeak,
        'ztrue_pit': ztrue_pit,
        'logpdf': logpdf,
        'pdf': pdf
    }
    return outputs_dict


# fit zgrid that marginalize over coefficients
def marginalize_zgrid(lamb_obs, spec_obs, err_obs, dictionary=None,
                      lamb_rest=None, D_rest=None, D_sys=None, D_allz=None, 
                      D_rest_alphas=None, D_rest_alphas_ndict=None, D_sys_alphas=None, D_sys_alphas_ndict=None,
                      D_rest_list=None, D_allz_list=None, D_rest_list_tab=None,
                      zgrid=None, zinput=None, filters=None, median_ynorm=None, median_spectra=None,
                      alpha=0, alpha_sig=0.0, cdf_tol=1e-3, cdf_max_steps=200,
                      dz=0.002, zmin=0.0, zmax=3.0, scale_1plusz=True, probline=0.317/2, conv=False,
                      zoutput=None, pz=None, marginalize_active_only=True):

    use_D_allz_list = False
    if (D_rest_list is not None or D_allz_list is not None) and D_rest_list_tab is not None:
        use_D_allz_list = True
        if lamb_rest is None:
            raise ValueError("lamb_rest should be provided when using D_rest_list!")
        D_rest = None
        D_rest_alphas = None
        D_rest_alphas_ndict = None
        D_sys = None
        D_sys_alphas = None
        D_sys_alphas_ndict = None
        if D_allz_list is None:
            D_allz_list = []
            for i in range(len(D_rest_list)):
                D_allz_list.append(apply_redshift_all(D_rest_list[i], zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv))

    # if dictionary object is provided, read info from it
    elif dictionary is not None:
        lamb_rest = dictionary.lamb_rest
        D_rest = dictionary.D_rest
        D_rest_alphas = dictionary.D_rest_alphas
        D_rest_alphas_ndict = dictionary.D_rest_alphas_ndict
        D_sys = dictionary.D_sys
        D_sys_alphas = dictionary.D_sys_alphas
        D_sys_alphas_ndict = dictionary.D_sys_alphas_ndict
        D_allz = dictionary.D_allz
        if (D_allz is None) or (conv != dictionary._D_allz_filtered):
            # if conv and dictionary._D_allz_filtered do not match, rebuild D_allz
            if (conv != dictionary._D_allz_filtered):
                print(f'libs.marginalize_zgrid: conv={conv}, dictionary._D_allz_filtered={dictionary._D_allz_filtered}, rebuild D_allz')
            D_allz = dictionary.make_D_allz(lamb_obs, zgrid=zgrid, filters=filters, conv=conv)
    # determine original Ndict
    if use_D_allz_list:
    # if in D_rest_list mode, Ndicts_original is max number of dictionary
        Ndicts_original = D_allz_list[-1].shape[1]
    elif D_rest is not None:
        Ndicts_original = D_rest.shape[0]
    elif D_allz is not None:
        Ndicts_original = D_allz.shape[1]
    else:
        Ndicts_original = 0

    # determine original Ndict_sys
    if D_sys is not None:
        Ndicts_sys_original = D_sys.shape[0]
    else:
        Ndicts_sys_original = 0

    # if median_spectra is set to > 1, pre-process input signal with a median filter
    if (median_spectra is not None) and (median_spectra > 1):
        spec_obs = median_filter(spec_obs, size=median_spectra)

    # calculate ynorm first and use D_rest_alphas to determine how many templates to use
    if D_rest_alphas is not None or D_sys_alphas is not None or use_D_allz_list:
        if (median_ynorm is None) or (median_ynorm == 1):
            ynorm = np.linalg.norm(spec_obs/err_obs)
        else:
            ynorm = np.linalg.norm(median_filter(spec_obs, size=median_ynorm)/err_obs)
        if ynorm != 0:
            alpha_now = 1/ynorm
        else:
            # safeguard against some 0 flux objects having ynorm=0
            alpha_now = 0.2
    else:
        ynorm = np.linalg.norm(spec_obs/err_obs)

    # if D_rest_alphas is given, calculate SNR and select the correct number of dictionaries to be used based on D_rest_alphas_ndict
    if D_rest_alphas is not None:
        idx_Ndicts_now = np.searchsorted(-D_rest_alphas, -alpha_now, side='right') - 1
        if idx_Ndicts_now < 0:
            idx_Ndicts_now = 0
        Ndicts_now = D_rest_alphas_ndict[idx_Ndicts_now]
        if D_rest is not None:
            D_rest = D_rest[:Ndicts_now]
        if D_allz is not None:
            D_allz = D_allz[:, :Ndicts_now]
    elif use_D_allz_list:
        idx_D_list = get_D_rest_list_idx(alpha=alpha_now, D_rest_list_tab=D_rest_list_tab)
        if D_rest_list is not None:
            D_rest = D_rest_list[idx_D_list]
            Ndicts_now = D_rest.shape[0]
        if D_allz_list is not None:
            D_allz = D_allz_list[idx_D_list]
            Ndicts_now = D_allz.shape[1]
        # else:
            # D_allz = apply_redshift_all(D_rest, zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv)
    else:
        Ndicts_now = Ndicts_original

    # if D_sys_alphas is given, calculate SNR and select the correct number of sys dictionaries to be used based on D_sys_alphas_ndict
    if D_sys_alphas is not None:
        idx_Ndicts_sys_now = np.searchsorted(-D_sys_alphas, -alpha_now, side='right') - 1
        if idx_Ndicts_sys_now < 0:
            idx_Ndicts_sys_now = 0
        Ndicts_sys_now = D_sys_alphas_ndict[idx_Ndicts_sys_now]
        if D_sys is not None:
            D_sys = D_sys[:Ndicts_sys_now]
    else:
        Ndicts_sys_now = Ndicts_sys_original

    # marginalize_zgrid always evaluate the whole zgrid
    # if zgrid is not given, create a default zgrid    
    if zgrid is None:
        zgrid = generate_zgrid(zmin=zmin, zmax=zmax, dz=dz, scale_1plusz=scale_1plusz) 
    else:
        zgrid_limit = (zgrid<=zmax) & (zgrid>=zmin)
        zgrid = zgrid[zgrid_limit]
    # zgrid_original = zgrid.copy()   # keep a copy of original zgrid before inserting zinput for pz

    # check if D_allz is given
    if D_allz is None:
        D_allz = apply_redshift_all(D_rest, zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv)

    # insert D_sys to the beginning of each element of D_allz
    if D_sys is not None:
        D_allz = np.concatenate((np.broadcast_to(D_sys, (D_allz.shape[0], *D_sys.shape)), D_allz), axis=1)

    # determine modified alpha_sig if median keywords are set
    if (median_ynorm is not None) and (median_ynorm > 1) and (median_spectra is None):
        # adjust alpha_sig
        original_1sig = 1/np.linalg.norm(spec_obs/err_obs)
        if np.isnan(original_1sig):
            original_1sig = 0.2
        new_1sig = 1/np.linalg.norm(median_filter(spec_obs, size=median_ynorm)/err_obs)
        if np.isnan(new_1sig):
            new_1sig = 0.2
        one_sig_ratio = new_1sig/original_1sig
        alpha_sig = alpha_sig * one_sig_ratio

    # calculate the actual alpha value used in evaluation
    alpha = alpha + 1/ynorm * alpha_sig

    y = spec_obs/err_obs
    log_pdfs = np.zeros(zgrid.shape[0])
    for i in range(zgrid.shape[0]):
        if marginalize_active_only:
            D_thisz = D_allz[i]
            coef, _, _, _, _, _ = fit_model_larslasso(
                D_thisz, 
                spec_obs, 
                err_obs, 
                alpha=alpha, 
                alpha_sig=alpha_sig, 
                positive=True, 
                center_Xy=True, 
                unit_X=True, 
                unit_y=True, 
                max_iter=200, 
                decimals=10)
            active_mask = coef > 0
            D_thisz = D_allz[i][active_mask]
        else:
            D_thisz = D_allz[i]

        if D_thisz.shape[0] > 0:
            X = (D_thisz/err_obs).T
            log_pdfs[i] = logpdf_D_thisz(X, y, alpha=alpha, ynorm=ynorm, tol=cdf_tol, max_steps=cdf_max_steps)
        else:
            log_pdfs[i] = -np.inf

    pdf = np.exp(log_pdfs - max(log_pdfs))
    if pz is not None:
        # if pz.shape[0] != zgrid.shape[0]:
            # pz = np.interp(zgrid, zgrid_original, pz)
        pdf = pdf * pz
        pdf = pdf / np.max(pdf)
    zpdf_out = evaluate_zpdf(zgrid, pdf=pdf, ztrue=zinput, error_probline=probline)
    z_mean, z_median, z_pdfmax, zl, zu, zstd, _, _, ztrue_pit = zpdf_out
    
    outputs_dict = {
        'zpeak': z_pdfmax,
        'zbest': z_mean,
        'zmed': z_median,
        'zlower': zl,
        'zupper': zu,
        'zstd': zstd,
        'zgrid': zgrid,
        'ztrue_pit': ztrue_pit,
        'logpdf': log_pdfs,
        'pdf': pdf
    }

    if zoutput is not None and D_rest is not None:
        if zoutput == 'zbest':
            zout = z_mean
        elif zoutput == 'zmed':
            zout = z_median
        elif zoutput == 'z_pdfmax' or zoutput == 'zpeak':
            zout = z_pdfmax
        else:
            zout = zoutput
        D_zout = apply_redshift(D_rest, zout, lamb_rest, lamb_obs, filters=filters, conv=conv)
        if D_sys is not None:
            D_out = np.vstack((D_sys, D_zout))
        else:
            D_out = D_zout
        coef_zout, b_out, model_out, cost_out, _, alpha_path = fit_model_larslasso(D_out, 
                                                                          spec_obs, 
                                                                          err_obs, 
                                                                          alpha=alpha, 
                                                                          alpha_sig=0.0, # alpha already includes alpha_sig from above
                                                                          positive=True, 
                                                                          path=False)
        coef_rest_out = np.pad(coef_zout[Ndicts_sys_now:], (0, Ndicts_original-Ndicts_now), mode='constant', constant_values=0.0)
        coef_sys_out = np.pad(coef_zout[:Ndicts_sys_now], (0, Ndicts_sys_original-Ndicts_sys_now), mode='constant', constant_values=0.0)
        coef_out = np.hstack([coef_sys_out, coef_rest_out])
    else:
        coef_out = None
        b_out = None
        cost_out = None
        model_out = None
        alpha_path = None

    outputs_dict['coef'] = coef_out
    outputs_dict['b'] = b_out
    outputs_dict['cost'] = cost_out
    outputs_dict['model'] = model_out
    outputs_dict['alpha_path'] = alpha_path

    return outputs_dict



# fit_zgrid function for multiprocessing input
def fit_zgrid_multiprocessing(child, 
                              cat=None, 
                              lamb_obs=None,
                              ztrue=None,
                              spec_obs=None,
                              err_obs=None,
                              dictionary=None, 
                              lamb_rest=None, 
                              D_rest=None, 
                              D_sys=None, 
                              D_allz=None, 
                              D_rest_alphas=None, 
                              D_rest_alphas_ndict=None, 
                              D_sys_alphas=None,
                              D_sys_alphas_ndict=None,
                              D_rest_list=None,
                              D_allz_list=None,
                              D_rest_list_tab=None,
                              fit_zgrid_validation_kws=None, 
                              marginalize_zgrid_validation_kws=None,
                              zgrid=None, 
                              filters=None,
                              marginalization=False,
                              marginalize_active_only=True,
                              marginalization_min_alpha=0.0,
                              marginalization_max_alpha=np.inf,
                              partial_marginalization_min_alpha=0.0,
                              partial_marginalization_max_alpha=np.inf,
                              return_logpdf=False,
                              pz=None,
                              verbose=False):

    if zgrid is None and return_logpdf:
        zgrid = generate_zgrid()

    conv = fit_zgrid_validation_kws['conv'] # both kws have the same conv setting
    if cat is not None:
        lamb_obs = cat.lamb_obs
        ztrue = cat.ztrue
        spec_obs = cat.spec_obs
        err_obs = cat.err_obs

    use_D_allz_list = False
    if (D_rest_list is not None or D_allz_list is not None) and D_rest_list_tab is not None:
        use_D_allz_list = True
        D_rest = None
        D_rest_alphas = None
        D_rest_alphas_ndict = None
        D_sys = None
        D_sys_alphas = None
        D_sys_alphas_ndict = None
        if D_allz_list is None:
            D_allz_list = []
            for i in range(len(D_rest_list)):
                D_allz_list.append(apply_redshift_all(D_rest_list[i], zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv))

    elif dictionary is not None:
        lamb_rest = dictionary.lamb_rest
        D_rest = dictionary.D_rest
        D_rest_alphas = dictionary.D_rest_alphas
        D_rest_alphas_ndict = dictionary.D_rest_alphas_ndict
        D_sys = dictionary.D_sys
        D_sys_alphas = dictionary.D_sys_alphas
        D_sys_alphas_ndict = dictionary.D_sys_alphas_ndict
        D_allz = dictionary.D_allz
        if (D_allz is None) or (conv != dictionary._D_allz_filtered):
            # if conv and dictionary._D_allz_filtered do not match, rebuild D_allz
            if (conv != dictionary._D_allz_filtered):
                print(f'libs.fit_zgrid_multiprocessing: conv={conv}, dictionary._D_allz_filtered={dictionary._D_allz_filtered}, rebuild D_allz')
            D_allz = dictionary.make_D_allz(lamb_obs, zgrid=zgrid, filters=filters, conv=conv)

    if not use_D_allz_list:
        Ndict = D_rest.shape[0]
    else:
        Ndict = D_allz_list[-1].shape[1]
    if D_sys is not None:
        Ndict_sys = D_sys.shape[0]
    else:
        Ndict_sys = 0

    Ngal = len(ztrue)
    zpeak_fitted = np.zeros(Ngal)
    zbest_fitted = np.zeros(Ngal)
    zmed_fitted = np.zeros(Ngal)
    zlow_fitted = np.zeros(Ngal)
    zhigh_fitted = np.zeros(Ngal)
    zstd_fitted = np.zeros(Ngal)
    coefs_fitted = np.zeros((Ngal, Ndict+Ndict_sys))
    bs_fitted = np.zeros(Ngal)
    # cost_zgrid = np.zeros(Ngal) + np.inf
    # last_alpha = np.zeros(Ngal)
    ztrue_pit_fitted = np.zeros(Ngal)
    if return_logpdf:
        logpdf_fitted = np.zeros((Ngal, len(zgrid)))
    else:
        logpdf_fitted = None

    for i in range(Ngal):
        if verbose:
            print(f"\rSegment Redshift Estimation:\t\t{i+1}/{Ngal} sources", end="")
        spec_obs_i = np.ascontiguousarray(spec_obs[i])
        err_obs_i = np.ascontiguousarray(err_obs[i])
        alpha_i = 1/np.linalg.norm(spec_obs_i/err_obs_i)

        if marginalization and (alpha_i >= marginalization_min_alpha) and (alpha_i < marginalization_max_alpha):
            if marginalize_active_only and ((alpha_i >= partial_marginalization_min_alpha) and (alpha_i < partial_marginalization_max_alpha)):
                marginalize_active_only_i = True
            else:
                marginalize_active_only_i = False
            fit_out = marginalize_zgrid(
                lamb_obs, 
                spec_obs_i,
                err_obs_i,
                lamb_rest=lamb_rest,
                D_rest=D_rest, 
                D_sys=D_sys, 
                D_allz=D_allz, 
                D_rest_alphas=D_rest_alphas, 
                D_rest_alphas_ndict=D_rest_alphas_ndict,
                D_sys_alphas=D_sys_alphas,
                D_sys_alphas_ndict=D_sys_alphas_ndict,
                D_rest_list=D_rest_list,
                D_allz_list=D_allz_list,
                D_rest_list_tab=D_rest_list_tab,
                zinput=ztrue[i], 
                zgrid=zgrid, 
                filters=filters, 
                pz=pz, 
                marginalize_active_only=marginalize_active_only_i,
                **marginalize_zgrid_validation_kws
                )

        else:
            fit_out = fit_zgrid(
                lamb_obs, 
                spec_obs_i, 
                err_obs_i, 
                lamb_rest=lamb_rest, 
                D_rest=D_rest, 
                D_sys=D_sys, 
                D_allz=D_allz, 
                D_rest_alphas=D_rest_alphas, 
                D_rest_alphas_ndict=D_rest_alphas_ndict,
                D_sys_alphas=D_sys_alphas,
                D_sys_alphas_ndict=D_sys_alphas_ndict,
                D_rest_list=D_rest_list,
                D_allz_list=D_allz_list,
                D_rest_list_tab=D_rest_list_tab,
                zinput=ztrue[i], 
                zgrid=zgrid, 
                filters=filters, 
                pz=pz, 
                **fit_zgrid_validation_kws)
            # zpeak, zbest, zlow, zhigh, coef, _, model, zgrid, _, alpha_path = fit_out
        zpeak = fit_out['zpeak']
        zbest = fit_out['zbest']
        zmed = fit_out['zmed']
        zlow = fit_out['zlower']
        zhigh = fit_out['zupper']
        zstd = fit_out['zstd']
        coef = fit_out['coef']
        b = fit_out['b']
        # model = fit_out['model']
        # alpha_path = fit_out['alpha_path']
        ztrue_pit = fit_out['ztrue_pit']

        zpeak_fitted[i] = zpeak
        zbest_fitted[i] = zbest
        zmed_fitted[i] = zmed
        zlow_fitted[i] = zlow
        zhigh_fitted[i] = zhigh
        zstd_fitted[i] = zstd
        coefs_fitted[i] = coef
        bs_fitted[i] = b
        # idx_minchi2 = np.argmin(cost_zgrid)
        # cost_zgrid[i] = np.sum(((spec_obs[i]-model)/err_obs[i])**2)
        ztrue_pit_fitted[i] = ztrue_pit
        if return_logpdf:
            logpdf_fitted[i] = fit_out['logpdf']
        # if alpha_path.shape[0] == 1:
        #     last_alpha[i] = alpha_path[0]
        # else:
        #     last_alpha[i] = alpha_path[-2]

    # results = (zpeak_fitted, zbest_fitted, zlow_fitted, zhigh_fitted, zstd_fitted, coefs_fitted, cost_zgrid, last_alpha, ztrue_pit_fitted)
    results = (zpeak_fitted, zbest_fitted, zmed_fitted, zlow_fitted, zhigh_fitted, zstd_fitted, coefs_fitted, bs_fitted, ztrue_pit_fitted, logpdf_fitted)
    child.send(results)
    child.close()


# function to fit entire catalog using multiprocessing
def fit_cat_mp(cat, 
               lamb_obs=None, 
               zgrid=None, 
               dictionary=None, 
               nthread=4, 
               lamb_rest=None, 
               D_rest=None, 
               D_sys=None, 
               D_allz=None, 
               D_rest_alphas=None, 
               D_rest_alphas_ndict=None, 
               D_sys_alphas=None, 
               D_sys_alphas_ndict=None,
               D_rest_list=None,
               D_allz_list=None,
               D_rest_list_tab=None,
               fit_zgrid_validation_kws=None, 
               marginalize_zgrid_validation_kws=None, 
               filters=None, 
               marginalization=False, 
               marginalize_active_only=True, 
               marginalization_min_alpha=0.0,
               marginalization_max_alpha=np.inf,
               partial_marginalization_min_alpha=0.0,
               partial_marginalization_max_alpha=np.inf,
               pz=None, 
               return_logpdf=False,
               ):
    if zgrid is None and return_logpdf:
        zgrid = generate_zgrid()

    if lamb_obs is None:
        lamb_obs = cat.lamb_obs
    ztrue = cat.ztrue
    spec_obs = cat.spec_obs
    err_obs = cat.err_obs
    Ndat = cat.Ndat

    fit_zgrid_validation_kws = fit_zgrid_validation_kws or {}
    fit_zgrid_validation_kws.setdefault('alpha', 0.0)
    fit_zgrid_validation_kws.setdefault('alpha_sig', 1.0)
    fit_zgrid_validation_kws.setdefault('conv', False)
    # if not marginalization:
    fit_zgrid_validation_kws.setdefault('larslasso', True)
    fit_zgrid_validation_kws.setdefault('positive', True)
    fit_zgrid_validation_kws.setdefault('error', True)

    marginalize_zgrid_validation_kws = marginalize_zgrid_validation_kws or {}
    marginalize_zgrid_validation_kws.setdefault('alpha', 0.0)
    marginalize_zgrid_validation_kws.setdefault('alpha_sig', 1.0)
    marginalize_zgrid_validation_kws.setdefault('zoutput', 'zpeak')
    marginalize_zgrid_validation_kws.setdefault('conv', False)

    conv = fit_zgrid_validation_kws['conv'] # both kws have the same conv setting

    use_D_allz_list = False
    if (D_rest_list is not None or D_allz_list is not None) and D_rest_list_tab is not None:
        use_D_allz_list = True
        D_rest = None
        D_rest_alphas = None
        D_rest_alphas_ndict = None
        D_sys = None
        D_sys_alphas = None
        D_sys_alphas_ndict = None
        if D_allz_list is None:
            D_allz_list = []
            for i in range(len(D_rest_list)):
                D_allz_list.append(apply_redshift_all(D_rest_list[i], zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv))

    elif dictionary is not None:
        lamb_rest = dictionary.lamb_rest
        D_rest = dictionary.D_rest
        D_rest_alphas = dictionary.D_rest_alphas
        D_rest_alphas_ndict = dictionary.D_rest_alphas_ndict
        D_sys = dictionary.D_sys
        D_sys_alphas = dictionary.D_sys_alphas
        D_sys_alphas_ndict = dictionary.D_sys_alphas_ndict
        D_allz = dictionary.D_allz
        if (D_allz is None) or (conv != dictionary._D_allz_filtered):
            # if conv and dictionary._D_allz_filtered do not match, rebuild D_allz
            if (conv != dictionary._D_allz_filtered):
                print(f'libs.fit_cat_mp: conv={conv}, dictionary._D_allz_filtered={dictionary._D_allz_filtered}, rebuild D_allz')
            D_allz = dictionary.make_D_allz(lamb_obs, zgrid=zgrid, filters=filters, conv=conv)
    Ndict_total = 0
    if D_rest is not None:
        Ndict_total = Ndict_total + D_rest.shape[0]
    if D_sys is not None:
        Ndict_total = Ndict_total + D_sys.shape[0]
    if use_D_allz_list:
        Ndict_total = Ndict_total + D_allz_list[-1].shape[1]

    # in case dictionary and D_allz are not provided, build it here
    if D_allz is None and not use_D_allz_list:
        D_allz = apply_redshift_all(D_rest, zgrid, lamb_rest, lamb_obs, filters=filters, conv=conv)

    # sort catalog by ynorm and partition them roughly equally to nthread bins so the subprocess runtime are roughly equal
    ys = spec_obs/err_obs
    ynorms = np.linalg.norm(ys, axis=1)
    idx_sort = np.argsort(ynorms)[::-1]
    group_idx = [idx_sort[i::nthread] for i in range(nthread)]
    group_spec_obs = [spec_obs[group_idx[i]] for i in range(nthread)]
    group_err_obs = [err_obs[group_idx[i]] for i in range(nthread)]
    group_ztrue = [ztrue[group_idx[i]] for i in range(nthread)]
    group_verbose = [False for i in range(nthread)]
    group_verbose[0] = True

    parents_mplist = []
    process_mplist = []
    for mi in range(nthread):
        ztrue_mi = group_ztrue[mi]
        spec_obs_mi = group_spec_obs[mi]
        err_obs_mi = group_err_obs[mi]

        parent_mi, child_mi = mp.Pipe(duplex=False)
        parents_mplist.append(parent_mi)
        p_mi = mp.Process(target=fit_zgrid_multiprocessing, 
                          args=(child_mi, ), 
                          kwargs={'lamb_obs': lamb_obs,
                                  'ztrue': ztrue_mi,
                                  'spec_obs': spec_obs_mi,
                                  'err_obs': err_obs_mi,
                                  'lamb_rest': lamb_rest,
                                  'D_rest': D_rest,
                                  'D_sys': D_sys,
                                  'D_allz': D_allz,
                                  'D_rest_alphas': D_rest_alphas,
                                  'D_rest_alphas_ndict': D_rest_alphas_ndict,
                                  'D_sys_alphas': D_sys_alphas,
                                  'D_sys_alphas_ndict': D_sys_alphas_ndict,
                                  'D_rest_list': D_rest_list,
                                  'D_allz_list': D_allz_list,
                                  'D_rest_list_tab': D_rest_list_tab,
                                  'fit_zgrid_validation_kws': fit_zgrid_validation_kws,
                                  'marginalize_zgrid_validation_kws': marginalize_zgrid_validation_kws,
                                  'zgrid': zgrid,
                                  'filters': filters,
                                  'marginalization': marginalization, 
                                  'marginalize_active_only': marginalize_active_only,
                                  'marginalization_min_alpha': marginalization_min_alpha,
                                  'marginalization_max_alpha': marginalization_max_alpha,
                                  'partial_marginalization_min_alpha': partial_marginalization_min_alpha,
                                  'partial_marginalization_max_alpha': partial_marginalization_max_alpha,
                                  'pz': pz, 
                                  'return_logpdf': return_logpdf,
                                  'verbose': group_verbose[mi]})
        process_mplist.append(p_mi)
        p_mi.start()

    zpeak_final = np.zeros(Ndat)
    zbest_final = np.zeros_like(zpeak_final)
    zmed_final = np.zeros_like(zpeak_final)
    zlow_final = np.zeros_like(zpeak_final)
    zhigh_final = np.zeros_like(zpeak_final)
    zstd_final = np.zeros_like(zpeak_final)
    coefs_final = np.zeros((Ndat, Ndict_total))
    bs_final = np.zeros_like(zpeak_final)
    ztrue_pit_final = np.zeros_like(zpeak_final)
    if return_logpdf:
        logpdf_final = np.zeros((Ndat, len(zgrid)))
    else:
        logpdf_final = None
    for mi in range(nthread):
        zpeak_mi, zbest_mi, zmed_mi, zlow_mi, zhigh_mi, zstd_mi, coefs_mi, bs_mi, ztrue_pit_mi, logpdf_mi = parents_mplist[mi].recv()
        zpeak_final[group_idx[mi]] = zpeak_mi
        zbest_final[group_idx[mi]] = zbest_mi
        zmed_final[group_idx[mi]] = zmed_mi
        zlow_final[group_idx[mi]] = zlow_mi
        zhigh_final[group_idx[mi]] = zhigh_mi
        zstd_final[group_idx[mi]] = zstd_mi
        coefs_final[group_idx[mi]] = coefs_mi
        bs_final[group_idx[mi]] = bs_mi
        ztrue_pit_final[group_idx[mi]] = ztrue_pit_mi
        if return_logpdf:
            logpdf_final[group_idx[mi]] = logpdf_mi
    try:
        for mi in range(nthread):
            process_mplist[mi].join()
    except KeyboardInterrupt:
        for mi in range(nthread):
            process_mplist[mi].terminate()
        for mi in range(nthread):
            process_mplist[mi].join()

    outputs_dict = {'zpeak': zpeak_final,
                    'zbest': zbest_final,
                    'zmed': zmed_final,
                    'zlower': zlow_final,
                    'zupper': zhigh_final,
                    'zstd': zstd_final,
                    'coefs': coefs_final,
                    'bs': bs_final,
                    'ztrue_pit': ztrue_pit_final,
                    'logpdf': logpdf_final,
                    }
    return outputs_dict


# function similar to fit_zgrid but only output specified redshift model and coef
def fit_z_model(z, lamb_obs, spec_obs, err_obs,
                dictionary=None, lamb_rest=None, D_rest=None, D_sys=None,
                D_rest_alphas=None, D_rest_alphas_ndict=None, D_sys_alphas=None, D_sys_alphas_ndict=None,
                median_ynorm=None, median_spectra=None, path=True,
                alpha=0.0, alpha_sig=0.0, positive=True, center_Xy=False, unit_X=True, unit_y=True, 
                best_estimator=None, max_feature=None, alpha_ns_scaling=False, 
                active_alpha_0=False, active_alpha_0_fitting_penalty=False, active_ols=False,
                conv=False, filters=None):

    if (median_spectra is not None) and (median_spectra > 1):
        median_ynorm = None     # safeguard; if median_spectra, there is no reason to use median_ynorm

    # if dictionary object is provided, read info from it
    if dictionary is not None:
        lamb_rest = dictionary.lamb_rest
        D_rest = dictionary.D_rest
        D_rest_alphas = dictionary.D_rest_alphas
        D_rest_alphas_ndict = dictionary.D_rest_alphas_ndict
        D_sys = dictionary.D_sys
        D_sys_alphas = dictionary.D_sys_alphas
        D_sys_alphas_ndict = dictionary.D_sys_alphas_ndict

    # determine original Ndict
    if D_rest is not None:
        Ndicts_original = D_rest.shape[0]
    else:
        Ndicts_original = 0

    # determine original Ndict_sys
    if D_sys is not None:
        Ndicts_sys_original = D_sys.shape[0]
    else:
        Ndicts_sys_original = 0

    # if median_spectra is set to > 1, pre-process input signal with a median filter
    if (median_spectra is not None) and (median_spectra > 1):
        spec_obs = median_filter(spec_obs, size=median_spectra)

    # calculate ynorm first and use D_rest_alphas to determine how many templates to use
    if D_rest_alphas is not None or D_sys_alphas is not None:
        if (median_ynorm is None) or (median_ynorm == 1):
            ynorm = np.linalg.norm(spec_obs/err_obs)
        else:
            ynorm = np.linalg.norm(median_filter(spec_obs, size=median_ynorm)/err_obs)
        if ynorm != 0:
            alpha_now = 1/ynorm
        else:
            # safeguard against some 0 flux objects having ynorm=0
            alpha_now = 0.2

    # if D_rest_alphas is given, calculate SNR and select the correct number of dictionaries to be used based on D_rest_alphas_ndict
    if D_rest_alphas is not None:
        idx_Ndicts_now = np.searchsorted(-D_rest_alphas, -alpha_now, side='right') - 1
        if idx_Ndicts_now < 0:
            idx_Ndicts_now = 0
        Ndicts_now = D_rest_alphas_ndict[idx_Ndicts_now]
        if D_rest is not None:
            D_rest = D_rest[:Ndicts_now]
    else:
        Ndicts_now = Ndicts_original

    # if D_sys_alphas is given, calculate SNR and select the correct number of sys dictionaries to be used based on D_sys_alphas_ndict
    if D_sys_alphas is not None:
        idx_Ndicts_sys_now = np.searchsorted(-D_sys_alphas, -alpha_now, side='right') - 1
        if idx_Ndicts_sys_now < 0:
            idx_Ndicts_sys_now = 0
        Ndicts_sys_now = D_sys_alphas_ndict[idx_Ndicts_sys_now]
        if D_sys is not None:
            D_sys = D_sys[:Ndicts_sys_now]
    else:
        Ndicts_sys_now = Ndicts_sys_original

    D_zinput = apply_redshift(D_rest, z, lamb_rest, lamb_obs, filters=filters, conv=conv)

    # modify alpha_sig if median keywords are supplied
    if (median_ynorm is not None) and (median_ynorm > 1) and (median_spectra is None):
        # adjust alpha_sig
        original_1sig = 1/np.linalg.norm(spec_obs/err_obs)
        if np.isnan(original_1sig):
            original_1sig = 0.2
        new_1sig = 1/np.linalg.norm(median_filter(spec_obs, size=median_ynorm)/err_obs)
        if np.isnan(new_1sig):
            new_1sig = 0.2
        one_sig_ratio = new_1sig/original_1sig
        alpha_sig = alpha_sig * one_sig_ratio

    coef, b, model, cost, _, alpha_path = fit_model_larslasso(D_zinput, spec_obs, err_obs, alpha=alpha, alpha_sig=alpha_sig, positive=positive, 
                                                              center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator, 
                                                              max_iter=200, decimals=10, alpha_ns_scaling=alpha_ns_scaling, max_feature=max_feature, 
                                                              active_alpha_0=active_alpha_0, active_alpha_0_fitting_penalty=active_alpha_0_fitting_penalty, active_ols=active_ols)

    coef_rest_out = np.pad(coef[Ndicts_sys_now:], (0, Ndicts_original-Ndicts_now), mode='constant', constant_values=0.0)
    coef_sys_out = np.pad(coef[:Ndicts_sys_now], (0, Ndicts_sys_original-Ndicts_sys_now), mode='constant', constant_values=0.0)
    coef_out = np.hstack([coef_sys_out, coef_rest_out])

    outputs_dir = {
        'coef_all': coef_out,
        'coef': coef,
        'b': b,
        'model': model,
        'cost': cost,
        'alpha_path': alpha_path
    }
    return outputs_dir

@njit(fastmath=True)
def evaluate_zpdf(zgrid, 
                  pdf=None, 
                  cost=None, 
                  logpdf=None, 
                  error=True, 
                  error_probline=0.317/2, 
                  ztrue=None, 
                  ztrue_idx=None, 
                  weighted_results=False):
    
    if pdf is not None:
        zpdf = pdf
    elif cost is not None:
        min_cost = np.min(cost)
        zpdf = np.exp(-(cost-min_cost)/2)
    elif logpdf is not None:
        zpdf = np.exp(logpdf-np.max(logpdf))

    nz = zgrid.shape[0]
    dz = np.diff(zgrid)
    dA = dz * 0.5 * (zpdf[0:-1] + zpdf[1:]) # triangular interpolation in midpoints between each zgrid[i] and multiply it by dz
    zcdf1 = np.cumsum(dA)   # Pz CDF from zi=1
    zcdf1 = zcdf1 / np.max(zcdf1)
    zcdf = np.zeros(nz)
    zcdf[1:] = zcdf1    # make zcdf the same length as zgrid

    zpdf_norm = np.trapezoid(zpdf, zgrid)
    z_mean = np.trapezoid(zpdf*zgrid, zgrid) / zpdf_norm
    idx_med = np.searchsorted(zcdf, 0.5)
    z_median = (0.5-zcdf[idx_med-1])/(zcdf[idx_med]-zcdf[idx_med-1]) * dz[idx_med-1] + zgrid[idx_med-1]
    z_pdfmax = zgrid[np.argmax(zpdf)]

    if error:
        prob_l = error_probline
        prob_u = 1 - prob_l
        zidx_l = np.searchsorted(zcdf, prob_l)
        zl = (prob_l-zcdf[zidx_l-1])/(zcdf[zidx_l]-zcdf[zidx_l-1]) * dz[zidx_l-1] + zgrid[zidx_l-1]
        zidx_u = np.searchsorted(zcdf, prob_u)
        zu = (prob_u-zcdf[zidx_u-1])/(zcdf[zidx_u]-zcdf[zidx_u-1]) * dz[zidx_u-1] + zgrid[zidx_u-1]    
        zstd = np.sqrt(np.trapezoid((zpdf*zgrid**2), zgrid)/zpdf_norm - z_mean**2)
    else:
        idx_zpeak = np.argmax(zpdf)
        zl = zgrid[idx_zpeak]
        zu = zl
        zstd = 0.0

    if weighted_results:
        zpdf_w1 = zpdf / (1+zgrid)
        zpdf_w2 = zpdf / (1+zgrid)**2
        dA_w1 = dz * (zpdf_w1[0:-1]+zpdf_w1[1:]) * 0.5
        zcdf1_w1 = np.cumsum(dA_w1)
        zcdf1_w1 = zcdf1_w1 / np.max(zcdf1_w1)
        zcdf_w1 = np.zeros(nz)
        zcdf_w1[1:] = zcdf1_w1

        z_wtd_mean = np.trapezoid(zpdf_w2*zgrid, zgrid) / np.trapezoid(zpdf_w2, zgrid)
        idx_wtd_med = np.searchsorted(zcdf_w1, 0.5)
        z_wtd_median = (0.5-zcdf_w1[idx_wtd_med-1])/(zcdf_w1[idx_wtd_med]-zcdf_w1[idx_wtd_med-1]) * dz[idx_wtd_med-1] + zgrid[idx_wtd_med-1]
    else:
        z_wtd_mean = -99.0
        z_wtd_median = -99.0

    if ztrue is not None:
        if ztrue_idx is not None:
            ztrue_pit = zcdf[ztrue_idx]
        else:
            if np.isin(ztrue, zgrid):
                ztrue_idx = np.searchsorted(zgrid, ztrue)
                ztrue_pit = zcdf[ztrue_idx]
            else:
                ztrue_idx1 = np.searchsorted(zgrid, ztrue) - 1
                ztrue_idx2 = ztrue_idx1 + 1
                ztrue_pit = zcdf[ztrue_idx1] + ((ztrue-zgrid[ztrue_idx1])/(zgrid[ztrue_idx2]-zgrid[ztrue_idx1])) * (zcdf[ztrue_idx2]-zcdf[ztrue_idx1])
    else:
        ztrue_pit = -99.0
    return z_mean, z_median, z_pdfmax, zl, zu, zstd, z_wtd_mean, z_wtd_median, ztrue_pit

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


def logpdf_D_thisz(X, y, alpha, ynorm=None, tol=1e-3, max_steps=200):
    if ynorm is None:
        ynorm = np.lonalg.norm(y)
    P = X.T @ X
    cov = np.linalg.inv(P)
    mu = cov @ X.T @ y
    chi2_ols = np.sum((mu @ X.T - y)**2)
    Xnorms = np.linalg.norm(X, axis=0)

    lmb = ynorm * alpha * Xnorms
    u = mu - cov @ lmb
    log_intJ_term1 = -chi2_ols/2
    log_intJ_term2 = -lmb.T @ mu
    log_intJ_term3 = 0.5 * lmb.T @ cov @ lmb
    log_intJ_term4 = np.log(func.multivariate_gaussian_int(cov))
    logcdf, _, _, _ = func.probit_logcdf(u, cov, tol=tol, max_steps=max_steps)        
    log_pdf = log_intJ_term1 + log_intJ_term2 + log_intJ_term3 + log_intJ_term4 + logcdf
    return log_pdf

# ---------------------------------------------------------------
# fitting functions
# ---------------------------------------------------------------


@jit(nopython=True, fastmath=True)
def fit_models_ols(D_thisz, spec_obs, err_obs):
    X = (D_thisz/err_obs).T
    X = np.ascontiguousarray(X)
    y = spec_obs/err_obs
    coef = func.ols(X, y)
    model = (D_thisz*err_obs).T @ coef
    cov = np.linalg.inv(X.T @ X)
    return coef, model, cov


# model fitting function wrapping larslasso
@jit(nopython=True, fastmath=True)
def fit_model_larslasso(D_thisz, spec_obs, err_obs, alpha=0.0, alpha_sig=0.0, positive=False,  
                        center_Xy=False, unit_X=True, unit_y=True, path=True, best_estimator=None,
                        max_iter=200, decimals=10, max_feature=None, alpha_ns_scaling=False, 
                        active_alpha_0=False, active_alpha_0_fitting_penalty=False, active_ols=False):
# def fit_model_larslasso(D_thisz, spec_obs, err_obs, *args, **kwargs):
    X = (D_thisz/err_obs).T    # currently D_thisz is X.T in usual LARSlasso convention
    X = np.ascontiguousarray(X)
    y = spec_obs/err_obs
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
    alpha = alpha * np.ones_like(alpha_in)
    alpha_in[alpha_in<alpha] = alpha[alpha_in<alpha]

    coef, b, coef_path, alpha_path, est_path = func.larslasso(X, y, alpha=alpha_in, positive=positive, 
                        center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator,
                        max_iter=max_iter, decimals=decimals, max_feature=max_feature, 
                        alpha_ns_scaling=alpha_ns_scaling, active_ols=active_ols)
    last_alpha = alpha_path[-1]
    if active_alpha_0 and ((alpha>0) or (alpha_sig>0)):
        active_idx = np.where(coef!=0)[0]
        X_a = X[:, active_idx]
        coef_a, b, coef_path, alpha_path, est_path = func.larslasso(X_a, y, alpha=0.0, positive=positive, 
                        center_Xy=center_Xy, unit_X=unit_X, unit_y=unit_y, path=path, best_estimator=best_estimator,
                        max_iter=max_iter, decimals=decimals, max_feature=max_feature, 
                        alpha_ns_scaling=alpha_ns_scaling, active_ols=active_ols)
        coef = np.zeros(X.shape[1])
        coef[active_idx] = coef_a
        if not active_alpha_0_fitting_penalty:
            last_alpha = 0.0

    ymodel = X @ coef + b
    model = ymodel * err_obs

    if best_estimator is not None:
        alpha_out = np.full_like(alpha_in, alpha_path[np.argmin(est_path)])
    else:
        # alpha_out = alpha_in
        alpha_out = np.full_like(alpha_in, last_alpha)  # use the last alpha_path as alpha_out
    # alpha_out = alpha_in
    cost = np.sum((y - ymodel)**2) + 2 * (ynorm/ynorm_inside) * (np.sum(np.fabs(alpha_out*Xnorms/Xnorms_inside*coef)))
    # cost = np.sum((y - ymodel)**2)
    return coef, b, model, cost, coef_path, alpha_path

# ---------------------------------------------------------------
# training functions
# ---------------------------------------------------------------
















# ============================================================================================================
# Code graveyard
# ============================================================================================================

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

# @njit(fastmath=True)
# def error_estimation(zgrid, chi2, probline=0.317/2, ztrue=None, ztrue_idx=None):
#     min_chi2 = np.min(chi2)
#     likelihood = np.exp(-(chi2-min_chi2)/2)
#     dz = np.diff(zgrid)
#     d_likelihood = np.diff(likelihood)
#     d_area = dz * (likelihood[:-1] + 0.5*d_likelihood)  # area by treating each connecting points as triangle
#     total_area = np.sum(d_area)
#     likelihood = likelihood / total_area
#     d_area = d_area / total_area

#     cum_d_area_L = np.cumsum(d_area)    # cumulative sum of area from left
#     idx_lower1 = np.argwhere((cum_d_area_L - probline)>0)[0][0] - 1
#     idx_lower2 = idx_lower1 + 1
#     zgrid_area_L = zgrid[1:]    # cumulative area correspond to zgrid[1:]
#     if idx_lower1 == -1:
#         zlower = 0.0
#     else:
#         zlower1 = zgrid_area_L[idx_lower1]
#         zlower2 = zgrid_area_L[idx_lower2]
#         dz_lower = zlower2 - zlower1
#         factorL = (probline - cum_d_area_L[idx_lower1]) / (cum_d_area_L[idx_lower2] - cum_d_area_L[idx_lower1])
#         zlower = zlower1 + factorL * dz_lower

#     cum_d_area_R = np.cumsum(d_area[::-1])    # cumulative sum of area from right
#     idx_upper1 = np.argwhere((cum_d_area_R - probline)>0)[0][0] - 1
#     idx_upper2 = idx_upper1 + 1
#     if idx_upper1 == -1:
#         zupper = zgrid[-1]
#     else:
#         zgrid_area_R = zgrid[:-1][::-1]     # start counting from 2nd to last of zgrid, then reverse
#         zupper1 = zgrid_area_R[idx_upper1]
#         zupper2 = zgrid_area_R[idx_upper2]
#         dz_upper = zupper2 - zupper1
#         factorR = (probline - cum_d_area_R[idx_upper1]) / (cum_d_area_R[idx_upper2] - cum_d_area_R[idx_upper1])
#         zupper = zupper1 + factorR * dz_upper

#     # calculate PIT (probability integral transform) for ztrue
#     if ztrue is not None:
#         if ztrue_idx is not None:
#             ztrue_pit = cum_d_area_L[ztrue_idx-1]   # index for zgrid has to subtract 1 to match culmulative area array
#         else:
#             if np.isin(ztrue, zgrid):
#                 ztrue_idx = np.searchsorted(zgrid, ztrue, side='left')
#                 ztrue_pit = cum_d_area_L[ztrue_idx-1]
#             else:
#                 ztrue_idx1 = np.searchsorted(zgrid, ztrue) - 1
#                 ztrue_idx2 = ztrue_idx1 + 1
#                 ztrue_pit = cum_d_area_L[ztrue_idx1-1] + ((ztrue - zgrid[ztrue_idx1])/(zgrid[ztrue_idx2] - zgrid[ztrue_idx1])) * (cum_d_area_L[ztrue_idx2-1] - cum_d_area_L[ztrue_idx1-1])
#     else:
#         ztrue_pit = -99.0
#     return zlower, zupper, ztrue_pit


