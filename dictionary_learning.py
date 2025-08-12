'''

Config file not specified!

    $ python dictionary_learn_from_config.py config.yaml

Make sure "dictionary_learn_fx.py" is in the same directory as this code.
Output files to OUTPUT/ is default.

'''
import numpy as np
# import pandas as pd
import sys
import time
# from numba import jit, njit
# from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import libs.lib_dictlearn as fx
import libs.diagnostic_plots as dplots
import yaml
from datetime import datetime
import multiprocessing as mp



# define fitting worker function
def fitting_mp(child, cat, lamb_rest, D_rest, D_allz, zgrid, filters, fit_zgrid_validation_kws, verbose=False):
    lamb_obs = cat.lamb_obs
    ztrue = cat.ztrue
    spec_obs = cat.spec_obs
    err_obs = cat.err_obs

    Ngal = len(ztrue)
    zpeak_fitted = np.zeros(Ngal)
    zbest_fitted = np.zeros(Ngal)
    zlow_fitted = np.zeros(Ngal)
    zhigh_fitted = np.zeros(Ngal)
    coefs_fitted = np.zeros((Ngal, D_rest.shape[0]))

    for i in range(Ngal):
        if verbose:
            print(f"\rValidation Catalog Segment Redshift Estimation:\t\t{i+1}/{Ngal} sources", end="")
        spec_obs_i = np.ascontiguousarray(spec_obs[i])
        err_obs_i = np.ascontiguousarray(err_obs[i])
        zpeak,zbest,zlow,zhigh,coefs,b,model,_,_ = fx.fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest, D_rest=D_rest, D_allz=D_allz, zinput=False, 
                                                    zgrid=zgrid, filters=filters, **fit_zgrid_validation_kws)
        zpeak_fitted[i] = zpeak
        zbest_fitted[i] = zbest
        zlow_fitted[i] = zlow
        zhigh_fitted[i] = zhigh
        coefs_fitted[i] = coefs

    results = (zpeak_fitted, zbest_fitted, zlow_fitted, zhigh_fitted, coefs_fitted)
    child.send(results)
    child.close()

if __name__ == "__main__":


    if len(sys.argv) == 1:
        print(__doc__)
        exit()

    start_datetime = datetime.now().isoformat(timespec='seconds')

    # Read input config file with Configs class
    config_file = sys.argv[1]
    config = fx.Configs(config=config_file)

    # TEMP still name variables from config for now
    # Catalog configurations
    training_catalog = config.training_catalog
    validation_catalog = config.validation_catalog
    filter_central_wavelengths = config.filter_central_wavelengths
    Ndat = config.Ndat
    Ndat_validation = config.Ndat_validation
    Ncalibrators = config.Ncalibrators
    use_DESI_flag = config.use_DESI_flag
    calibrator_SNR = config.calibrator_SNR
    f_lambda_mode = config.f_lambda_mode
    # Dictionary input configurations
    dict_read_from_file = config.dict_read_from_file
    add_constant = config.add_constant
    fix_dicts = config.fix_dicts
    Ndict = config.Ndict
    num_EAZY_as_dict = config.num_EAZY_as_dict
    dicts_fluctuation_scaling_const = config.dicts_fluctuation_scaling_const
    dict_fluctuation_scaling_base = config.dict_fluctuation_scaling_base
    # Algorithm configurations
    training = config.training
    Nepoch =config.Nepoch
    algorithm = config.algorithm
    fix_z = config.fix_z
    centering = config.centering
    AB_update_tolerance = config.AB_update_tolerance
    max_AB_loops = config.max_AB_loops
    remove_old_ab_info = config.remove_old_ab_info
    epochs_to_keep = config.epochs_to_keep
    scale_past_data = config.scale_past_data
    separate_training_weights = config.separate_training_weights
    lamb_obs_sep = 3.79 # TEMP
    weights1 = config.weights1
    weights2 = config.weights2
    learning_rate0 = config.learning_rate0
    learning_rate_cali = config.learning_rate_cali
    # LARSlasso configurations
    larslasso = config.larslasso
    larslasso_alpha_train = config.larslasso_alpha_train
    larslasso_alpha_sigma_train = config.larslasso_alpha_sigma_train
    larslasso_alpha_fit = config.larslasso_alpha_fit
    larslasso_alpha_sigma_fit = config.larslasso_alpha_sigma_fit
    larslasso_positive = config.larslasso_positive
    train_best_estimator = config.train_best_estimator
    fit_best_estimator = config.fit_best_estimator
    max_feature = config.max_feature
    active_OLS_training = config.active_OLS_training
    active_OLS_fitting = config.active_OLS_fitting
    center_Xy = config.center_Xy
    unit_X = config.unit_X
    unit_y = config.unit_y
    larslasso_alpha_scaling = config.larslasso_alpha_scaling
    # Fitting configurations
    probline = config.probline
    fit_training_catalog = config.fit_training_catalog
    fit_initial_dicts = config.fit_initial_dicts
    convolve_filter = config.convolve_filter
    last_stage_convolve_filter = config.last_stage_convolve_filter
    fitting_convolve_filter = config.fitting_convolve_filter
    multiprocess = config.multiprocess
    mp_threads = config.mp_threads
    # Zgrid configurations
    zmax = config.zmax
    zmin = config.zmin
    dz = config.dz
    scale_1plusz = config.scale_1plusz
    testing_zgrid = config.testing_zgrid
    # Directory locations
    eazy_templates_location = config.eazy_templates_location
    filter_list = config.filter_list
    output_dirname = config.output_dirname
    Plots_subfolder = config.Plots_subfolder
    parameters_report = config.parameters_report
    # some common keywords for fit_zgrid function as dictionary
    fit_zgrid_training_kws = config.fit_zgrid_training_kws
    fit_zgrid_validation_kws = config.fit_zgrid_validation_kws

    fit_zgrid = fx.fit_zgrid

    # Generate Zgrid based on configuration
    zgrid = fx.generate_zgrid(zmin=zmin, zmax=zmax, dz=dz, scale_1plusz=scale_1plusz, testing_zgrid=testing_zgrid)

    # Create catalog object and save items as variables
    cat = fx.Catalog(pathfile=training_catalog, Ndat=Ndat, centering=centering)
    if filter_central_wavelengths is None:
        lamb_obs = cat.lamb_obs
    else:
        central_wavelengths_tab = np.loadtxt(filter_central_wavelengths, dtype={'names': ('filter','wavelengths'), 'formats': ('U25','f8')})
        lamb_obs = central_wavelengths_tab['wavelengths']
    # lamb_obs = cat.lamb_obs
    ztrue = cat.ztrue
    spec_obs = cat.spec_obs
    spec_obs_original = cat.spec_obs_original
    err_obs = cat.err_obs
    desi_flag = cat.desi_flag
    snr = cat.snr

    # Decide the true Ndat
    Ngal = len(cat.ztrue)
    if Ngal < Ndat:
        Ndat = Ngal

    # 51th channel SNR require for calibration galaxies
    idx_original = np.arange(Ngal)  # index corresponding to read input
    if use_DESI_flag and desi_flag.any():
        h_cali = desi_flag == 1.0
        idx_cali = idx_original[h_cali]
        Ncalibrators = np.sum(h_cali)
    else:
        h_cali = cat.snr_i > calibrator_SNR
        # h_cali = cat.snr_norm > calibrator_SNR
        idx_cali = idx_original[h_cali][:Ncalibrators] # index for calibration galaxies within read input
        use_DESI_flag = False
        Ncalibrators = len(idx_cali)


    # If output_dir doesn't exist, create one
    if output_dirname is None:
        output_dirname = ''
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    if Plots_subfolder is None:
        Plots_subfolder = ''
    plots_dir = Path(output_dir / Plots_subfolder)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if eazy_templates_location is not None:
        eazy_templates_dir = Path(eazy_templates_location)
    if filter_list is not None:
        filters = fx.read_filters(filter_list)
    else:
        filters = None


    # add a tiny number to z=0.0 so that if zinput is given as 0 it will not be considered False
    for i in range(len(ztrue)):
        if ztrue[i] == 0.0:
            ztrue[i] = 0.00000001

    # Read dictionaries from file
    if dict_read_from_file:
        D_rest_input = np.load(dict_read_from_file)
        lamb_rest = D_rest_input['lamb_rest']
        D_rest = D_rest_input['D_rest']
        try:
            D_rest_save_flambda = D_rest_input['save_flambda']
        except:
            D_rest_save_flambda = False
        if Ndict >= len(D_rest):
            Ndict = len(D_rest)
        else:
            D_rest = D_rest[:Ndict]
        D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
        if f_lambda_mode and not D_rest_save_flambda:
            D_rest = fx.fnu2flambda(lamb_rest*1e5, D_rest)
            D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
        elif not f_lambda_mode and D_rest_save_flambda:
            D_rest = fx.flambda2fnu(lamb_rest*1e5, D_rest)
            D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
        if centering:
            D_rest = D_rest - np.mean(D_rest, axis=1)[:,None]
            D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
        if add_constant:
            D_rest = np.vstack((D_rest, np.ones_like(lamb_rest)/np.linalg.norm(np.ones_like(lamb_rest))))
        # lamb_rest_resolution = np.diff(lamb_rest)[0]
        lamb_rest_resolution = np.mean(np.diff(lamb_rest))
    else:
        lamb_rest = np.arange(0.2,6.0,0.01)
        lamb_rest_resolution = 0.01

    # Read EAZY templates for dictionary initialization and also evaluation test if they are given
    if eazy_templates_location is not None and eazy_templates_dir.is_dir():
        templates_EAZY = fx.load_EAZY(lamb_rest, eazy_templates_location)
        if not f_lambda_mode:
            templates_EAZY = fx.flambda2fnu(lamb_rest*10000, templates_EAZY)
        templates_EAZY = np.array(templates_EAZY)
        if centering:
            templates_EAZY = templates_EAZY - np.mean(templates_EAZY, axis=1)[:,None]

    # initialize dictionaries if they are not given
    if not dict_read_from_file:
        # If initializing dictionaries, create them as noise with different level of fluctuation
        dictionary_fluctuation_scaling = np.array([dicts_fluctuation_scaling_const/(dict_fluctuation_scaling_base**i) for i in range(Ndict-num_EAZY_as_dict)])
        D_rest = fx.initialize_dicts(Ndict, dictionary_fluctuation_scaling=dictionary_fluctuation_scaling, templates_EAZY=templates_EAZY, 
                                num_EAZY_as_dict=num_EAZY_as_dict, lamb_rest=lamb_rest, add_constant=add_constant)

    D_rest_initial = D_rest.copy()

    tic = time.time()

    # Grab the Ngal for validation catalog just for output information
    with np.load(validation_catalog) as fitting_dat:
        Ndat_validation = min(len(fitting_dat['z']), Ndat_validation)


    if epochs_to_keep >= Nepoch:
        remove_old_ab_info = False


    # print(f"Algorithm = {algorithm}")
    if training:
        print(f"Training Catalog:\t{training_catalog} ({Ngal} sources)")
    print(f"Validation Catalog:\t{validation_catalog} ({Ndat_validation} sources)")
    if dict_read_from_file:
        print(f"Dictionaries:\t\t{dict_read_from_file}")
    else:
        print(f"{num_EAZY_as_dict} of 7 EAZY templates used as initialized dictionaries")    
    print('')
    print(f"Ndict = {Ndict}")
    print(f"Add constant: {add_constant}")
    if training and (fix_dicts > 0):
        print(f"Fix {fix_dicts} dictionaries from the end of list")
    print(f"Centering: {centering}")    
    if training:
        print('')
        print(f"Fix z in training: {fix_z}")
        print(f"Scale past data: {scale_past_data}")
        if not remove_old_ab_info:
            print(f"Remove old AB info: {remove_old_ab_info}, Scale past data: {scale_past_data}")
        else:
            print(f"Remove old AB info: {remove_old_ab_info} (keeps {epochs_to_keep} Epochs)")
        if separate_training_weights:
            print(f"Separate training weights: Bands 1-4 x{weights1}, Bands 5-6 x{weights2}")
        if use_DESI_flag:
            print(f"Ncalibrators = {Ncalibrators} (DESI Flags)")
        elif Ncalibrators>0:
            print(f"Ncalibrators = {Ncalibrators} (SNR > {calibrator_SNR})")
        if algorithm == 0:
            print(f"Learning rates = {learning_rate0}/{learning_rate_cali}")
        else:
            if larslasso:
                # print(fr"LARSlasso: {larslasso} (alpha={lars_alpha} + {LARSlasso_alpha_sigma} sigma, Positive: {lars_positive}, max_feature={max_feature}, center_Xy={center_Xy})")
                print(f"LARS-lasso: {larslasso}")
                print(f"\talpha (training) = {larslasso_alpha_train} + {larslasso_alpha_sigma_train} sigma")
                print(f"\talpha (fitting) = {larslasso_alpha_fit} + {larslasso_alpha_sigma_fit} sigma")
                print(f"\tPositive = {larslasso_positive}")
                print(f"\tcenter_Xy = {center_Xy}")
                print(f"\tunit_X = {unit_X}")
                print(f"\tunit_y = {unit_y}")
                if max_feature:
                    print(f"\tmax_feature = {max_feature}")
                if train_best_estimator is not None:
                    print(f"\tTraining Best Estimator = {train_best_estimator}")
                if fit_best_estimator is not None:
                    print(f"\tFitting Best Estimator = {fit_best_estimator}")
            else:
                print(f"LARS-lasso: {larslasso}")
        if testing_zgrid:
            print('Testing Zgrid: On')
    else:
        if larslasso:    
            print(f"\talpha (fitting) = {larslasso_alpha_fit} + {larslasso_alpha_sigma_fit} sigma")
            print(f"\tPositive = {larslasso_positive}")
            print(f"\tcenter_Xy = {center_Xy}")
            print(f"\tunit_X = {unit_X}")
            print(f"\tunit_y = {unit_y}")
            if max_feature:
                print(f"\tmax_feature = {max_feature}")
            if train_best_estimator is not None:
                print(f"\tTraining Best Estimator = {train_best_estimator}")
            if fit_best_estimator is not None:
                print(f"\tFitting Best Estimator = {fit_best_estimator}")


    if convolve_filter or last_stage_convolve_filter or fitting_convolve_filter:
        print(f"Convolving filters: 1st stage:{convolve_filter}, 2nd stage:{last_stage_convolve_filter}, fitting:{fitting_convolve_filter}")

    if multiprocess:
        print(f'Validation catalog fitting multiprocessing threads: {mp_threads}')

    if training:
        print('')

        # Initialize A and B matrices for dictionary updates
        if not separate_training_weights:
            A = np.zeros((D_rest.shape[0], D_rest.shape[0]))
            B = np.zeros((len(lamb_rest), D_rest.shape[0]))
        else:
            A1 = np.zeros((D_rest.shape[0], D_rest.shape[0]))
            A2 = np.zeros((D_rest.shape[0], D_rest.shape[0]))
            B1 = np.zeros((len(lamb_rest), D_rest.shape[0]))
            B2 = np.zeros((len(lamb_rest), D_rest.shape[0]))

        if remove_old_ab_info: # save previous coefficients
            # A_history = np.zeros((Ngal, D_rest.shape[0], D_rest.shape[0]))
            # B_history = np.zeros((Ngal, len(lamb_rest), D_rest.shape[0]))
            coefs_history = np.zeros((Ngal, epochs_to_keep, D_rest.shape[0]))
            if not separate_training_weights:
                spec_rest_history = np.zeros((Ngal, epochs_to_keep, lamb_rest.shape[0]))
            else:
                spec_rest1_history = np.zeros((Ngal, epochs_to_keep, lamb_rest.shape[0]))
                spec_rest2_history = np.zeros((Ngal, epochs_to_keep, lamb_rest.shape[0]))
            if scale_past_data:
                prev_igal0 = np.zeros((Ngal, epochs_to_keep), dtype=int)

            # coefs_history = np.zeros((Nepoch, Ngal, D_rest.shape[0]))

        # Training iterations
        for i_epoch in range(Nepoch):

            # pick out calibration galaxies
            idx_other = np.setdiff1d(idx_original, idx_cali)
            np.random.shuffle(idx_cali)
            np.random.shuffle(idx_other)
            idx_shuffle = np.hstack((idx_cali, idx_other))

            for i_gal0 in range(Ngal): 
                i_gal = idx_shuffle[i_gal0]
                # update with number of galaxies processed
                # print(f"\r\t{i_gal0+1}/{Ngal} spectra", end="")
                ndigits_training = int(np.log10(Ngal)) + 1
                str_i_gal0_plus1 = str(i_gal0+1).zfill(ndigits_training)
                print(f"\rTraining epochs: {i_epoch+1}/{Nepoch} ({str_i_gal0_plus1}/{Ngal} sources)", end="")
                # if this is a calibrator galaxy
                if i_gal in idx_cali or fix_z:
                    # use the known redshift
                    zinput = ztrue[i_gal]
                else:
                    # otherwise perform a best-fit for the redshift
                    zinput = False

                # fit this spectrum and obtain the redshift
                spec_obs_i = np.ascontiguousarray(spec_obs[i_gal])
                err_obs_i = np.ascontiguousarray(err_obs[i_gal])
                z, _, zlow, zhigh, coefs, b, model, _, _= fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_D=lamb_rest, D_rest=D_rest, zinput=zinput, 
                                                                        zgrid=zgrid, filters=filters, **fit_zgrid_training_kws)
                # update the spectral dictionary using the residuals between the model and data
                residual = spec_obs[i_gal] - model
                if not separate_training_weights:
                    j_update = np.where((lamb_rest > np.min(lamb_obs)/(1+z)) & (lamb_rest < np.max(lamb_obs)/(1+z)))[0]     # Find overlap between this spectra in rest-frame and dictionary
                else:
                    j_update1 = np.where((lamb_rest > np.min(lamb_obs)/(1+z)) & (lamb_rest < lamb_obs_sep/(1+z)))[0]     # rest frame wavelength of detector 1~4
                    j_update2 = np.where((lamb_rest > lamb_obs_sep/(1+z)) & (lamb_rest < np.max(lamb_obs)/(1+z)))[0]     # detector 5~6
                # j_update_outside = np.setdiff1d(np.arange(len(lamb_rest)), j_update)

                if not separate_training_weights:
                    interpolated_residual = np.interp(lamb_rest[j_update], lamb_obs/(1+z), residual)    # interpolate the residual to these values
                    interpolated_spec_obs = np.interp(lamb_rest[j_update], lamb_obs/(1+z), spec_obs[i_gal])
                else:
                    interpolated_residual1 = np.interp(lamb_rest[j_update1], lamb_obs/(1+z), residual)    # interpolate the residual to these values
                    interpolated_residual2 = np.interp(lamb_rest[j_update2], lamb_obs/(1+z), residual)    # interpolate the residual to these values
                    interpolated_spec_obs1 = np.interp(lamb_rest[j_update1], lamb_obs/(1+z), spec_obs[i_gal])
                    interpolated_spec_obs2 = np.interp(lamb_rest[j_update2], lamb_obs/(1+z), spec_obs[i_gal])

                # construct the model from best fit coefficients and dictionaries
                error_rest = np.interp(lamb_rest, lamb_obs/(1+z), err_obs[i_gal])
                model_rest = (D_rest).T @ coefs + b * error_rest
                if not separate_training_weights:
                    spec_rest = model_rest.copy()
                    spec_rest[j_update] = interpolated_spec_obs    # replace the overlapped part with interpolated observed spectra
                else:
                    spec_rest1 = model_rest.copy()
                    spec_rest2 = model_rest.copy()
                    spec_rest1[j_update1] = interpolated_spec_obs1    # replace the overlapped part with interpolated observed spectra
                    spec_rest2[j_update2] = interpolated_spec_obs2    # replace the overlapped part with interpolated observed spectra
                # this will be considered the observed spectra for update purpose; outside overlap range just use model (no update)

                if scale_past_data:
                    t_current = i_gal0 + i_epoch * Ngal + 1
                    AB_scale_factor = (t_current-1)/t_current
                    if remove_old_ab_info:
                        if i_epoch >= epochs_to_keep:
                            t_prev = prev_igal0[i_gal][0] + (i_epoch-epochs_to_keep)*Ngal + 1
                            old_AB_scale_factor = (t_prev/(t_current-1))
                        else:
                            t_prev = 0
                            old_AB_scale_factor = 0.0
                else:
                    old_AB_scale_factor = 1.0
                    AB_scale_factor = 1.0

                # update each item in the dictionary (do not modify the DC offset term at the end)
                if algorithm == 0:  # pseudo-inverse vector method
                    # set the learning rate
                    learning_rate = learning_rate0
                    # if this is a calibrator galaxy
                    if i_gal in idx_cali:
                        # use a higher learning rate since we know the redshift is correct
                        learning_rate = learning_rate_cali
                    for i in range(D_rest.shape[0]-add_constant-fix_dicts):
                        update_factor = learning_rate * (coefs[i]/(np.sum(coefs**2)))
                        D_rest[i,j_update] = D_rest[i,j_update] + update_factor * interpolated_residual

                elif algorithm == 1:    # block-coordinate descent
                    # update A and B
                    if remove_old_ab_info:
                        # A -= A_history[i_gal]
                        # B -= B_history[i_gal]
                        # A_history[i_gal] = coefs[:,None] @ coefs[:,None].T
                        # # A_history[i_gal] = coefs[:,None] @ coefs[:,None].T * AB_factor
                        # B_history[i_gal] = spec_rest[:,None] @ coefs[:,None].T                    
                        # # B_history[i_gal] = spec_rest[:,None] @ coefs[:,None].T * residue_factor[:,None] * AB_factor
                        # A += A_history[i_gal]
                        # B += B_history[i_gal]

                        # Use first coefficients to calculate A and B to remove
                        coef_old = coefs_history[i_gal, 0].copy()
                        if not separate_training_weights:
                            spec_rest_old = spec_rest_history[i_gal, 0].copy()
                            A_old = coef_old[:,None] @ coef_old[:,None].T
                            B_old = spec_rest_old[:,None] @ coef_old[:,None].T
                            A -= A_old * old_AB_scale_factor
                            B -= B_old * old_AB_scale_factor
                        else:
                            spec_rest1_old = spec_rest1_history[i_gal, 0].copy()
                            spec_rest2_old = spec_rest2_history[i_gal, 0].copy()
                            A1_old = coef_old[:,None] @ coef_old[:,None].T
                            A2_old = coef_old[:,None] @ coef_old[:,None].T
                            B1_old = spec_rest1_old[:,None] @ coef_old[:,None].T
                            B2_old = spec_rest2_old[:,None] @ coef_old[:,None].T
                            A1 -= A1_old * old_AB_scale_factor
                            A2 -= A2_old * old_AB_scale_factor                        
                            B1 -= B1_old * old_AB_scale_factor
                            B2 -= B2_old * old_AB_scale_factor

                        # move igal epochs ahead by 1
                        if epochs_to_keep > 1:
                            coefs_history[i_gal, 0:-1] = coefs_history[i_gal, 1:]
                            coefs_history[i_gal, -1] = coefs
                            if not separate_training_weights:
                                spec_rest_history[i_gal, 0:-1] = spec_rest_history[i_gal, 1:]
                                spec_rest_history[i_gal, -1] = spec_rest
                            else:
                                spec_rest1_history[i_gal, 0:-1] = spec_rest1_history[i_gal, 1:]
                                spec_rest2_history[i_gal, 0:-1] = spec_rest2_history[i_gal, 1:]
                                spec_rest1_history[i_gal, -1] = spec_rest1
                                spec_rest2_history[i_gal, -1] = spec_rest2
                            prev_igal0[i_gal, 0:-1] = prev_igal0[i_gal, 1:]
                            prev_igal0[i_gal, -1] = i_gal0

                        elif epochs_to_keep == 1:
                            coefs_history[i_gal, 0] = coefs.copy()
                            if not separate_training_weights:
                                spec_rest_history[i_gal, 0] = spec_rest.copy()
                            else:
                                spec_rest1_history[i_gal, 0] = spec_rest1.copy()
                                spec_rest2_history[i_gal, 0] = spec_rest2.copy()
                            prev_igal0[i_gal, 0] = i_gal0

                    if not separate_training_weights:
                        A = A * AB_scale_factor + coefs[:,None] @ coefs[:,None].T
                        B = B * AB_scale_factor + spec_rest[:,None] @ coefs[:,None].T
                    else:
                        A1 = A1 * AB_scale_factor + coefs[:,None] @ coefs[:,None].T
                        A2 = A2 * AB_scale_factor + coefs[:,None] @ coefs[:,None].T
                        A = A1 * weights1 + A2 * weights2
                        B1 = B1 * AB_scale_factor + spec_rest1[:,None] @ coefs[:,None].T
                        B2 = B2 * AB_scale_factor+ spec_rest2[:,None] @ coefs[:,None].T
                        B = B1 * weights1 + B2 * weights2

                    # iterative over A and B matrices until converge or max_AB_loop
                    for j in range(D_rest.shape[0]-add_constant-fix_dicts):
                        score = 1.0
                        # Dj_old = D_rest[j].copy()
                        abi = 0
                        while score >= AB_update_tolerance and abi < max_AB_loops:
                            diag_Aj = np.diagonal(A)[j]
                            if diag_Aj != 0:
                                Dj_old = D_rest[j].copy()
                                uj = 1/diag_Aj * (B[:,j] - (D_rest.T @ A[j])) + D_rest[j]
                                uj_norm = np.linalg.norm(uj)
                                D_rest[j] = uj/max(1, uj_norm)
                                score = np.linalg.norm(D_rest[j]-Dj_old)/np.linalg.norm(Dj_old)
                                abi += 1
                            else:
                                break

        # del A, B, coefs_history, spec_rest_history
        if not separate_training_weights:
            del A, B, coefs_history, spec_rest_history
        else:
            del A1, B1, A2, B2, coefs_history, spec_rest1_history, spec_rest2_history
        

        np.savez_compressed(output_dir / 'trained_templates.npz',lamb_rest=lamb_rest, D_rest=D_rest)

    # Make template plots and multiplots
    makefigs = dplots.diagnostic_plots(output_dirname=plots_dir)
    makefigs.template_plots(lamb_rest=lamb_rest, D_rest=D_rest[:Ndict], D_rest_initial=D_rest_initial[:Ndict])


    # pre-redshift D_rest into all redshift in zgrid
    D_allz = fx.apply_redshift_all(D_rest, zgrid=zgrid, lamb_in=lamb_rest, lamb_out=lamb_obs, 
                                    filters=filters, conv=fitting_convolve_filter)
    if training and fit_initial_dicts:
        D_allz_initial = fx.apply_redshift_all(D_rest_initial, zgrid=zgrid, lamb_in=lamb_rest, lamb_out=lamb_obs, 
                                    filters=filters, conv=fitting_convolve_filter)

    # print('')
    # Fitting training catalog if set to True
    if fit_training_catalog:
        # fit all galaxies with final template
        # print('\nTraining Catalog Redshift Estimation')
        zpeak_trained = np.zeros(Ngal)
        zbest_trained = np.zeros(Ngal)
        zlow_trained = np.zeros(Ngal)
        zhigh_trained = np.zeros(Ngal)
        coefs_trained = np.zeros((Ngal, D_rest.shape[0]))

        print('')
        for i in range(Ngal):
            # update with number of galaxies processed
            # print(f"\r\t{i+1}/{Ngal} spectra", end="")
            print(f"\rTraining Catalog Redshift Estimation: {i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            zinput = False
            zpeak,zbest,zlow,zhigh,coefs,b,model,_,_ = fit_zgrid(lamb_obs, spec_obs[i], err_obs[i], lamb_rest, D_rest=D_rest, D_allz=D_allz, zinput=zinput, 
                                                    zgrid=zgrid, filters=filters, **fit_zgrid_validation_kws)
            # store the redshift
            zpeak_trained[i] = zpeak
            zbest_trained[i] = zbest
            zlow_trained[i] = zlow
            zhigh_trained[i] = zhigh
            coefs_trained[i] = coefs

        # for comparison, fit again with original template
        zpeak_initial = np.zeros(Ngal)
        zbest_initial = np.zeros(Ngal)
        # print('\nTraining Catalog Untrained Redshift Estimation')
        for i in range(Ngal):
            print(f"\rTraining Catalog Untrained Redshift Estimation: {i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            zinput = False
            spec_obs_i = np.ascontiguousarray(spec_obs[i])
            err_obs_i = np.ascontiguousarray(err_obs[i])
            zpeak_bi,zbest_bi,zlow_bi,zhigh_bi,coefs_bi,b_bi,model_bi,_,_ = fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest, D_rest=D_rest_initial,
                                                D_allz=D_allz_initial, zinput=zinput, zgrid=zgrid, filters=filters, **fit_zgrid_validation_kws)
            zpeak_initial[i] = zpeak_bi
            zbest_initial[i] = zbest_bi

        makefigs.zp_zs_plots(ztrue=ztrue, z_initial=zpeak_initial, z_trained=zpeak_trained, zmin=zmin, zmax=zmax, catalog='training_zpeak')
        makefigs.zp_zs_plots(ztrue=ztrue, z_initial=zbest_initial, z_trained=zbest_trained, zmin=zmin, zmax=zmax, catalog='training_zbest')

        # save estimated redshifts
        np.savez_compressed(output_dir / 'estimated_redshifts_training.npz', ztrue=ztrue, zpeak=zpeak_trained, zbest=zbest_trained, zlow=zlow_trained, zhigh=zhigh_trained, 
             zpeak_initial=zpeak_initial, zbest_initial=zbest_initial, coefs=coefs_trained, idx_cali=idx_cali)

    # Read validation catalog for final fitting
    cat = fx.Catalog(pathfile=validation_catalog, Ndat=Ndat_validation, centering=centering)
    ztrue = cat.ztrue
    print('')

    if not multiprocess:

        lamb_obs = cat.lamb_obs
        spec_obs = cat.spec_obs
        # spec_obs_original = cat.spec_obs_original
        err_obs = cat.err_obs
        # desi_flag = cat.desi_flag
        # snr = cat.snr_i
        Ngal = len(ztrue)

        zpeak_trained = np.zeros(Ngal)
        zbest_trained = np.zeros(Ngal)
        zlow_trained = np.zeros(Ngal)
        zhigh_trained = np.zeros(Ngal)
        coefs_trained = np.zeros((Ngal, D_rest.shape[0]))

        # print('')
        for i in range(Ngal):
            # update with number of galaxies processed
            # print(f"\r\t{i+1}/{Ngal} spectra", end="")
            print(f"\rValidation Catalog Redshift Estimation:\t\t{i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            zinput = False
            zpeak,zbest,zlow,zhigh,coefs,b,model,_,_ = fit_zgrid(lamb_obs, spec_obs[i], err_obs[i], lamb_rest, D_rest=D_rest, D_allz=D_allz, zinput=zinput, 
                                                    zgrid=zgrid, filters=filters, **fit_zgrid_validation_kws)
            # store the redshift
            zpeak_trained[i] = zpeak
            zbest_trained[i] = zbest
            zlow_trained[i] = zlow
            zhigh_trained[i] = zhigh
            coefs_trained[i] = coefs

        # for comparison, fit again with original template
        if training and fit_initial_dicts:
            zpeak_initial = np.zeros(Ngal)
            zbest_initial = np.zeros(Ngal)
            zlow_initial= np.zeros(Ngal)
            zhigh_initial = np.zeros(Ngal)
            # print('\nTraining Catalog Untrained Redshift Estimation')
            for i in range(Ngal):
                print(f"\rValidation Catalog Untrained Redshift Estimation:\t\t{i+1}/{Ngal} sources", end="")
                # fit this spectrum and obtain the redshift
                zinput = False
                spec_obs_i = np.ascontiguousarray(spec_obs[i])
                err_obs_i = np.ascontiguousarray(err_obs[i])
                zpeak_bi,zbest_bi,zlow_bi,zhigh_bi,coefs_bi,b_bi,model_bi,_,_ = fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest, D_rest=D_rest_initial,
                                                    D_allz=D_allz_initial, zinput=zinput, zgrid=zgrid, filters=filters, **fit_zgrid_validation_kws)
                zpeak_initial[i] = zpeak_bi
                zbest_initial[i] = zbest_bi
                zlow_initial[i] = zlow_bi
                zhigh_initial[i] = zhigh_bi
        else:
            zpeak_initial = None
            zbest_initial = None
            zlow_initial= None
            zhigh_initial = None

    else:   # multiprocessing

        segment_Ndat = Ndat_validation // mp_threads
        parents_mplist = []
        process_mplist = []

        for mi in range(mp_threads):
            istart = segment_Ndat * mi
            if mi == mp_threads-1:
                Ndat_mi = segment_Ndat + mp_threads
            else:
                Ndat_mi = segment_Ndat
            cat_mi = fx.Catalog(pathfile=validation_catalog, Ndat=Ndat_mi, istart=istart, centering=centering)
            parent_mi, child_mi = mp.Pipe(duplex=False)
            parents_mplist.append(parent_mi)

            if mi == 0:
                verbose = True
            else:
                verbose = False
            p_mi = mp.Process(target=fitting_mp, args=(child_mi, cat_mi, lamb_rest, D_rest, D_allz, 
                                                        zgrid, filters, fit_zgrid_validation_kws, verbose))
            process_mplist.append(p_mi)
            p_mi.start()
        
        zpeak_mplist = []
        zbest_mplist = []
        zlow_mplist = []
        zhigh_mplist = []
        coefs_mplist = []

        for mi in range(mp_threads):
            zpeak_mi, zbest_mi, zlow_mi, zhigh_mi, coefs_mi = parents_mplist[mi].recv()
            zpeak_mplist.append(zpeak_mi)
            zbest_mplist.append(zbest_mi)
            zlow_mplist.append(zlow_mi)
            zhigh_mplist.append(zhigh_mi)
            coefs_mplist.append(coefs_mi)
            
        try:
            for mi in range(mp_threads):
                process_mplist[mi].join()
        except KeyboardInterrupt:
            for mi in range(mp_threads):
                process_mplist[mi].terminate()
            for mi in range(mp_threads):
                process_mplist[mi].join()

        zpeak_trained = np.hstack(zpeak_mplist)
        zbest_trained = np.hstack(zbest_mplist)
        zlow_trained = np.hstack(zlow_mplist)
        zhigh_trained = np.hstack(zhigh_mplist)
        coefs_trained = np.vstack(coefs_mplist) # coefs_trained is 2d array


        if training and fit_initial_dicts:
            print('')
            segment_Ndat = Ndat_validation // mp_threads
            parents_mplist = []
            process_mplist = []

            for mi in range(mp_threads):
                istart = segment_Ndat * mi
                if mi == mp_threads-1:
                    Ndat_mi = segment_Ndat + mp_threads
                else:
                    Ndat_mi = segment_Ndat
                cat_mi = fx.Catalog(pathfile=validation_catalog, Ndat=Ndat_mi, istart=istart, centering=centering)
                parent_mi, child_mi = mp.Pipe(duplex=False)
                parents_mplist.append(parent_mi)

                if mi == 0:
                    verbose = True
                else:
                    verbose = False
                p_mi = mp.Process(target=fitting_mp, args=(child_mi, cat_mi, lamb_rest, D_rest_initial, D_allz_initial, 
                                                            zgrid, filters, fit_zgrid_validation_kws, verbose))
                process_mplist.append(p_mi)
                p_mi.start()
            
            zpeak_mplist = []
            zbest_mplist = []
            zlow_mplist = []
            zhigh_mplist = []

            for mi in range(mp_threads):
                zpeak_mi, zbest_mi, zlow_mi, zhigh_mi, coefs_mi = parents_mplist[mi].recv()
                zpeak_mplist.append(zpeak_mi)
                zbest_mplist.append(zbest_mi)
                zlow_mplist.append(zlow_mi)
                zhigh_mplist.append(zhigh_mi)
                
            try:
                for mi in range(mp_threads):
                    process_mplist[mi].join()
            except KeyboardInterrupt:
                for mi in range(mp_threads):
                    process_mplist[mi].terminate()
                for mi in range(mp_threads):
                    process_mplist[mi].join()

            zpeak_initial = np.hstack(zpeak_mplist)
            zbest_initial = np.hstack(zbest_mplist)
            zlow_initial = np.hstack(zlow_mplist)
            zhigh_initial = np.hstack(zhigh_mplist)

        else:
            zpeak_initial = None
            zbest_initial = None
            zlow_initial= None
            zhigh_initial = None


    plotting_training = (training & fit_training_catalog)

    # Create zphot vs zspec plots
    makefigs.zp_zs_plots(ztrue=ztrue, z_initial=zpeak_initial, z_trained=zpeak_trained, zmin=zmin, zmax=zmax, catalog='fitting_zpeak', training=plotting_training)
    makefigs.zp_zs_plots(ztrue=ztrue, z_initial=zbest_initial, z_trained=zbest_trained, zmin=zmin, zmax=zmax, catalog='fitting_zbest', training=plotting_training)

    # Create 6 bin uncertainty fraction and zscore plots
    makefigs.uncertainty_binplots(zs=ztrue, zp=zpeak_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zpeak_initial, zl0=zlow_initial, zh0=zhigh_initial, ztype='zpeak', training=plotting_training)
    makefigs.uncertainty_binplots(zs=ztrue, zp=zbest_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zbest_initial, zl0=zlow_initial, zh0=zhigh_initial, ztype='zbest', training=plotting_training)
    # Create 6 bin zphot vs zspec hexbin plots
    makefigs.hexbin_binplot(zs=ztrue, zp=zpeak_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zpeak_initial, zl0=zlow_initial, zh0=zhigh_initial, ztype='zpeak', training=plotting_training)
    makefigs.hexbin_binplot(zs=ztrue, zp=zbest_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zbest_initial, zl0=zlow_initial, zh0=zhigh_initial, ztype='zbest', training=plotting_training)
    # save estimated redshifts
    np.savez_compressed(output_dir / 'estimated_redshifts.npz', ztrue=ztrue, zpeak=zpeak_trained, zbest=zbest_trained, zlow=zlow_trained, zhigh=zhigh_trained, 
             zpeak_initial=zpeak_initial, zbest_initial=zbest_initial, zlow_initial=zlow_initial, zhigh_initial=zhigh_initial, coefs=coefs_trained, idx_cali=idx_cali)

    # Sparsity plot
    if larslasso:
        makefigs.sparsity_report(coefs_trained=coefs_trained, max_feature=max_feature, add_constant=add_constant)

    # select galaxy examples based on redshift error bin
    idx_examples = []
    idx_validation = np.arange(Ndat_validation, dtype=int)
    idx_num_perbin = np.ones(6, dtype=int)
    sigma_1pz_ranges = [
            (0, 0.003),
            (0.003, 0.01),
            (0.01, 0.03),
            (0.03, 0.1),
            (0.1, 0.2),
            (0.2, 0.5)]
    sig_trained = (zhigh_trained-zlow_trained)/2
    z_uncertainty = sig_trained/(1+zbest_trained)
    for i , (low, high) in enumerate(sigma_1pz_ranges):
        h = (z_uncertainty<high) & (z_uncertainty>low)
        idx_in_bin = idx_validation[h]
        if len(idx_in_bin) >= idx_num_perbin[i]:
            selected_idx = np.random.choice(idx_in_bin, size=idx_num_perbin[i]).tolist()
            idx_examples.extend(selected_idx)
        else:
            idx_num_perbin[i+1] += 1

    # Create example SED plots
    makefigs.example_seds(idx=idx_examples, cat=cat, lamb_rest=lamb_rest, D_rest=D_rest, D_rest_initial=D_rest_initial, \
                          zgrid=zgrid, filters=filters, validation_fit_kws=fit_zgrid_validation_kws, ztype='zpeak')
    # makefigs.example_seds(idx=idx_examples, cat=cat, lamb_rest=lamb_rest, D_rest=D_rest, D_rest_initial=D_rest_initial, \
    #                       zgrid=zgrid, filters=filters, validation_fit_kws=fit_zgrid_validation_kws, ztype='zbest')
    
    # fit EAZY with trained dictionaries
    # makefigs.fit_eazy_plots(lamb_rest, D_rest, templates_EAZY)

    tfc = time.time()
    print('\nElapsed Time = '+str(tfc-tic)+' seconds')

    end_datetime = datetime.now().isoformat(timespec='seconds')
    output_reports = {
            'Time': {
                'Start': start_datetime,
                'End': end_datetime
            },
            'Catalogs': {
                'Training Catalog': training_catalog,
                'Validation Catalog': validation_catalog,
                'Dictionary': dict_read_from_file,
                'Ndict': Ndict,
                'Ndat_training': Ndat,
                'Ndat_validation': Ndat_validation,
                'Ncalibrators': int(Ncalibrators),
                'use_DESI_flag': use_DESI_flag
            },
            'Algorithm': {
                'Training': training,
                'Nepoch': Nepoch,
                'update_algorithm': algorithm,
                'Fix Z in training': fix_z,
                'Centering': centering,
                'remove_old_AB_info': remove_old_ab_info,
                'epochs_to_keep': epochs_to_keep,
                'scale_past_data': scale_past_data
            },
            'LARSlasso': {
                'LARSlasso': larslasso,
                'Positive': larslasso_positive,
                'Alpha training': larslasso_alpha_train,
                'Alpha sigma training': larslasso_alpha_sigma_train,
                'Alpha fitting': larslasso_alpha_fit,
                'Alpha sigma fitting': larslasso_alpha_sigma_fit,
                'Training Best Estimator': train_best_estimator,
                'Fitting Best Estimator': fit_best_estimator,
                'max_feature': max_feature,
                'active_OLS_training': active_OLS_training,
                'active_OLS_fitting': active_OLS_fitting,
                'center_Xy': center_Xy,
                'unit_X': unit_X,
                'unit_y': unit_y
            },
            'Zgrid': {
                'zmin': zmin,
                'zmax': zmax,
                'scale_1plusz': scale_1plusz,
                'local_finegrid': config.local_finegrid,
            }

    }

    with open(output_dir / parameters_report, 'w') as output_file:
        yaml.safe_dump(output_reports, output_file, sort_keys=False)


