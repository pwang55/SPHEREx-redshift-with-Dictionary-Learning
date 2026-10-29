'''


Usage:
    $ python dictionary_learn_from_config.py config.yaml

Make sure "dictionary_learn_libs.py" is in the same directory as this code.
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
import libs.lib_dictlearn as libs
import libs.diagnostic_plots as dplots
import yaml
from datetime import datetime
import multiprocessing as mp
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        # default=None,
        metavar='<str>',
        help='Configuration YAML file',
        required=True,
    )
    parser.add_argument(
        '-d',
        '--dictionary',
        type=str,
        default=None,
        metavar='<str>',
        help='Input dictionary file',
    )
    parser.add_argument(
        '-tc',
        '--training_cat',
        type=str,
        default=None,
        metavar='<str>',
        help='Training catalog',
    )
    parser.add_argument(
        '-at',
        '--alpha_train',
        type=float,
        default=None,
        metavar='<float>',
        help='Training alpha sparsity parameter',
    )
    parser.add_argument(
        '-ast',
        '--alpha_sig_train',
        type=float,
        default=None,
        metavar='<float>',
        help='Training alpha sparsity parameter in source sigma unit',
    )
    parser.add_argument(
        '-nd',
        '--ndicts',
        type=int,
        default=None,
        metavar='<int>',
        help='Number of dictionaries',
    )
    parser.add_argument(
        '-fd',
        '--fix_dicts',
        type=int,
        default=None,
        metavar='<int>',
        help='Number of dictionaries to be fixed (from the start of list)',
    )
    parser.add_argument(
        '-myt',
        '--median_ynorm_train',
        type=int,
        default=None,
        metavar='<int>',
        help='use median filter to calculate corrected ynorm (for training)',
    )
    parser.add_argument(
        '-myf',
        '--median_ynorm_fit',
        type=int,
        default=None,
        metavar='<int>',
        help='use median filter to calculate corrected ynorm (for training)',
    )
    parser.add_argument(
        '-do',
        '--dicts_out',
        type=str,
        default=None,
        metavar='<str>',
        help='dictionaries output filename',
    )
    parser.add_argument(
        '-dso',
        '--dicts_sys_out',
        type=str,
        default=None,
        metavar='<str>',
        help='obs-frame dictionaries output filename',
    )
    parser.add_argument(
        '-od',
        '--output_dir',
        type=str,
        default=None,
        metavar='<str>',
        help='output files directory',
    )

    return parser.parse_args()

# if __name__ == "__main__":
def main():

    start_datetime = datetime.now().isoformat(timespec='seconds')

    args = parse_args()
    config_file = args.config

    # if len(sys.argv) == 1:
        # print(__doc__)
        # exit()

    # Read input config file with Configs class
    # config_file = sys.argv[1]
    config = libs.Configs(config=config_file)

    # Catalog configurations
    training_catalog = config.training_catalog
    validation_catalog = config.validation_catalog
    filter_central_wavelengths = config.filter_central_wavelengths
    Ndat = config.Ndat
    Ndat_validation = config.Ndat_validation
    Ncalibrators = config.Ncalibrators
    use_DESI_flag = config.use_DESI_flag
    calibrator_ynorm = config.calibrator_ynorm
    f_lambda_mode = config.f_lambda_mode
    # Dictionary input configurations
    dict_read_from_file = config.dict_read_from_file
    # dict_sys_from_file = config.dict_sys_from_file
    # add_constant = config.add_constant
    lamb_rest_step = config.lamb_rest_step
    Ndict_sys = config.Ndict_sys
    fix_dicts_sys = config.fix_dicts_sys
    fix_dicts = config.fix_dicts
    Ndict = config.Ndict
    num_EAZY_as_dict = config.num_EAZY_as_dict
    # dicts_fluctuation_scaling_const = config.dicts_fluctuation_scaling_const
    # dict_fluctuation_scaling_base = config.dict_fluctuation_scaling_base
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
    correct_nodata_offset = config.correct_nodata_offset
    separate_training_weights = config.separate_training_weights
    lamb_obs_sep = 3.79
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
    active_alpha_0_training = config.active_alpha_0_training
    active_alpha_0_fitting = config.active_alpha_0_fitting
    active_alpha_0_fitting_penalty = config.active_alpha_0_fitting_penalty
    active_OLS_training = config.active_OLS_training
    active_OLS_fitting = config.active_OLS_fitting
    center_Xy = config.center_Xy
    unit_X = config.unit_X
    unit_y = config.unit_y
    larslasso_alpha_scaling = config.larslasso_alpha_scaling
    # Fitting configurations
    fitting = config.fitting

    marginalization = config.marginalization
    marginalization_min_alpha = config.marginalization_min_alpha
    marginalization_max_alpha = config.marginalization_max_alpha
    if marginalization_min_alpha is None:
        marginalization_min_alpha = 0.0
    if marginalization_max_alpha is None:
        marginalization_max_alpha = np.inf
    
    marginalize_active_only = config.marginalize_active_only
    partial_marginalization_min_alpha = config.partial_marginalization_min_alpha
    partial_marginalization_max_alpha = config.partial_marginalization_max_alpha
    if partial_marginalization_min_alpha is None:
        partial_marginalization_min_alpha = 0.0
    if partial_marginalization_max_alpha is None:
        partial_marginalization_max_alpha = np.inf

    probline = config.probline
    fit_training_catalog = config.fit_training_catalog
    fit_initial_dicts = config.fit_initial_dicts
    median_ynorm_train = config.median_ynorm_train
    median_ynorm_fit = config.median_ynorm_fit
    median_spectra_train = config.median_spectra_train
    median_spectra_fit = config.median_spectra_fit
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
    marginalize_zgrid_validation_kws = config.marginalize_zgrid_validation_kws

    if args.alpha_train is not None:
        larslasso_alpha_train = args.alpha_train
        fit_zgrid_training_kws['alpha'] = larslasso_alpha_train

    if args.alpha_sig_train is not None:
        larslasso_alpha_sigma_train = args.alpha_sig_train
        fit_zgrid_training_kws['alpha_sig'] = larslasso_alpha_sigma_train

    if args.ndicts is not None:
        Ndict = args.ndicts
    if args.dictionary is not None:
        dict_read_from_file = args.dictionary
        if args.dictionary == 'None' or args.dictionary == 'none':
            dict_read_from_file = None
    if args.training_cat is not None:
        training_catalog = args.training_cat
    if args.fix_dicts is not None:
        fix_dicts = args.fix_dicts
    if args.dicts_out is not None:
        dicts_out_filename = args.dicts_out
    else:
        dicts_out_filename = 'trained_templates.npz'
    if args.dicts_sys_out is not None:
        dicts_sys_out_filename = args.dicts_sys_out
    else:
        dicts_sys_out_filename = 'trained_templates_sys.npz'
    if args.output_dir is not None:
        output_dirname = args.output_dir
    if args.median_ynorm_train is not None:
        median_ynorm_train = args.median_ynorm_train
        fit_zgrid_training_kws['median_ynorm'] = median_ynorm_train

    if args.median_ynorm_fit is not None:
        median_ynorm_fit = args.median_ynorm_fit
        fit_zgrid_validation_kws['median_ynorm'] = median_ynorm_fit
        marginalize_zgrid_validation_kws['median_ynorm'] = median_ynorm_fit


    # fit_zgrid = libs.fit_zgrid

    # Generate Zgrid based on configuration
    zgrid = libs.generate_zgrid(zmin=zmin, zmax=zmax, dz=dz, scale_1plusz=scale_1plusz, testing_zgrid=testing_zgrid)

    # Create catalog object and save items as variables
    cat = libs.Catalog(pathfile=training_catalog, Ndat=Ndat, centering=centering)
    if filter_central_wavelengths is None:
        lamb_obs = cat.lamb_obs
    else:
        if filter_central_wavelengths.split('.')[-1] == 'txt':
            # central_wavelengths_tab = np.loadtxt(filter_central_wavelengths, dtype={'names': ('filter','wavelengths'), 'formats': ('U25','f8')})
            # lamb_obs = central_wavelengths_tab['wavelengths']
            lamb_obs = np.genfromtxt(filter_central_wavelengths)[:,1]
        elif filter_central_wavelengths.split('.')[-1] == 'npz':
            central_wavelengths_tab = np.load(filter_central_wavelengths)
            lamb_obs = central_wavelengths_tab['wavelengths']

    Nfilt = len(lamb_obs)   # number of datapoints, even if not convolving with filters
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
        h_cali = cat.snr_norm > calibrator_ynorm
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
        # filters = libs.read_filters(filter_list)
        filters = libs.read_filters(filter_list, return_unique_inverse=True)
    else:
        filters = None

    # if training is False, and if Ndict > total number of dictionaries in file, set input Ndict to None 
    # so that libs.Dictionary doesn't add noise templates
    if training:
        dicts = libs.Dictionary(
            dictfile=dict_read_from_file,
            lamb_sys_size=len(lamb_obs),
            lamb_rest_step=lamb_rest_step,
            Ndict=Ndict,
            Ndict_sys=Ndict_sys,
            alpha_train=larslasso_alpha_train)
    else:
        dicts = libs.Dictionary(
            dictfile=dict_read_from_file,
            lamb_sys_size=len(lamb_obs),
            lamb_rest_step=lamb_rest_step,
            Ndict=None,
            Ndict_sys=None)
        
    lamb_rest = dicts.lamb_rest
    D_rest = dicts.D_rest
    D_rest_initial = dicts.D_rest_initial
    D_sys = dicts.D_sys
    D_sys_initial = dicts.D_sys_initial
    D_rest_alphas = dicts.D_rest_alphas
    D_rest_alphas_ndict = dicts.D_rest_alphas_ndict
    D_sys_alphas = dicts.D_sys_alphas
    D_sys_alphas_ndict = dicts.D_sys_alphas_ndict
    # print('tttttt', Ndict_sys, dicts.D_rest_alphas_ndict, dicts.D_sys_alphas_ndict)
    # # lamb_rest_idx = np.arange(len(lamb_rest))
    # # Read EAZY templates for dictionary initialization and also evaluation test if they are given
    # if eazy_templates_location is not None and eazy_templates_dir.is_dir():
    #     templates_EAZY = libs.load_EAZY(lamb_rest, eazy_templates_location)
    #     if not f_lambda_mode:
    #         templates_EAZY = libs.flambda2fnu(lamb_rest*10000, templates_EAZY)
    #     templates_EAZY = np.array(templates_EAZY)
    #     if centering:
    #         templates_EAZY = templates_EAZY - np.mean(templates_EAZY, axis=1)[:,None]

    # # initialize dictionaries if they are not given
    # if not dict_read_from_file:
    #     # If initializing dictionaries, create them as noise with different level of fluctuation
    #     # dictionary_fluctuation_scaling = np.array([dicts_fluctuation_scaling_const/(dict_fluctuation_scaling_base**i) for i in range(Ndict-num_EAZY_as_dict)])
    #     D_rest = libs.initialize_dicts(Ndict, 
    #                                  templates_EAZY=templates_EAZY, 
    #                                  num_EAZY_as_dict=num_EAZY_as_dict, 
    #                                  lamb_rest=lamb_rest, 
    #                                  add_constant=add_constant)

    tic = time.time()

    # Grab the Ngal for validation catalog just for output information
    if validation_catalog.split('.')[-1] == 'npz':
        with np.load(validation_catalog) as fitting_dat:
            Ndat_validation = min(len(fitting_dat['z']), Ndat_validation)
    elif validation_catalog.split('.')[-1] == 'parq' or validation_catalog.split('.')[-1] == 'parquet':
        fitting_dat = pd.read_parquet(validation_catalog)
        Ndat_validation = min(len(fitting_dat), Ndat_validation)
        del fitting_dat

    if epochs_to_keep >= Nepoch:
        remove_old_ab_info = False


    # print(f"Algorithm = {algorithm}")
    if training:
        print(f"Training Catalog:\t{training_catalog} ({Ngal} sources)")
    if fitting:
        print(f"Validation Catalog:\t{validation_catalog} ({Ndat_validation} sources)")
    if dict_read_from_file:
        print(f"Dictionaries:\t\t{dict_read_from_file}")
    else:
        print(f"{num_EAZY_as_dict} of 7 EAZY templates used as initialized dictionaries")
    # if dict_sys_from_file:
    #     print(f"obs-frame Dictionaries:\t\t{dict_sys_from_file}")
    print('')
    print(f"Ndict = {Ndict}")
    # print(f"Add constant: {add_constant}")
    if training and (fix_dicts > 0):
        print(f"Fixed dictionaries: {fix_dicts} (from the start of list)")
    if training and (Ndict_sys>0):
        print(f"Rest frame systematic dictionaries: {Ndict_sys} (from the start of list)")
    print(f"Centering: {centering}")    
    if training:
        print('')
        print(f"Fix z in training: {fix_z}")
        print(f"Scale past data: {scale_past_data}")
        if not remove_old_ab_info:
            print(f"Remove old AB info: {remove_old_ab_info}, Scale past data: {scale_past_data}")
        else:
            print(f"Remove old AB info: {remove_old_ab_info} (keeps {epochs_to_keep} Epochs)")
        print(f'Correct nodata offset: {correct_nodata_offset}')
        if separate_training_weights:
            print(f"Separate training weights: Bands 1-4 x{weights1}, Bands 5-6 x{weights2}")
        if use_DESI_flag:
            print(f"Ncalibrators = {Ncalibrators} (DESI Flags)")
        elif Ncalibrators>0:
            print(f"Ncalibrators = {Ncalibrators} (L2-norm > {calibrator_ynorm})")
        if algorithm == 0:
            print(f"Learning rates = {learning_rate0}/{learning_rate_cali}")
        else:
            if larslasso:
                # print(fr"LARSlasso: {larslasso} (alpha={lars_alpha} + {LARSlasso_alpha_sigma} sigma, Positive: {lars_positive}, max_feature={max_feature}, center_Xy={center_Xy})")
                print(f"LARS-lasso: {larslasso}")
                print(f"\talpha (training) = {larslasso_alpha_train} + {larslasso_alpha_sigma_train} sigma")
                if fitting:
                    print(f"\talpha (fitting) = {larslasso_alpha_fit} + {larslasso_alpha_sigma_fit} sigma")
                print(f"\tPositive = {larslasso_positive}")
                print(f"\tactive alpha zero = {active_alpha_0_training}/{active_alpha_0_fitting} (Fitting penalty: {active_alpha_0_fitting_penalty & active_alpha_0_fitting})")
                print(f"\tcenter_Xy = {center_Xy}")
                print(f"\tunit_X = {unit_X}")
                print(f"\tunit_y = {unit_y}")
                if max_feature:
                    print(f"\tmax_feature = {max_feature}")
                if train_best_estimator is not None:
                    print(f"\tTraining Best Estimator = {train_best_estimator}")
                if fit_best_estimator is not None:
                    print(f"\tFitting Best Estimator = {fit_best_estimator}")
                if median_ynorm_train is not None:
                    print(f"median_ynorm_train = {median_ynorm_train}")
            else:
                print(f"LARS-lasso: {larslasso}")
        if testing_zgrid:
            print('Testing Zgrid: On')
    else:
        if fitting:
            if marginalization:
                print(f"\tmarginalization = True (alpha={marginalization_min_alpha}~{marginalization_max_alpha})")
            else:
                print("marginalization = False")
            if larslasso:    
                print(f"\talpha (fitting) = {larslasso_alpha_fit} + {larslasso_alpha_sigma_fit} sigma")
                print(f"\tactive alpha zero = {active_alpha_0_fitting}")
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
            if median_ynorm_fit is not None:
                print(f"median_ynorm_fit = {median_ynorm_fit}")


    if convolve_filter or last_stage_convolve_filter or fitting_convolve_filter:
        print(f"Convolving filters: 1st stage:{convolve_filter}, 2nd stage:{last_stage_convolve_filter}, fitting:{fitting_convolve_filter}")

    if multiprocess and fitting:
        print(f'Validation catalog fitting multiprocessing threads: {mp_threads}')
    # TODO option to warm restart with A and B from dictionary file
    if training:
        print('')
        if Ndict > 0:
            D_rest_previous = D_rest.copy()
        elif Ndict == 0:
            D_rest_previous = None
        if Ndict_sys is None:
            Ndict_sys = 0
        D_rest_change_rates = np.zeros((Ndict + Ndict_sys, Nepoch))
        if Ndict_sys > 0:
            D_sys_previous = D_sys.copy()

        # Initialize A and B matrices for dictionary updates
        if not separate_training_weights:
            A = np.zeros((Ndict, Ndict))
            B = np.zeros((len(lamb_rest), Ndict))
            if Ndict_sys > 0:
                Asys = np.zeros((Ndict_sys, Ndict_sys))
                Bsys = np.zeros((len(lamb_obs), Ndict_sys))
        else:
            A1 = np.zeros((Ndict, Ndict))
            A2 = np.zeros((Ndict, Ndict))
            B1 = np.zeros((len(lamb_rest), Ndict))
            B2 = np.zeros((len(lamb_rest), Ndict))

        if remove_old_ab_info: # save previous coefficients
            # A_history = np.zeros((Ngal, D_rest.shape[0], D_rest.shape[0]))
            # B_history = np.zeros((Ngal, len(lamb_rest), D_rest.shape[0]))
            coefs_history = np.zeros((Ngal, epochs_to_keep, Ndict+Ndict_sys))
            # if Ndict_sys > 0:
                # coefs_sys_history = np.zeros((Ngal, epochs_to_keep, Ndict_sys))

            if not separate_training_weights:
                spec_rest_history = np.zeros((Ngal, epochs_to_keep, lamb_rest.shape[0]))
                if Ndict_sys > 0:
                    spec_sys_history = np.zeros((Ngal, epochs_to_keep, lamb_obs.shape[0]))
            else:
                spec_rest1_history = np.zeros((Ngal, epochs_to_keep, lamb_rest.shape[0]))
                spec_rest2_history = np.zeros((Ngal, epochs_to_keep, lamb_rest.shape[0]))
            # if scale_past_data:
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

                # fit this spectrum and obtain the redshift
                spec_obs_i = np.ascontiguousarray(spec_obs[i_gal])
                err_obs_i = np.ascontiguousarray(err_obs[i_gal])
                fit_out = libs.fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest=lamb_rest, D_rest=D_rest, D_sys=D_sys, zinput=zinput, 
                                                                        zgrid=zgrid, filters=filters, **fit_zgrid_training_kws)
                # z, _, zlow, zhigh, coefs, b, model, _, _, _ = fit_out
                z = fit_out['zpeak']
                coefs = fit_out['coef']
                b = fit_out['b']
                model = fit_out['model']

                # update the spectral dictionary using the residuals between the model and data
                residual = spec_obs[i_gal] - model

                if Ndict_sys > 0:
                    # D_thiszi = libs.apply_redshift(D_rest, z=z, lamb_in=lamb_rest, lamb_out=lamb_obs, filters=filters, conv=convolve_filter)
                    model_sys = D_sys.T @ coefs[:Ndict_sys]
                    # model_intrinsic = D_thiszi.T @ coefs[Ndict_sys:]
                    spec_obs_i_intrinsic = spec_obs[i_gal] - model_sys
                    # spec_obs_i_sys = spec_obs[i_gal] - model_intrinsic
                else:
                    spec_obs_i_intrinsic = spec_obs[i_gal]

                if not separate_training_weights:
                    # print(z)
                    j_update = np.where((lamb_rest >= np.min(lamb_obs)/(1+z)) & (lamb_rest <= np.max(lamb_obs)/(1+z)))[0]     # Find overlap between this spectra in rest-frame and dictionary
                    # print(j_update)
                    j_longerside = np.where(lamb_rest > np.max(lamb_obs)/(1+z))[0]    # index for lamb_rest that has wavelength longer than overlap regions
                    j_shorterside = np.where(lamb_rest < np.min(lamb_obs)/(1+z))[0]
                else:
                    j_update1 = np.where((lamb_rest >= np.min(lamb_obs)/(1+z)) & (lamb_rest <= lamb_obs_sep/(1+z)))[0]     # rest frame wavelength of detector 1~4
                    j_update2 = np.where((lamb_rest >= lamb_obs_sep/(1+z)) & (lamb_rest <= np.max(lamb_obs)/(1+z)))[0]     # detector 5~6
                    j_longerside1 = np.where(lamb_rest > lamb_obs_sep/(1+z))[0]    # index for lamb_rest that has wavelength longer than overlap regions
                    j_shorterside1 = np.where((lamb_rest < np.min(lamb_obs)/(1+z)))[0]
                    j_longerside2 = np.where(lamb_rest > np.max(lamb_obs)/(1+z))[0]    # index for lamb_rest that has wavelength longer than overlap regions
                    j_shorterside2 = np.where((lamb_rest < lamb_obs_sep/(1+z)))[0]
                # j_update_outside = np.setdiff1d(np.arange(len(lamb_rest)), j_update)

                if not separate_training_weights:
                    interpolated_residual = np.interp(lamb_rest[j_update], lamb_obs/(1+z), residual)    # interpolate the residual to these values
                    interpolated_spec_obs = np.interp(lamb_rest[j_update], lamb_obs/(1+z), spec_obs_i_intrinsic)
                else:
                    interpolated_residual1 = np.interp(lamb_rest[j_update1], lamb_obs/(1+z), residual)    # interpolate the residual to these values
                    interpolated_residual2 = np.interp(lamb_rest[j_update2], lamb_obs/(1+z), residual)    # interpolate the residual to these values
                    interpolated_spec_obs1 = np.interp(lamb_rest[j_update1], lamb_obs/(1+z), spec_obs_i_intrinsic)
                    interpolated_spec_obs2 = np.interp(lamb_rest[j_update2], lamb_obs/(1+z), spec_obs_i_intrinsic)

                # construct the model from best fit coefficients and dictionaries
                if Ndict > 0:
                    error_rest = np.interp(lamb_rest, lamb_obs/(1+z), err_obs[i_gal])
                    model_rest = D_rest.T @ coefs[Ndict_sys:] + b * error_rest
                    if not separate_training_weights:
                        spec_rest = model_rest.copy()
                        spec_rest[j_update] = interpolated_spec_obs    # replace the overlapped part with interpolated observed spectra
                        if correct_nodata_offset:
                            spec_rest[j_longerside] += interpolated_residual[-1] # add the offset at the end of update region to longer wavelength side
                            spec_rest[j_shorterside] += interpolated_residual[0]
                    else:
                        spec_rest1 = model_rest.copy()
                        spec_rest2 = model_rest.copy()
                        spec_rest1[j_update1] = interpolated_spec_obs1    # replace the overlapped part with interpolated observed spectra
                        spec_rest2[j_update2] = interpolated_spec_obs2    # replace the overlapped part with interpolated observed spectra
                        if correct_nodata_offset:
                            spec_rest1[j_longerside1] += interpolated_residual1[-1] # add the offset at the end of update region to longer wavelength side
                            spec_rest1[j_shorterside1] += interpolated_residual1[0]
                            spec_rest2[j_longerside2] += interpolated_residual2[-1] # add the offset at the end of update region to longer wavelength side
                            spec_rest2[j_shorterside2] += interpolated_residual2[0]
                    # this will be considered the observed spectra for update purpose; outside overlap range just use model (no update)


                # # update each item in the dictionary (do not modify the DC offset term at the end)
                # if algorithm == 0:  # pseudo-inverse vector method
                #     # set the learning rate
                #     learning_rate = learning_rate0
                #     # if this is a calibrator galaxy
                #     if i_gal in idx_cali:
                #         # use a higher learning rate since we know the redshift is correct
                #         learning_rate = learning_rate_cali
                #     for i in range(Ndict+Ndict_sys-add_constant-fix_dicts):
                #         update_factor = learning_rate * (coefs[i]/(np.sum(coefs**2)))
                #         D_rest[i,j_update] = D_rest[i,j_update] + update_factor * interpolated_residual

                # elif algorithm == 1:    # block-coordinate descent

                # D_rest update block (updates A and B)
                if (Ndict - fix_dicts) > 0:
                    if remove_old_ab_info:

                        # Use first coefficients to calculate A and B to remove
                        coef_old = coefs_history[i_gal, 0].copy()
                        if not separate_training_weights:
                            spec_rest_old = spec_rest_history[i_gal, 0].copy()
                            A_old = coef_old[Ndict_sys:,None] @ coef_old[Ndict_sys:,None].T
                            B_old = spec_rest_old[:,None] @ coef_old[Ndict_sys:,None].T
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
                                spec_rest_history[i_gal, -1] = spec_rest.copy()

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
                        A = A * AB_scale_factor + coefs[Ndict_sys:,None] @ coefs[Ndict_sys:,None].T
                        B = B * AB_scale_factor + spec_rest[:,None] @ coefs[Ndict_sys:,None].T

                    else:
                        A1 = A1 * AB_scale_factor + coefs[:,None] @ coefs[:,None].T
                        A2 = A2 * AB_scale_factor + coefs[:,None] @ coefs[:,None].T
                        A = A1 * weights1 + A2 * weights2
                        B1 = B1 * AB_scale_factor + spec_rest1[:,None] @ coefs[:,None].T
                        B2 = B2 * AB_scale_factor+ spec_rest2[:,None] @ coefs[:,None].T
                        B = B1 * weights1 + B2 * weights2


                # iterative over A and B matrices until converge or max_AB_loop, only for Ndicts
                # for j in range(Ndict-add_constant-fix_dicts):
                # for j in range(Ndict-fix_dicts):
                if (Ndict - fix_dicts) > 0:
                    for j in range(fix_dicts, Ndict):
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

                # D_sys update block
                if (Ndict_sys - fix_dicts_sys) > 0:
                    if Ndict > 0:
                        D_thiszi = libs.apply_redshift(D_rest, z=z, lamb_in=lamb_rest, lamb_out=lamb_obs, filters=filters, conv=convolve_filter)
                        model_intrinsic = D_thiszi.T @ coefs[Ndict_sys:]
                        spec_obs_i_sys = spec_obs[i_gal] - model_intrinsic
                    elif Ndict == 0:
                        spec_obs_i_sys = spec_obs[i_gal]
                        coef_old = coefs_history[i_gal, 0].copy()

                    if remove_old_ab_info:
                        if not separate_training_weights:
                            spec_sys_old = spec_sys_history[i_gal, 0].copy()
                            Asys_old = coef_old[:Ndict_sys,None] @ coef_old[:Ndict_sys,None].T
                            Bsys_old = spec_sys_old[:,None] @ coef_old[:Ndict_sys,None].T
                            Asys -= Asys_old * old_AB_scale_factor
                            Bsys -= Bsys_old * old_AB_scale_factor
                        else:
                            pass# TEMP

                        # move igal epochs ahead by 1
                        if epochs_to_keep > 1:
                            if not separate_training_weights:
                                spec_sys_history[i_gal, 0:-1] = spec_sys_history[i_gal, 1:]
                                spec_sys_history[i_gal, -1] = spec_obs_i_sys
                            else:
                                pass    # TEMP

                        elif epochs_to_keep == 1:
                            # coefs_history[i_gal, 0] = coefs.copy()
                            if not separate_training_weights:
                                spec_sys_history[i_gal, 0] = spec_obs_i_sys
                            else:
                                pass    # TEMP

                    if not separate_training_weights:
                        Asys = Asys * AB_scale_factor + coefs[:Ndict_sys,None] @ coefs[:Ndict_sys,None].T
                        Bsys = Bsys * AB_scale_factor + spec_obs_i_sys[:,None] @ coefs[:Ndict_sys,None].T
                    else:
                        pass    # TEMP

                    # iterate over Asys and Bsys, only for Ndict_sys
                    # for j in range(Ndict_sys-fix_dicts_sys):
                    for j in range(fix_dicts_sys, Ndict_sys):
                        score = 1.0
                        # Dj_old = D_rest[j].copy()
                        abi = 0
                        while score >= AB_update_tolerance and abi < max_AB_loops:
                            diag_Aj = np.diagonal(Asys)[j]
                            if diag_Aj != 0:
                                Dj_old = D_sys[j].copy()
                                uj = 1/diag_Aj * (Bsys[:,j] - (D_sys.T @ Asys[j])) + D_sys[j]
                                uj_norm = np.linalg.norm(uj)
                                D_sys[j] = uj/max(1, uj_norm)
                                score = np.linalg.norm(D_sys[j]-Dj_old)/np.linalg.norm(Dj_old)
                                abi += 1
                            else:
                                break

            # record percentage change of all dictionaries
            if Ndict > 0:
                D_rest_change_rates[:Ndict, i_epoch] = np.linalg.norm(D_rest-D_rest_previous, axis=1)/np.linalg.norm(D_rest_previous, axis=1)
                D_rest_previous = D_rest.copy()
            if Ndict_sys > 0:
                D_rest_change_rates[Ndict:Ndict+Ndict_sys, i_epoch] = np.linalg.norm(D_sys-D_sys_previous, axis=1)/np.linalg.norm(D_sys_previous, axis=1)
                D_sys_previous = D_sys.copy()

        # record new D_rest and D_sys to dicts object
        dicts.D_rest = D_rest
        dicts.D_sys = D_sys
        # D_allz = dicts.make_D_allz(lamb_obs=lamb_obs, zgrid=zgrid, filters=filters, conv=fitting_convolve_filter)
        # D_zgrid = dicts.make_D_zgrid(zgrid=zgrid, filters=filters, conv=fitting_convolve_filter)

        fig_converge, ax_converge = plt.subplots(figsize=(10, 6))
        ax_converge.plot(np.arange(Nepoch), 100*D_rest_change_rates.T, '-', linewidth=1, alpha=0.5)
        ax_converge.grid(alpha=0.4)
        ax_converge.set_title('Dictionaries Convergence')
        ax_converge.set_xlabel('Epoch')
        ax_converge.set_ylabel('Percent change rate')
        ax_converge.set_ylim(-1, 8)
        plt.savefig(plots_dir / 'dictionaries_convergence.png', dpi=300)

        # After training is done, fit all training data with training settings to get mean chi2 and mean cost
        coef_trained1 = np.zeros((Ngal, Ndict))
        coef_trained2 = np.zeros((Ngal, Ndict))
        chi2_trained1 = np.zeros(Ngal) + np.inf
        chi2_trained2 = np.zeros(Ngal) + np.inf
        cost_trained1 = np.zeros(Ngal) + np.inf
        cost_trained2 = np.zeros(Ngal) + np.inf
        k1 = np.zeros(Ngal, dtype=np.int64)
        k2 = np.zeros(Ngal, dtype=np.int64)
        aic1 = np.zeros(Ngal)
        aic2 = np.zeros(Ngal)
        bic1 = np.zeros(Ngal)
        bic2 = np.zeros(Ngal)

        print('')
        for i in range(Ngal):
            # update with number of galaxies processed
            # print(f"\r\t{i+1}/{Ngal} spectra", end="")
            print(f"\rTraining Catalog re-fit (fix z=ztrue): {i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            zinput = ztrue[i]
            fit_out1 = libs.fit_zgrid(
                lamb_obs, 
                spec_obs[i], 
                err_obs[i], 
                dictionary=dicts,
                zinput=zinput, 
                filters=filters, 
                **fit_zgrid_training_kws)
            fit_zgrid_mse_kws = fit_zgrid_validation_kws.copy()
            fit_zgrid_mse_kws['error'] = False
            fit_out2 = libs.fit_zgrid(
                lamb_obs, 
                spec_obs[i], 
                err_obs[i], 
                dictionary=dicts,
                zinput=zinput, 
                filters=filters, 
                **fit_zgrid_mse_kws)
            # _, _, _, _, _, _, model, _, cost_zgrid_trained, _ = fit_out
            model1 = fit_out1['model']
            model2 = fit_out2['model']
            coef1 = fit_out1['coef']
            coef2 = fit_out2['coef']
            cost_zgrid_trained1 = fit_out1['cost_zgrid']
            cost_zgrid_trained2 = fit_out2['cost_zgrid']
            coef_trained1[i] = coef1[:Ndict]
            coef_trained2[i] = coef2[:Ndict]
            chi2_trained1[i] = np.sum(((spec_obs[i]-model1)/err_obs[i])**2)
            chi2_trained2[i] = np.sum(((spec_obs[i]-model2)/err_obs[i])**2)
            cost_trained1[i] = cost_zgrid_trained1[0]
            cost_trained2[i] = cost_zgrid_trained2[0]
            k1[i] = len(fit_out1['alpha_path'])-1
            k2[i] = len(fit_out2['alpha_path'])-1
            aic1[i] = chi2_trained1[i] + 2*k1[i]
            aic2[i] = chi2_trained2[i] + 2*k2[i]
            bic1[i] = chi2_trained1[i] + k1[i] * np.log(len(lamb_obs))
            bic2[i] = chi2_trained2[i] + k2[i] * np.log(len(lamb_obs))

        # mean_chi21 = np.nanmean(chi2_trained1)
        # # std_chi21 = np.nanstd(chi2_trained1)
        # # percentile_chi21 = np.nanpercentile(chi2_trained1, q=[16, 50, 84])
        # mean_cost1 = np.nanmean(cost_trained1)
        # # std_cost1 = np.nanstd(cost_trained1)
        # # percentile_cost1 = np.nanpercentile(cost_trained1, q=[16, 50, 84])
        # aic1_tot = np.sum(aic1)
        # bic1_tot = np.sum(bic1)

        # mean_chi22 = np.nanmean(chi2_trained2)
        # # std_chi22 = np.nanstd(chi2_trained2)
        # # percentile_chi22 = np.nanpercentile(chi2_trained2, q=[16, 50, 84])
        # mean_cost2 = np.nanmean(cost_trained2)
        # # std_cost2 = np.nanstd(cost_trained2)
        # # percentile_cost2 = np.nanpercentile(cost_trained2, q=[16, 50, 84])
        # aic2_tot = np.sum(aic2)
        # bic2_tot = np.sum(bic2)

        bins = 100
        if Ndict > 0:
            nrows_dict_z = int(np.ceil(np.sqrt(Ndict)))
            ncols_dict_z = int(np.ceil(Ndict / nrows_dict_z))
            fig_dict_z, ax_dict_z = plt.subplots(nrows_dict_z, ncols_dict_z, figsize=(10, 8))
            for i in range(nrows_dict_z*ncols_dict_z):
                row = i % nrows_dict_z
                col = i // nrows_dict_z
                if Ndict == 1:
                    ax = ax_dict_z
                elif Ndict == 2:
                    ax = ax_dict_z[i]
                else:
                    ax = ax_dict_z[row, col]
                if i < Ndict:
                    dicti_non_zeros_mask1 = coef_trained1[:, i] > 0
                    dicti_non_zeros_mask2 = coef_trained2[:, i] > 0
                    ztrue_i1 = ztrue[dicti_non_zeros_mask1]
                    ztrue_i2 = ztrue[dicti_non_zeros_mask2]
                    ax.hist(ztrue_i1, range=(zmin, zmax), bins=bins, alpha=0.6)
                    ax.hist(ztrue_i2, range=(zmin, zmax), bins=bins, alpha=0.6)
                    ax.grid(alpha=0.2)
                    ax.set_title(f"Dict {i}")
                else:
                    ax.set_visible(False)
            fig_dict_z.tight_layout()
            plt.savefig(plots_dir / 'dicts_zdistribution_training.png', dpi=300)

        # TODO change to saving hdf5 file
        if Ndict > 0:
            np.savez_compressed(output_dir / dicts_out_filename,
                                lamb_rest=lamb_rest, 
                                D_rest=D_rest, 
                                A=A, 
                                B=B, 
                                # D_sys=D_sys,
                                # Asys=Asys,
                                # Bsys=Bsys,
                                D_rest_change_rates=D_rest_change_rates
                                )

        if Ndict_sys > 0:
            np.savez_compressed(output_dir / dicts_sys_out_filename, 
                                lamb_rest=lamb_obs, 
                                D_sys=D_sys, 
                                A_sys=Asys, 
                                B_sys=Bsys)
    
        # fit training data with fix redshift and training setting to make sparsity plot and chi2 plot
        ynorms = np.linalg.norm(spec_obs/err_obs, axis=1)
        chi2_training = np.zeros(Ngal)
        last_alpha_training = np.zeros(Ngal)
        coefs_training = np.zeros((Ngal, Ndict+Ndict_sys))
        for i in range(Ngal):
            fit_out_training = libs.fit_zgrid(
                    lamb_obs, 
                    spec_obs[i], 
                    err_obs[i], 
                    dictionary=dicts, 
                    zinput=ztrue[i], 
                    zgrid=zgrid, 
                    filters=filters, 
                    fix_at_zinput=True, 
                    **fit_zgrid_training_kws)                  
            coef_training = fit_out_training['coef']
            model_training = fit_out_training['model']
            alpha_path_training = fit_out_training['alpha_path']
            chi2_training[i] = np.sum(((spec_obs[i]-model_training)/err_obs[i])**2)
            coefs_training[i] = coef_training
            if alpha_path_training.shape[0] == 1:
                last_alpha_training[i] = alpha_path_training[0]
            else:
                last_alpha_training[i] = alpha_path_training[-2]

        # dplots.chi2_plots(ynorms=ynorms, 
        #                   reduced_chi2s_fixz=chi2_training/Nfilt, 
        #                   last_alpha_fixz=last_alpha_training, 
        #                   catalog='training_setting',
        #                   savefig=True,
        #                   output_dirname=plots_dir)
        # Sparsity plot
        if larslasso:
            dplots.sparsity_report(coefs_trained=coefs_training, 
                                    ynorm=ynorms, 
                                    max_feature=max_feature, 
                                    # add_constant=add_constant, 
                                    catalog='_training_result',
                                    savefig=True,
                                    output_dirname=plots_dir)


        # del A, B, coefs_history, spec_rest_history
        if not separate_training_weights:
            del A, B, coefs_history, spec_rest_history
        else:
            del A1, B1, A2, B2, coefs_history, spec_rest1_history, spec_rest2_history

    # Make template plots and multiplots
    # makefigs = dplots.diagnostic_plots(output_dirname=plots_dir)
    dplots.template_plots(lamb_rest=lamb_rest, 
                          D_rest=D_rest, 
                          D_rest_initial=D_rest_initial,
                          lamb_obs=lamb_obs,
                          D_sys=D_sys,
                          D_sys_initial=D_sys_initial,
                          savefig=True,
                          output_dirname=plots_dir)

    # pre-redshift D_rest into all redshift in zgrid
    # D_allz = libs.apply_redshift_all(D_rest, zgrid=zgrid, lamb_in=lamb_rest, lamb_out=lamb_obs, 
                                    # filters=filters, conv=fitting_convolve_filter)
    if training and fit_initial_dicts:
        D_allz_initial = libs.apply_redshift_all(
            D_rest_initial, 
            zgrid=zgrid, 
            lamb_in=lamb_rest, 
            lamb_out=lamb_obs, 
            filters=filters, 
            conv=fitting_convolve_filter)

    if fitting:
        D_allz = dicts.make_D_allz(lamb_obs=lamb_obs, zgrid=zgrid, filters=filters, conv=fitting_convolve_filter)

    # print('')
    # Fitting training catalog if set to True
    if fit_training_catalog and fitting:
        # fit all galaxies with final template
        # print('\nTraining Catalog Redshift Estimation')
        zpeak_trained = np.zeros(Ngal)
        zbest_trained = np.zeros(Ngal)
        zlow_trained = np.zeros(Ngal)
        zhigh_trained = np.zeros(Ngal)
        coefs_trained = np.zeros((Ngal, Ndict+Ndict_sys))
        chi2_trained = np.zeros(Ngal) + np.inf
        last_alpha = np.zeros(Ngal)
        ztrue_pit_trained = np.zeros(Ngal)
        # larslasso_step_trained = np.zeros(Ngal)

        print('')
        for i in range(Ngal):
            # update with number of galaxies processed
            # print(f"\r\t{i+1}/{Ngal} spectra", end="")
            print(f"\rTraining Catalog Redshift Estimation: {i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            # zinput = False
            zinput = ztrue[i]
            alpha_i = 1/np.linalg.norm(spec_obs[i]/err_obs[i])
            if marginalization and (alpha_i >= marginalization_min_alpha) and (alpha_i < marginalization_max_alpha):
                if marginalize_active_only and (alpha_i >= partial_marginalization_min_alpha) and (alpha_i < partial_marginalization_max_alpha):
                    marginalize_active_only_i = True
                else:
                    marginalize_active_only_i = False
                fit_out = libs.marginalize_zgrid(
                    lamb_obs,
                    spec_obs[i],
                    err_obs[i],
                    dictionary=dicts,
                    zinput=zinput, 
                    zgrid=zgrid,
                    filters=filters,
                    marginalize_active_only=marginalize_active_only_i,
                    **marginalize_zgrid_validation_kws
                )
            else:
                fit_out = libs.fit_zgrid(
                    lamb_obs, 
                    spec_obs[i], 
                    err_obs[i], 
                    dictionary=dicts,
                    zinput=zinput, 
                    zgrid=zgrid, 
                    filters=filters, 
                    **fit_zgrid_validation_kws)
            # zpeak, zbest, zlow, zhigh, coefs, b, model, _, _, alpha_path = fit_out
            zpeak = fit_out['zpeak']
            zbest = fit_out['zbest']
            zlow = fit_out['zlower']
            zhigh = fit_out['zupper']
            coefs = fit_out['coef']
            b = fit_out['b']
            model = fit_out['model']
            alpha_path = fit_out['alpha_path']
            ztrue_pit = fit_out['ztrue_pit']

            # store the redshift
            zpeak_trained[i] = zpeak
            zbest_trained[i] = zbest
            zlow_trained[i] = zlow
            zhigh_trained[i] = zhigh
            coefs_trained[i] = coefs
            ztrue_pit_trained[i] = ztrue_pit
            if alpha_path.shape[0] == 1:
                last_alpha[i] = alpha_path[0]
            else:
                last_alpha[i] = alpha_path[-2]
            # idx_minchi2 = np.argmin(chi2_zgrid)
            # chi2_trained[i] = chi2_zgrid[idx_minchi2]
            # larslasso_step_trained[i] = larslasso_step_zgrid[idx_minchi2]
            # calculate chi2 without penalty
            chi2_trained[i] = np.sum(((spec_obs[i]-model)/err_obs[i])**2)

        if larslasso:
            dplots.sparsity_report(coefs_trained=coefs_trained, 
                                    ynorm=ynorms, 
                                    max_feature=max_feature, 
                                    catalog='_training_set_fit',
                                    savefig=True,
                                    output_dirname=plots_dir)

        # fit training catalog with fix z
        # zpeak_trained = np.zeros(Ngal)
        # zbest_trained = np.zeros(Ngal)
        # zlow_trained = np.zeros(Ngal)
        # zhigh_trained = np.zeros(Ngal)
        # coefs_trained = np.zeros((Ngal, D_rest.shape[0]))
        coefs_trained_fixz = np.zeros((Ngal, Ndict+Ndict_sys))
        chi2_trained_fixz = np.zeros(Ngal) + np.inf
        last_alpha_fixz = np.zeros(Ngal)
        # larslasso_step_trained_fixz = np.zeros(Ngal)

        print('')
        for i in range(Ngal):
            # update with number of galaxies processed
            # print(f"\r\t{i+1}/{Ngal} spectra", end="")
            print(f"\rTraining Catalog Redshift Estimation (fix z=ztrue): {i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            zinput = ztrue[i]
            fit_zgrid_validation_kws['error'] = False
            fit_out = libs.fit_zgrid(
                lamb_obs, 
                spec_obs[i], 
                err_obs[i], 
                dictionary=dicts,
                zinput=zinput, 
                zgrid=zgrid, 
                filters=filters, 
                **fit_zgrid_validation_kws, 
                fix_at_zinput=True)
            # _, _, _, _, coefs_fixz, _, model, _, _, alpha_path = fit_out
            coefs_fixz = fit_out['coef']
            model = fit_out['model']
            alpha_path = fit_out['alpha_path']
            # print(chi2_zgrid.shape)
            # store the redshift
            # zpeak_trained[i] = zpeak
            # zbest_trained[i] = zbest
            # zlow_trained[i] = zlow
            # zhigh_trained[i] = zhigh
            # coefs_trained[i] = coefs
            # idx_minchi2_fixz = np.argmin(chi2_zgrid)
            # chi2_trained_fixz[i] = chi2_zgrid[idx_minchi2_fixz]
            chi2_trained_fixz[i] = np.sum(((spec_obs[i]-model)/err_obs[i])**2)
            coefs_trained_fixz[i] = coefs_fixz
            if alpha_path.shape[0] == 1:
                last_alpha_fixz[i] = alpha_path[0]
            else:
                last_alpha_fixz[i] = alpha_path[-2]
            # larslasso_step_trained_fixz[i] = larslasso_step_zgrid[idx_minchi2_fixz]
            fit_zgrid_validation_kws['error'] = True
        if larslasso:
            dplots.sparsity_report(coefs_trained=coefs_trained_fixz, 
                                    ynorm=ynorms, 
                                    max_feature=max_feature, 
                                    catalog='_training_set_fixz',
                                    savefig=True,
                                    output_dirname=plots_dir)

        # print('\nTraining Catalog Untrained Redshift Estimation')
        if fit_initial_dicts:
            # for comparison, fit again with original template
            zpeak_initial = np.zeros(Ngal)
            zbest_initial = np.zeros(Ngal)
            for i in range(Ngal):
                print(f"\rTraining Catalog Untrained Redshift Estimation: {i+1}/{Ngal} sources", end="")
                # fit this spectrum and obtain the redshift
                zinput = False
                spec_obs_i = np.ascontiguousarray(spec_obs[i])
                err_obs_i = np.ascontiguousarray(err_obs[i])
                alpha_i = 1/np.linalg.norm(spec_obs_i/err_obs_i)
                if marginalization and (alpha_i >= marginalization_min_alpha) and (alpha_i < marginalization_max_alpha):
                    if marginalize_active_only and (alpha_i >= partial_marginalization_min_alpha) and (alpha_i < partial_marginalization_max_alpha):
                        marginalize_active_only_i = True
                    else:
                        marginalize_active_only_i = False
                    fit_out = libs.marginalize_zgrid(
                        lamb_obs, 
                        spec_obs_i, 
                        err_obs_i, 
                        lamb_rest=lamb_rest, 
                        D_rest=D_rest_initial, 
                        D_sys=D_sys_initial, 
                        D_allz=D_allz_initial, 
                        D_rest_alphas=D_rest_alphas, 
                        D_rest_alphas_ndict=D_rest_alphas_ndict,
                        D_sys_alphas=D_sys_alphas,
                        D_sys_alphas_ndict=D_sys_alphas_ndict,
                        zinput=zinput, 
                        zgrid=zgrid, 
                        filters=filters, 
                        marginalize_active_only=marginalize_active_only_i,
                        **marginalize_zgrid_validation_kws
                    )
                else:
                    fit_out = libs.fit_zgrid(
                        lamb_obs, 
                        spec_obs_i, 
                        err_obs_i, 
                        lamb_rest=lamb_rest, 
                        D_rest=D_rest_initial, 
                        D_sys=D_sys_initial, 
                        D_allz=D_allz_initial, 
                        D_rest_alphas=D_rest_alphas, 
                        D_rest_alphas_ndict=D_rest_alphas_ndict,
                        D_sys_alphas=D_sys_alphas,
                        D_sys_alphas_ndict=D_sys_alphas_ndict,
                        zinput=zinput, 
                        zgrid=zgrid, 
                        filters=filters, 
                        **fit_zgrid_validation_kws)
                # zpeak_bi, zbest_bi, _, _, _, _, _, _, _, _ = fit_out
                zpeak_bi = fit_out['zpeak']
                zbest_bi = fit_out['zbest']

                zpeak_initial[i] = zpeak_bi
                zbest_initial[i] = zbest_bi
        else:
            zpeak_initial = None
            zbest_initial = None
        
        # dofs = Nfilt - larslasso_step_trained
        # reduced_chi2_trained = chi2_trained/dofs
        ave_residual = chi2_trained/Nfilt
        # dofs_fixz = Nfilt - larslasso_step_trained_fixz
        # reduced_chi2_trained_fixz = chi2_trained_fixz/dofs_fixz
        ave_residual_fixz = chi2_trained_fixz/Nfilt

        # ynorms = np.linalg.norm(spec_obs/err_obs, axis=1)
        # dplots.chi2_plots(ynorms=ynorms, 
        #                   reduced_chi2s=ave_residual, 
        #                   reduced_chi2s_fixz=ave_residual_fixz, 
        #                   last_alpha=last_alpha, 
        #                   last_alpha_fixz=last_alpha_fixz, 
        #                   catalog='training',
        #                   savefig=True,
        #                   output_dirname=plots_dir)

        # dplots.zp_zs_plots(ztrue=ztrue, 
        #                    z_initial=zpeak_initial, 
        #                    z_trained=zpeak_trained, 
        #                    zmin=zmin, 
        #                    zmax=zmax, 
        #                    catalog='training_zpeak', 
        #                    fit_initial=fit_initial_dicts,
        #                    savefig=True,
        #                    output_dirname=plots_dir)
        dplots.zp_zs_plots(ztrue=ztrue, 
                           z_initial=zbest_initial, 
                           z_trained=zbest_trained, 
                           zmin=zmin, 
                           zmax=zmax, 
                           catalog='training_set', 
                           fit_initial=fit_initial_dicts,
                           savefig=True,
                           output_dirname=plots_dir)

        dplots.qq_plots(ztrue_pit=ztrue_pit_trained,
                        output_dirname=plots_dir,
                        output_filename='qq_plot_training_set.png',
                        savefig=True)
        
        # dplots.qq_binplots(ztrue_pit=ztrue_pit_trained,
        #                    zp=zbest_trained,
        #                    zl=zlow_trained,
        #                    zh=zhigh_trained,
        #                    output_dirname=plots_dir,
        #                    output_filename='qq_binplots_training_set.png',
        #                    savefig=True)

        # save estimated redshifts
        np.savez_compressed(output_dir / 'estimated_redshifts_training.npz', 
                            ztrue=ztrue, 
                            zpeak=zpeak_trained, 
                            zbest=zbest_trained, 
                            zlow=zlow_trained, 
                            zhigh=zhigh_trained, 
                            zpeak_initial=zpeak_initial, 
                            zbest_initial=zbest_initial, 
                            coefs=coefs_trained, 
                            idx_cali=idx_cali, 
                            ztrue_pit=ztrue_pit_trained)

    if fitting:
        # Read validation catalog for final fitting
        cat = libs.Catalog(pathfile=validation_catalog, Ndat=Ndat_validation, centering=centering)
        ztrue = cat.ztrue
        print('')

        # first fit with fix z regardless of multiprocessing or not
        lamb_obs = cat.lamb_obs
        if lamb_obs is None:
            if filter_central_wavelengths.split('.')[-1] == 'txt':
                # central_wavelengths_tab = np.loadtxt(filter_central_wavelengths, dtype={'names': ('filter','wavelengths'), 'formats': ('U25','f8')})
                # lamb_obs = central_wavelengths_tab['wavelengths']
                lamb_obs = np.genfromtxt(filter_central_wavelengths)[:,1]
            elif filter_central_wavelengths.split('.')[-1] == 'npz':
                central_wavelengths_tab = np.load(filter_central_wavelengths)
                lamb_obs = central_wavelengths_tab['wavelengths']
        cat.lamb_obs = lamb_obs

        spec_obs = cat.spec_obs
        # spec_obs_original = cat.spec_obs_original
        err_obs = cat.err_obs
        # desi_flag = cat.desi_flag
        # snr = cat.snr_i
        Ngal = len(ztrue)

        chi2_trained_fixz = np.zeros(Ngal) + np.inf
        last_alpha_fixz = np.zeros(Ngal)
        # larslasso_step_trained_fixz = np.zeros(Ngal)

        # print('')
        for i in range(Ngal):
            # update with number of galaxies processed
            # print(f"\r\t{i+1}/{Ngal} spectra", end="")
            print(f"\rValidation Catalog Redshift Estimation (fix z=ztrue):\t\t{i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            zinput = ztrue[i]
            fit_zgrid_validation_kws['error'] = False
            fit_out = libs.fit_zgrid(
                lamb_obs, 
                spec_obs[i], 
                err_obs[i], 
                dictionary=dicts,
                zinput=zinput, 
                zgrid=zgrid, 
                filters=filters, 
                **fit_zgrid_validation_kws, 
                fix_at_zinput=True)
            # _, _, _, _, _, _, model, _, _, alpha_path = fit_out
            model = fit_out['model']
            alpha_path = fit_out['alpha_path']

            # idx_minchi2_fixz = np.argmin(chi2_zgrid)
            chi2_trained_fixz[i] = np.sum(((spec_obs[i]-model)/err_obs[i])**2)
            if alpha_path.shape[0] == 1:
                last_alpha_fixz[i] = alpha_path[0]
            else:
                last_alpha_fixz[i] = alpha_path[-2]
            # larslasso_step_trained_fixz[i] = larslasso_step_zgrid[idx_minchi2_fixz]
            fit_zgrid_validation_kws['error'] = True

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
            coefs_trained = np.zeros((Ngal, Ndict+Ndict_sys))
            chi2_trained = np.zeros(Ngal) + np.inf
            last_alpha = np.zeros(Ngal)
            ztrue_pit_trained = np.zeros(Ngal)
            zstd_trained = np.zeros(Ngal)
            # larslasso_step_trained = np.zeros(Ngal)

            print('')
            for i in range(Ngal):
                # update with number of galaxies processed
                # print(f"\r\t{i+1}/{Ngal} spectra", end="")
                print(f"\rValidation Catalog Redshift Estimation:\t\t{i+1}/{Ngal} sources", end="")
                # fit this spectrum and obtain the redshift
                # zinput = False
                zinput = ztrue[i]
                alpha_i = 1/np.linalg.norm(spec_obs[i]/err_obs[i])
                if marginalization and (alpha_i >= marginalization_min_alpha) and (alpha_i < marginalization_max_alpha):
                    if marginalize_active_only and (alpha_i >= partial_marginalization_min_alpha) and (alpha_i < partial_marginalization_max_alpha):
                        marginalize_active_only_i = True
                    else:
                        marginalize_active_only_i = False
                    fit_out = libs.marginalize_zgrid(
                        lamb_obs, 
                        spec_obs[i], 
                        err_obs[i], 
                        dictionary=dicts,
                        zinput=zinput, 
                        zgrid=zgrid, 
                        filters=filters, 
                        marginalize_active_only=marginalize_active_only_i,
                        **marginalize_zgrid_validation_kws                        
                    )
                else:
                    fit_out = libs.fit_zgrid(
                        lamb_obs, 
                        spec_obs[i], 
                        err_obs[i], 
                        dictionary=dicts,
                        zinput=zinput, 
                        zgrid=zgrid, 
                        filters=filters, 
                        **fit_zgrid_validation_kws)
                # zpeak, zbest, zlow, zhigh, coefs, b, model, _, _, alpha_path = fit_out
                zpeak = fit_out['zpeak']
                zbest = fit_out['zbest']
                zlow = fit_out['zlower']
                zhigh = fit_out['zupper']
                coefs = fit_out['coef']
                b = fit_out['b']
                model = fit_out['model']
                alpha_path = fit_out['alpha_path']
                ztrue_pit = fit_out['ztrue_pit']
                zstd = fit_out['zstd']

                # store the redshift
                zpeak_trained[i] = zpeak
                zbest_trained[i] = zbest
                zlow_trained[i] = zlow
                zhigh_trained[i] = zhigh
                coefs_trained[i] = coefs
                ztrue_pit_trained[i] = ztrue_pit
                zstd_trained[i] = zstd
                # idx_minchi2 = np.argmin(chi2_zgrid)
                chi2_trained[i] = np.sum(((spec_obs[i]-model)/err_obs[i])**2)
                if alpha_path.shape[0] == 1:
                    last_alpha[i] = alpha_path[0]
                else:
                    last_alpha[i] = alpha_path[-2]
                # larslasso_step_trained[i] = larslasso_step_zgrid[idx_minchi2]

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
                    alpha_i = 1/np.linalg.norm(spec_obs_i/err_obs_i)
                    if marginalization and (alpha_i >= marginalization_min_alpha) and (alpha_i < marginalization_max_alpha):
                        if marginalize_active_only and (alpha_i >= partial_marginalization_min_alpha) and (alpha_i < partial_marginalization_max_alpha):
                            marginalize_active_only_i = True
                        else:
                            marginalize_active_only_i = False
                        fit_out = libs.marginalize_zgrid(
                            lamb_obs, 
                            spec_obs_i, 
                            err_obs_i, 
                            dictionary=dicts, 
                            zinput=zinput, 
                            zgrid=zgrid, 
                            filters=filters, 
                            marginalize_active_only=marginalize_active_only_i,
                            **marginalize_zgrid_validation_kws)
                    else:
                        fit_out = libs.fit_zgrid(
                            lamb_obs, 
                            spec_obs_i, 
                            err_obs_i, 
                            dictionary=dicts, 
                            zinput=zinput, 
                            zgrid=zgrid, 
                            filters=filters, 
                            **fit_zgrid_validation_kws)

                        
                    zpeak_bi = fit_out['zpeak']
                    zbest_bi = fit_out['zbest']
                    zlow_bi = fit_out['zlower']
                    zhigh_bi = fit_out['zupper']

                    zpeak_initial[i] = zpeak_bi
                    zbest_initial[i] = zbest_bi
                    zlow_initial[i] = zlow_bi
                    zhigh_initial[i] = zhigh_bi
            else:
                zpeak_initial = None
                zbest_initial = None
                zlow_initial= None
                zhigh_initial = None

        # ===========================================================================================
        # multiprocessing
        # ===========================================================================================
        else:   
            print('')
            cat_val = libs.Catalog(pathfile=validation_catalog, centering=centering)
            # if not marginalization:
            #     validation_kws = fit_zgrid_validation_kws
            # else:
            #     validation_kws = marginalize_zgrid_validation_kws
            fit_cat_results = libs.fit_cat_mp(
                cat=cat_val,
                lamb_obs=lamb_obs,
                zgrid=zgrid,
                dictionary=dicts,
                nthread=mp_threads,
                fit_zgrid_validation_kws=fit_zgrid_validation_kws,
                marginalize_zgrid_validation_kws=marginalize_zgrid_validation_kws,
                marginalization=marginalization,
                marginalize_active_only=marginalize_active_only,
                marginalization_min_alpha=marginalization_min_alpha,
                marginalization_max_alpha=marginalization_max_alpha,
                partial_marginalization_min_alpha=partial_marginalization_min_alpha,
                partial_marginalization_max_alpha=partial_marginalization_max_alpha,
                filters=filters
            )

            zpeak_trained = fit_cat_results['zpeak']
            zbest_trained = fit_cat_results['zbest']
            zmed_trained = fit_cat_results['zmed']
            zlow_trained = fit_cat_results['zlower']
            zhigh_trained = fit_cat_results['zupper']
            zstd_trained = fit_cat_results['zstd']
            coefs_trained = fit_cat_results['coefs']
            bs_trained = fit_cat_results['bs']
            # chi2_trained = np.hstack(chi2_mplist)
            # last_alpha = np.hstack(last_alpha_mplist)
            ztrue_pit_trained = fit_cat_results['ztrue_pit']
            # larslasso_step_trained = np.hstack(larslasso_step_mplist)

            if training and fit_initial_dicts:
                print('')
                fit_cat_results_initial = libs.fit_cat_mp(
                    cat=cat_val,
                    lamb_obs=lamb_obs,
                    zgrid=zgrid,
                    # dictionary=dicts,
                    lamb_rest=dicts.lamb_rest,
                    D_rest=D_rest_initial,
                    D_sys=D_sys_initial,
                    D_allz=D_allz_initial,
                    D_rest_alphas=D_rest_alphas,
                    D_rest_alphas_ndict=D_rest_alphas_ndict,
                    D_sys_alphas=D_sys_alphas,
                    D_sys_alphas_ndict=D_sys_alphas_ndict,
                    nthread=mp_threads,
                    fit_zgrid_validation_kws=fit_zgrid_validation_kws,
                    marginalize_zgrid_validation_kws=marginalize_zgrid_validation_kws,
                    marginalization=marginalization,
                    marginalize_active_only=marginalize_active_only,
                    marginalization_min_alpha=marginalization_min_alpha,
                    marginalization_max_alpha=marginalization_max_alpha,
                    partial_marginalization_min_alpha=partial_marginalization_min_alpha,
                    partial_marginalization_max_alpha=partial_marginalization_max_alpha,
                    filters=filters
                )

                zpeak_initial = fit_cat_results_initial['zpeak']
                zbest_initial = fit_cat_results_initial['zbest']
                zlow_initial = fit_cat_results_initial['zlower']
                zhigh_initial = fit_cat_results_initial['zupper']
                zstd_initial = fit_cat_results_initial['zstd']

            else:
                zpeak_initial = None
                zbest_initial = None
                zlow_initial= None
                zhigh_initial = None
                zstd_initial = None


        # plotting_training = (training & fit_training_catalog)

        # dofs = Nfilt - larslasso_step_trained
        # reduced_chi2_trained = chi2_trained/dofs
        # ave_residual = chi2_trained/Nfilt
        # dofs_fixz = Nfilt - larslasso_step_trained_fixz
        # reduced_chi2_trained_fixz = chi2_trained_fixz/dofs_fixz
        # ave_residual_fixz = chi2_trained_fixz/Nfilt
        ynorms = np.linalg.norm(cat.spec_obs/cat.err_obs, axis=1)

        # dplots.chi2_plots(ynorms=ynorms, 
        #                   reduced_chi2s=ave_residual, 
        #                   reduced_chi2s_fixz=ave_residual_fixz, 
        #                   last_alpha=last_alpha, 
        #                   last_alpha_fixz=last_alpha_fixz, 
        #                   catalog='fitting',
        #                   savefig=True,
        #                   output_dirname=plots_dir)

        # Create zphot vs zspec plots
        # dplots.zp_zs_plots(ztrue=ztrue, 
        #                    z_initial=zpeak_initial, 
        #                    z_trained=zpeak_trained, 
        #                    zl=zlow_trained, 
        #                    zh=zhigh_trained, 
        #                    zmin=zmin, 
        #                    zmax=zmax, 
        #                    catalog='fitting_zpeak', 
        #                    fit_initial=fit_initial_dicts,
        #                    savefig=True,
        #                    output_dirname=plots_dir)
        dplots.zp_zs_plots(ztrue=ztrue, 
                           z_initial=zbest_initial, 
                           z_trained=zbest_trained, 
                           zl=zlow_trained, 
                           zh=zhigh_trained, 
                           zmin=zmin, 
                           zmax=zmax, 
                           catalog='validation_set', 
                           fit_initial=fit_initial_dicts,
                           savefig=True,
                           output_dirname=plots_dir)
        
        dplots.qq_plots(ztrue_pit=ztrue_pit_trained,
                        output_dirname=plots_dir,
                        output_filename='qq_plot_validation_set.png',
                        savefig=True)
        
        dplots.qq_binplots(ztrue_pit=ztrue_pit_trained,
                           zp=zbest_trained,
                           zl=zlow_trained,
                           zh=zhigh_trained,
                           output_dirname=plots_dir,
                           output_filename='qq_binplots_validation_set.png',
                           savefig=True)

        # Create 6 bin uncertainty fraction and zscore plots
        # dplots.uncertainty_binplots(zs=ztrue, 
        #                             zp=zpeak_trained, 
        #                             zl=zlow_trained, 
        #                             zh=zhigh_trained, 
        #                             zp0=zpeak_initial, 
        #                             zl0=zlow_initial, 
        #                             zh0=zhigh_initial, 
        #                             ztype='zpeak', 
        #                             fit_initial=fit_initial_dicts,
        #                             savefig=True,
        #                             output_dirname=plots_dir)
        dplots.uncertainty_binplots(zs=ztrue, 
                                    zp=zbest_trained, 
                                    zl=zlow_trained, 
                                    zh=zhigh_trained, 
                                    zp0=zbest_initial, 
                                    zl0=zlow_initial, 
                                    zh0=zhigh_initial, 
                                    ztype='zbest', 
                                    fit_initial=fit_initial_dicts,
                                    savefig=True,
                                    output_dirname=plots_dir)
        # Create 6 bin zphot vs zspec hexbin plots
        # dplots.hexbin_binplot(zs=ztrue, 
        #                       zp=zpeak_trained, 
        #                       zl=zlow_trained, 
        #                       zh=zhigh_trained, 
        #                       zp0=zpeak_initial, 
        #                       zl0=zlow_initial, 
        #                       zh0=zhigh_initial, 
        #                       ztype='zpeak', 
        #                       fit_initial=fit_initial_dicts,
        #                       savefig=True,
        #                       output_dirname=plots_dir)
        dplots.hexbin_binplot(zs=ztrue, 
                              zp=zbest_trained, 
                              zl=zlow_trained, 
                              zh=zhigh_trained, 
                              zp0=zbest_initial, 
                              zl0=zlow_initial, 
                              zh0=zhigh_initial, 
                              ztype='zbest', 
                              fit_initial=fit_initial_dicts,
                              savefig=True,
                              output_dirname=plots_dir)
        # save estimated redshifts
        np.savez_compressed(output_dir / 'estimated_redshifts.npz', 
                            ztrue=ztrue, 
                            zpeak=zpeak_trained, 
                            zbest=zbest_trained, 
                            zmed=zmed_trained,
                            zlow=zlow_trained, 
                            zhigh=zhigh_trained, 
                            ztrue_pit=ztrue_pit_trained,
                            zstd=zstd_trained,
                            zpeak_initial=zpeak_initial, 
                            zbest_initial=zbest_initial, 
                            zlow_initial=zlow_initial, 
                            zhigh_initial=zhigh_initial, 
                            zstd_initial=zstd_initial,
                            coefs=coefs_trained, 
                            bs=bs_trained,
                            idx_cali=idx_cali)

        # Sparsity plot
        if larslasso:
            try:    # TEMP no idea why it keeps breaking
                dplots.sparsity_report(coefs_trained=coefs_trained, 
                                    ynorm=ynorms, 
                                    max_feature=max_feature, 
                                    #    add_constant=add_constant,
                                    savefig=True,
                                    output_dirname=plots_dir)
            except:
                pass
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
        dplots.example_seds(idx=idx_examples, 
                            cat=cat, 
                            lamb_rest=lamb_rest, 
                            D_rest=D_rest, 
                            D_rest_alphas=D_rest_alphas,
                            D_rest_alphas_ndict=D_rest_alphas_ndict, 
                            D_rest_initial=D_rest_initial, 
                            D_sys=D_sys, 
                            D_sys_initial=D_sys_initial,
                            zgrid=zgrid, 
                            filters=filters, 
                            validation_fit_kws=fit_zgrid_validation_kws, 
                            ztype='zpeak',
                            savefig=True,
                            output_dirname=plots_dir)
        # makefigs.example_seds(idx=idx_examples, cat=cat, lamb_rest=lamb_rest, D_rest=D_rest, D_rest_initial=D_rest_initial, \
        #                       zgrid=zgrid, filters=filters, validation_fit_kws=fit_zgrid_validation_kws, ztype='zbest')
        
        # fit EAZY with trained dictionaries
        # makefigs.fit_eazy_plots(lamb_rest, D_rest, templates_EAZY)

    tfc = time.time()
    print('\nElapsed Time = '+str(tfc-tic)+' seconds')

    end_datetime = datetime.now().isoformat(timespec='seconds')
    if type(larslasso_alpha_train) == np.ndarray:
        larslasso_alpha_train = larslasso_alpha_train.tolist()
    if type(larslasso_alpha_fit) == np.ndarray:
        larslasso_alpha_fit = larslasso_alpha_fit.tolist()
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
                'active_alpha_0_training': active_alpha_0_training,
                'active_alpha_0_fitting': active_alpha_0_fitting,
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



if __name__ == "__main__":
    main()
