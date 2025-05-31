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
evaluation_catalog = config.evaluation_catalog
Ndat = config.Ndat
Ndat_evaluation = config.Ndat_evaluation
Ncalibrators = config.Ncalibrators
use_DESI_flag = config.use_DESI_flag
calibrator_SNR51 = config.calibrator_SNR51
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
Niterations =config.Niterations
algorithm = config.algorithm
fix_z = config.fix_z
centering = config.centering
AB_update_tolerance = config.AB_update_tolerance
max_AB_loops = config.max_AB_loops
remove_old_ab_info = config.remove_old_ab_info
learning_rate0 = config.learning_rate0
learning_rate_cali = config.learning_rate_cali
# LARSlasso configurations
larslasso = config.larslasso
lars_alpha = config.lars_alpha
LARSlasso_alpha_sigma = config.LARSlasso_alpha_sigma
lars_positive = config.lars_positive
best_cp = config.best_cp
max_feature = config.max_feature
active_OLS_training = config.active_OLS_training
active_OLS_fitting = config.active_OLS_fitting
center_Xy = config.center_Xy
unit_X = config.unit_X
unit_y = config.unit_y
LARSlasso_alpha_scaling = config.LARSlasso_alpha_scaling
# Fitting configurations
probline = config.probline
fit_training_catalog = config.fit_training_catalog
convolve_filter = config.convolve_filter
last_stage_convolve_filter = config.last_stage_convolve_filter
fitting_convolve_filter = config.fitting_convolve_filter
# Zgrid configurations
zmax = config.zmax
zmin = config.zmin
dz = config.dz
scale_1plusz = config.scale_1plusz
testing_zgrid = config.testing_zgrid
# Directory locations
eazy_templates_location = config.eazy_templates_location
filter_location = config.filter_location
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
lamb_obs = cat.lamb_obs
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
    h_cali = cat.snr_i > calibrator_SNR51
    idx_cali = idx_original[h_cali][:Ncalibrators] # index for calibration galaxies within read input
    use_DESI_flag = False



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
if filter_location is not None:
    filter_info = fx.read_filters(filter_location)
else:
    filter_info = None


# add a tiny number to z=0.0 so that if zinput is given as 0 it will not be considered False
for i in range(len(ztrue)):
    if ztrue[i] == 0.0:
        ztrue[i] = 0.00000001

# Read dictionaries from file
if dict_read_from_file:
    D_rest_input = np.load(dict_read_from_file)
    lamb_rest = D_rest_input['lamb_rest']
    D_rest = D_rest_input['D_rest']
    if Ndict >= len(D_rest):
        Ndict = len(D_rest)
    else:
        D_rest = D_rest[:Ndict]
    D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
    if f_lambda_mode and not D_rest_input['save_flambda']:
        D_rest = fx.fnu2flambda(lamb_rest*1e5, D_rest)
        D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
    elif not f_lambda_mode and D_rest_input['save_flambda']:
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
    lamb_rest = np.arange(0.01,6.0,0.01)
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
                             num_EAZY_as_dict=num_EAZY_as_dict, lamb_rest=lamb_rest)

D_rest_initial = D_rest.copy()


tic = time.time()

# Grab the Ngal for evaluation catalog just for output information
with np.load(evaluation_catalog) as fitting_dat:
    Ndat_evaluation = min(len(fitting_dat['z']), Ndat_evaluation)


# define fitting worker function
def fitting_mp(child, cat, lamb_rest, D_rest, D_allz, zgrid, filter_info, fit_zgrid_validation_kws, verbose=False):
    lamb_obs = cat.lamb_obs
    ztrue = cat.ztrue
    spec_obs = cat.spec_obs
    err_obs = cat.err_obs

    Ngal = len(ztrue)

    zbest_fitted = np.zeros(Ngal)
    zlow_fitted = np.zeros(Ngal)
    zhigh_fitted = np.zeros(Ngal)
    coefs_fitted = np.zeros((Ngal, D_rest.shape[0]))

    for i in range(Ngal):
        if verbose:
            print(f"\rValidation Catalog Segment Redshift Estimation:\t\t{i+1}/{Ngal} sources", end="")
        spec_obs_i = np.ascontiguousarray(spec_obs[i])
        err_obs_i = np.ascontiguousarray(err_obs[i])
        z,zlow,zhigh,coefs,b,model,_,_ = fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest, D_rest=D_rest, D_allz=D_allz, zinput=False, 
                                                    zgrid=zgrid, filter_info=filter_info, **fit_zgrid_validation_kws)
        zbest_fitted[i] = z
        zlow_fitted[i] = zlow
        zhigh_fitted[i] = zhigh
        coefs_fitted[i] = coefs

    results = (zbest_fitted, zlow_fitted, zhigh_fitted, coefs_fitted)
    child.send(results)
    child.close()




if __name__ == "__main__":

    # print(f"Algorithm = {algorithm}")
    print(f"Training Catalog:\t{training_catalog} ({Ngal} sources)")
    print(f"Evaluation Catalog:\t{evaluation_catalog} ({Ndat_evaluation} sources)")
    if dict_read_from_file:
        print(f"Dictionaries:\t\t{dict_read_from_file}")
    else:
        print(f"{num_EAZY_as_dict} of 7 EAZY templates used as initialized dictionaries")    
    print('')
    print(f"Ndict = {Ndict}")
    print(f"Add constant: {add_constant}")
    if fix_dicts > 0:
        print(f"Fix {fix_dicts} dictionaries from the end of list")
    print(f"Centering: {centering}")    
    print('')
    print(f"Fix z in training: {fix_z}")
    print(f"Remove old AB info: {remove_old_ab_info}")
    if use_DESI_flag:
        print(f"Ncalibrators = {Ncalibrators} (DESI Flags)")
    elif Ncalibrators>0:
        print(f"Ncalibrators = {Ncalibrators} (SNR > {calibrator_SNR51})")
    # print(f"Convolving filters: 1st stage:{convolve_filter}, 2nd stage:{last_stage_convolve_filter}, fitting:{fitting_convolve_filter}")
    if algorithm == 0:
        print(f"Learning rates = {learning_rate0}/{learning_rate_cali}")
    else:
        if larslasso:
            # print(fr"LARSlasso: {larslasso} (alpha={lars_alpha} + {LARSlasso_alpha_sigma} sigma, Positive: {lars_positive}, max_feature={max_feature}, center_Xy={center_Xy})")
            print(f"LARS-lasso: {larslasso}")
            print(f"\talpha = {lars_alpha} + {LARSlasso_alpha_sigma} sigma")
            print(f"\tPositive = {lars_positive}")
            print(f"\tcenter_Xy = {center_Xy}")
            print(f"\tunit_X = {unit_X}")
            print(f"\tunit_y = {unit_y}")
            if max_feature:
                print(f"\tmax_feature = {max_feature}")
            if best_cp:
                print(f"\tBest Cp = {best_cp}")
        else:
            print(f"LARS-lasso: {larslasso}")

    if testing_zgrid:
        print('Testing Zgrid: On')
    print('')

    # Initialize A and B matrices for dictionary updates
    A = np.zeros((D_rest.shape[0], D_rest.shape[0]))
    B = np.zeros((len(lamb_rest), D_rest.shape[0]))

    if remove_old_ab_info: # save previous A and B for later removal
        A_history = np.zeros((Ngal, D_rest.shape[0], D_rest.shape[0]))
        B_history = np.zeros((Ngal, len(lamb_rest), D_rest.shape[0]))

    # Training iterations
    for i_iter in range(Niterations):
        # print(f"\n{i_iter+1} of {Niterations} iterations")
        # print(f"\rTraining iterations: {i_iter+1}/{Niterations} ({Ngal} sources)", end="")

        # pick out calibration galaxies
        idx_other = np.setdiff1d(idx_original, idx_cali)
        np.random.shuffle(idx_cali)
        np.random.shuffle(idx_other)
        idx_shuffle = np.hstack((idx_cali, idx_other))

        for i_gal0 in range(Ngal): 
            i_gal = idx_shuffle[i_gal0]
            # update with number of galaxies processed
            # print(f"\r\t{i_gal0+1}/{Ngal} spectra", end="")
            print(f"\rTraining iterations: {i_iter+1}/{Niterations} ({i_gal0+1}/{Ngal} sources)", end="")
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
            z, zlow, zhigh, coefs, b, model, _, _= fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_D=lamb_rest, D_rest=D_rest, zinput=zinput, 
                                                                      zgrid=zgrid, filter_info=filter_info, **fit_zgrid_training_kws)
            # print(model)
            # update the spectral dictionary using the residuals between the model and data
            residual = spec_obs[i_gal] - model
            j_update = np.where((lamb_rest > np.min(lamb_obs)/(1+z)) & (lamb_rest < np.max(lamb_obs)/(1+z)))[0]     # Find overlap between this spectra in rest-frame and dictionary
            j_update_outside = np.setdiff1d(np.arange(len(lamb_rest)), j_update)

            interpolated_residual = np.interp(lamb_rest[j_update], lamb_obs/(1+z), residual)                          # interpolate the residual to these values
            interpolated_spec_obs = np.interp(lamb_rest[j_update], lamb_obs/(1+z), spec_obs[i_gal])
            # construct the model from best fit coefficients and dictionaries
            error_rest = np.interp(lamb_rest, lamb_obs/(1+z), err_obs[i_gal])
            model_rest = (D_rest).T @ coefs + b * error_rest
            model_rest[j_update] = interpolated_spec_obs                                                            # replace the overlapped part with interpolated observed spectra 
            # error_rest[j_update_outside] = min(error_rest)   # TESTING
                                                                                                                      # this will be considered the observed spectra for update purpose; outside overlap range just use model (no update)
            # TESTING
            # error_rest = np.ones_like(model_rest)
            # error_rest = np.interp(lamb_rest, lamb_obs/(1+z), err_obs[i_gal])
            # residue_factor = 1/error_rest * min(error_rest)
            # AB_factor = 1.0

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
                    # TESTING
                    A -= A_history[i_gal]
                    B -= B_history[i_gal]
                    A_history[i_gal] = coefs[:,None] @ coefs[:,None].T
                    # A_history[i_gal] = coefs[:,None] @ coefs[:,None].T * AB_factor
                    B_history[i_gal] = model_rest[:,None] @ coefs[:,None].T                    
                    # B_history[i_gal] = model_rest[:,None] @ coefs[:,None].T * residue_factor[:,None] * AB_factor
                    A += A_history[i_gal]
                    B += B_history[i_gal]
                else:
                    A += coefs[:,None] @ coefs[:,None].T
                    # A += coefs[:,None] @ coefs[:,None].T * AB_factor
                    B += model_rest[:,None] @ coefs[:,None].T
                    # TESTING 
                    # B += model_rest[:,None] @ coefs[:,None].T * residue_factor * AB_factor

                # iterative over A and B matrices until converge or max_AB_loop
                for j in range(D_rest.shape[0]-add_constant-fix_dicts):
                    score = 1.0
                    Dj_old = D_rest[j].copy()
                    abi = 0
                    while score >= AB_update_tolerance and abi < max_AB_loops:
                        diag_Aj = np.diagonal(A)[j]
                        if diag_Aj != 0:
                            uj = 1/diag_Aj * (B[:,j] - (D_rest.T @ A[j])) + D_rest[j]
                            # TESTING
                            # uj = 1/diag_Aj * (B[:,j] - ((D_rest*0.5).T @ A[j])) + D_rest[j]
                            uj_norm = np.linalg.norm(uj)
                            D_rest[j] = uj/max(1, uj_norm)
                            score = np.linalg.norm(D_rest[j]-Dj_old)/np.linalg.norm(Dj_old)
                            abi += 1
                        else:
                            break


    np.savez_compressed(output_dir / 'trained_templates.npz',lamb_rest=lamb_rest, D_rest=D_rest)

    # Make template plots and multiplots
    makefigs = dplots.diagnostic_plots(output_dirname=plots_dir)
    makefigs.template_plots(lamb_rest=lamb_rest, D_rest=D_rest[:Ndict], D_rest_initial=D_rest_initial[:Ndict])


    # pre-redshift D_rest into all redshift in zgrid
    D_allz = fx.apply_redshift_all(D_rest, zgrid=zgrid, lamb_in=lamb_rest, lamb_out=lamb_obs, 
                                    filter_info=filter_info, conv=fitting_convolve_filter)
    D_allz_initial = fx.apply_redshift_all(D_rest_initial, zgrid=zgrid, lamb_in=lamb_rest, lamb_out=lamb_obs, 
                                    filter_info=filter_info, conv=fitting_convolve_filter)

    # print('')
    # Fitting training catalog if set to True
    if fit_training_catalog:
        # fit all galaxies with final template
        # print('\nTraining Catalog Redshift Estimation')
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
            z,zlow,zhigh,coefs,b,model,_,_ = fit_zgrid(lamb_obs, spec_obs[i], err_obs[i], lamb_rest, D_rest=D_rest, D_allz=D_allz, zinput=zinput, 
                                                    zgrid=zgrid, filter_info=filter_info, **fit_zgrid_validation_kws)
            # store the redshift
            zbest_trained[i] = z
            zlow_trained[i] = zlow
            zhigh_trained[i] = zhigh
            coefs_trained[i] = coefs

        # for comparison, fit again with original template
        zbest_initial = np.zeros(Ngal)
        # print('\nTraining Catalog Untrained Redshift Estimation')
        for i in range(Ngal):
            print(f"\rTraining Catalog Untrained Redshift Estimation: {i+1}/{Ngal} sources", end="")
            # fit this spectrum and obtain the redshift
            zinput = False
            spec_obs_i = np.ascontiguousarray(spec_obs[i])
            err_obs_i = np.ascontiguousarray(err_obs[i])
            z_bi,zlow_bi,zhigh_bi,coefs_bi,b_bi,model_bi,_,_ = fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest, D_rest=D_rest_initial,
                                                D_allz=D_allz_initial, zinput=zinput, zgrid=zgrid, filter_info=filter_info, **fit_zgrid_validation_kws)
            zbest_initial[i] = z_bi

        makefigs.zp_zs_plots(ztrue=ztrue, zbest_initial=zbest_initial, zbest_trained=zbest_trained, zmin=zmin, zmax=zmax, catalog='training')



    # Read evaluation catalog for final fitting
    cat = fx.Catalog(pathfile=evaluation_catalog, Ndat=Ndat_evaluation, centering=centering)
    segment_idx = 10000
    cat1 = fx.Catalog(pathfile=evaluation_catalog, Ndat=segment_idx, centering=centering)
    cat2 = fx.Catalog(pathfile=evaluation_catalog, Ndat=Ndat_evaluation, istart=segment_idx, centering=centering)

    lamb_obs = cat.lamb_obs
    ztrue = cat.ztrue
    spec_obs = cat.spec_obs
    spec_obs_original = cat.spec_obs_original
    err_obs = cat.err_obs
    desi_flag = cat.desi_flag
    snr = cat.snr_i

    Ngal = len(ztrue)
    parent_t1, child_t1 = mp.Pipe(duplex=False)
    parent_t2, child_t2 = mp.Pipe(duplex=False)
    parent_u1, child_u1 = mp.Pipe(duplex=False)
    parent_u2, child_u2 = mp.Pipe(duplex=False)

    print('')
    p_t1 = mp.Process(target=fitting_mp, args=(child_t1, cat1, lamb_rest, D_rest, D_allz, 
                                                    zgrid, filter_info, fit_zgrid_validation_kws, True))
    p_t1.start()

    p_t2 = mp.Process(target=fitting_mp, args=(child_t2, cat2, lamb_rest, D_rest, D_allz, 
                                                    zgrid, filter_info, fit_zgrid_validation_kws, False))
    p_t2.start()

    p_u1 = mp.Process(target=fitting_mp, args=(child_u1, cat1, lamb_rest, D_rest_initial, D_allz_initial, 
                                                    zgrid, filter_info, fit_zgrid_validation_kws, False))
    p_u1.start()

    p_u2 = mp.Process(target=fitting_mp, args=(child_u2, cat2, lamb_rest, D_rest_initial, D_allz_initial, 
                                                    zgrid, filter_info, fit_zgrid_validation_kws, False))
    p_u2.start()

    zbest_t1, zlow_t1, zhigh_t1, coefs_t1 = parent_t1.recv()
    zbest_t2, zlow_t2, zhigh_t2, coefs_t2 = parent_t2.recv()
    zbest_u1, zlow_u1, zhigh_u1, coefs_u1 = parent_u1.recv()
    zbest_u2, zlow_u2, zhigh_u2, coefs_u2 = parent_u2.recv()

    try:
        p_t1.join()
        p_t2.join()
        p_u1.join()
        p_u2.join()
    except KeyboardInterrupt:
        p_t1.terminate()
        p_t1.join()
        p_t2.terminate()
        p_t2.join()
        p_u1.terminate()
        p_u1.join()
        p_u2.terminate()
        p_u2.join()

    zbest_trained = np.hstack((zbest_t1, zbest_t2))
    zlow_trained = np.hstack((zlow_t1, zlow_t2))
    zhigh_trained = np.hstack((zhigh_t1, zhigh_t2))
    coefs_trained = np.vstack((coefs_t1, coefs_t2)) # coefs_trained is 2d array
    zbest_initial = np.hstack((zbest_u1, zbest_u2))
    zlow_initial = np.hstack((zlow_u1, zlow_u2))
    zhigh_initial = np.hstack((zhigh_u1, zhigh_u2))

    # zbest_trained = zbest_t1
    # zlow_trained = zlow_t1
    # zhigh_trained = zhigh_t1
    # coefs_trained = coefs_t1
    # zbest_initial = zbest_u1
    # zlow_initial = zlow_u1
    # zhigh_initial = zhigh_u1

    # Create zphot vs zspec plots
    makefigs.zp_zs_plots(ztrue=ztrue, zbest_initial=zbest_initial, zbest_trained=zbest_trained, zmin=zmin, zmax=zmax, catalog='fitting')
    # Create 6 bin uncertainty fraction and zscore plots
    makefigs.uncertainty_binplots(zs=ztrue, zp=zbest_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zbest_initial, zl0=zlow_initial, zh0=zhigh_initial)
    # Create 6 bin zphot vs zspec hexbin plots
    makefigs.hexbin_binplot(zs=ztrue, zp=zbest_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zbest_initial, zl0=zlow_initial, zh0=zhigh_initial)

    # save estimated redshifts
    np.savez_compressed(output_dir / 'estimated_redshifts.npz', ztrue=ztrue, zest=zbest_trained, zlow=zlow_trained, zhigh=zhigh_trained, 
             zest_initial=zbest_initial, zlow_initial=zlow_initial, zhigh_initial=zhigh_initial, coefs=coefs_trained, idx_cali=idx_cali)

    # Sparsity plot
    if larslasso:
        makefigs.sparsity_report(coefs_trained=coefs_trained, max_feature=max_feature, add_constant=add_constant)

    # select galaxy examples based on redshift error bin
    idx_examples = []
    idx_evaluation = np.arange(Ndat_evaluation, dtype=int)
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
        idx_in_bin = idx_evaluation[h]
        if len(idx_in_bin) >= idx_num_perbin[i]:
            selected_idx = np.random.choice(idx_in_bin, size=idx_num_perbin[i]).tolist()
            idx_examples.extend(selected_idx)
        else:
            idx_num_perbin[i+1] += 1

    # Create example SED plots
    makefigs.example_seds(idx=idx_examples, cat=cat, lamb_rest=lamb_rest, D_rest=D_rest, D_rest_initial=D_rest_initial, \
                          zgrid=zgrid, filter_info=filter_info, validation_fit_kws=fit_zgrid_validation_kws)

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
            'Evaluation Catalog': evaluation_catalog,
            'Dictionary': dict_read_from_file,
            'Ndict': Ndict,
            'Ndat_training': Ndat,
            'Ndat_evaluation': Ndat_evaluation,
        },
        'Algorithm': {
            'Fix Z in training': fix_z,
            'Centering': centering,
            'remove_old_AB_info': remove_old_ab_info,
        },
        'LARSlasso': {
            'LARSlasso': larslasso,
            'Positive': lars_positive,
            'Alpha': lars_alpha,
            'Alpha sigma': LARSlasso_alpha_sigma,
            'max_feature': max_feature,
            'active_OLS_training': active_OLS_training,
            'active_OLS_fitting': active_OLS_fitting,
            'center_Xy': center_Xy,
        }

}

with open(output_dir / parameters_report, 'w') as output_file:
    yaml.safe_dump(output_reports, output_file, sort_keys=False)


