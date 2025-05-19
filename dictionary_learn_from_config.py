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
import dictionary_learn_fx as fx # type: ignore
import yaml


if len(sys.argv) == 1:
    print(__doc__)
    exit()

# Input config file
config_file = sys.argv[1]
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Catalog configurations
training_catalog = config['Catalog']['training_catalog']
evaluation_catalog = config['Catalog']['evaluation_catalog']
Ndat = config['Catalog']['Ndat_training']
Ndat_evaluation = config['Catalog']['Ndat_evaluation']
Ncalibrators = config['Catalog']['Ncalibrators']
use_DESI_flag = config['Catalog']['use_DESI_flag']
calibrator_SNR51 = config['Catalog']['calibrator_SNR']
f_lambda_mode = config['Catalog']['f_lambda_mode']    # fitting in f_lambda or f_nu

# Dictionary input configurations
dict_read_from_file = config['Dictionary']['read_from_file']
add_constant = config['Dictionary']['add_constant']
fix_dicts = config['Dictionary']['Fix_dicts']
Ndict = config['Dictionary']['Ndict']
num_EAZY_as_dict = config['Dictionary']['num_EAZY_as_dict']
dicts_fluctuation_scaling_const = config['Dictionary']['dict_fluctuation_scaling_start']
if type(dicts_fluctuation_scaling_const) == str:
    dicts_fluctuation_scaling_const = float(dicts_fluctuation_scaling_const)
dict_fluctuation_scaling_base = config['Dictionary']['dict_fluctuation_scaling_base']

# Algorithm configurations
Niterations = config['Algorithm']['Niterations']
algorithm = config['Algorithm']['update_algorithm']             # choose which algorithm to use for dictionary updates, 0: psudo-inverse vector method, 1: paper method
fix_z = config['Algorithm']['fix_z']
centering = config['Algorithm']['Centering']
AB_update_tolerance = config['Algorithm']['AB_update_tolerance']
max_AB_loops = config['Algorithm']['max_update_loops']            # if algorithm = 1, number of loops to run updates on dictionaries
remove_old_ab_info = config['Algorithm']['remove_old_ab_info']
# residue_factor = config['Algorithm']['residue_factor']
# AB_factor = config['Algorithm']['AB_factor']
learning_rate0 = config['Algorithm']['learning_rate0']
learning_rate_cali = config['Algorithm']['learning_rate_cali']

# LARSlasso Configurations
larslasso = config['LARSlasso']['LARSlasso']
lars_alpha = config['LARSlasso']['alpha']
LARSlasso_alpha_sigma = config['LARSlasso']['alpha_sigma']
lars_positive = config['LARSlasso']['positive']
# LARSlasso_alpha_selection_only = config['LARSlasso']['alpha_selection_only']
# LARSlasso_alpha_scaling = config['LARSlasso']['alpha_scaling']

# Fitting configurations
# probline = config['Fitting']['probline']
fit_training_catalog = config['Fitting']['fit_training_catalog']
convolve_filters = config['Fitting']['convolve_filters']
# convolve_filter = convolve_filters[0] # choose to convolve templates with filter or not in the first stage of optimized grid search
# last_stage_convolve_filter = convolve_filters[1]   # whether to colvolve with filters in the last stage of grid search 
fitting_convolve_filter = convolve_filters[2] # Whether to convolve with filters in the end when fitting final redshifts

# Zgrid configurations
zmax = config['Zgrid']['zmax']
zmin = config['Zgrid']['zmin']
dz = config['Zgrid']['dz']
scale_1plusz = config['Zgrid']['scale_1plusz']
testing_zgrid = config['Zgrid']['testing_zgrid']

zgrid = fx.generate_zgrid(zmin=zmin, zmax=zmax, dz=dz, scale_1plusz=scale_1plusz, testing_zgrid=testing_zgrid)

# Directory locations
eazy_templates_location = config['Directory_locations']['eazy_templates_location']
filter_location = config['Directory_locations']['filter_location']
output_dirname = config['Directory_locations']['OUTPUT']

# some common keywords for fit_zgrid function as dictionary
fit_kws = fx.keywords(config_file)
train_fit_kws = fit_kws.train_fit()
validation_fit_kws = fit_kws.validation_fit()

fit_zgrid = fx.fit_zgrid


cat = fx.Catalog(pathfile=training_catalog, Ndat=Ndat, centering=centering)
lamb_obs = cat.lamb_obs
ztrue = cat.ztrue
spec_obs = cat.spec_obs
spec_obs_original = cat.spec_obs_original
err_obs = cat.err_obs
desi_flag = cat.desi_flag
snr = cat.snr

Ngal = len(cat.ztrue)
if Ngal < Ndat:
    Ndat = Ngal

# 51th channel SNR require for calibration galaxies
idx_original = np.arange(Ngal)  # index corresponding to read input
if use_DESI_flag:
    h_cali = desi_flag == 1.0
    idx_cali = idx_original[h_cali]
    Ncalibrators = np.sum(h_cali)
else:
    h_cali = cat.snr_i > calibrator_SNR51
    idx_cali = idx_original[h_cali][:Ncalibrators] # index for calibration galaxies within read input

# Find indices for plotting final fitted SED
snr_max = np.max(snr, axis=1)
qs = [99.9, 95, 85, 75, 55, 15] # target SNR percentile for example SEDs
idx_plot_sed = []
for i in range(len(qs)):
    snr_target = np.percentile(snr_max, qs[i])
    idx_target = np.argmin(np.abs(snr_max - snr_target))
    idx_plot_sed.append(idx_target)

# Make sure directory strings have "/" at the end; if output directory doesn't exist, create one
if len(output_dirname) > 0:
    output_dirname = output_dirname.rstrip('/') + '/'
output_dir = Path(output_dirname)
if len(output_dirname) > 0 and not output_dir.is_dir():
    output_dir.mkdir()
eazy_templates_location = eazy_templates_location.rstrip('/') + '/'
if filter_location != 'None':
    filter_info = fx.read_filters(filter_location)
else:
    filter_info = None


# add a tiny number to z=0.0 so that if zinput is given as 0 it will not be considered False
for i in range(len(ztrue)):
    if ztrue[i] == 0.0:
        ztrue[i] = 0.00000001


# preparing to initialize dictionaries with lamb_rest and EAZY templates
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
        D_rest = np.vstack((D_rest, np.ones_like(lamb_rest)))
    # lamb_rest_resolution = np.diff(lamb_rest)[0]
    lamb_rest_resolution = np.mean(np.diff(lamb_rest))
else:
    lamb_rest = np.arange(0.01,6.0,0.01)
    lamb_rest_resolution = 0.01

# Read EAZY templates for dictionary initialization and also evaluation test
templates_EAZY = fx.load_EAZY(lamb_rest, eazy_templates_location)
if not f_lambda_mode:
    templates_EAZY = fx.flambda2fnu(lamb_rest*10000, templates_EAZY)
templates_EAZY = np.array(templates_EAZY)
if centering:
    templates_EAZY = templates_EAZY - np.mean(templates_EAZY, axis=1)[:,None]

# initialize dictionaries
if not dict_read_from_file:
    # If initializing dictionaries, create them as noise with different level of fluctuation
    dictionary_fluctuation_scaling = np.array([dicts_fluctuation_scaling_const/(dict_fluctuation_scaling_base**i) for i in range(Ndict-num_EAZY_as_dict)])
    D_rest = fx.initialize_dicts(Ndict, dictionary_fluctuation_scaling=dictionary_fluctuation_scaling, templates_EAZY=templates_EAZY, 
                             num_EAZY_as_dict=num_EAZY_as_dict, lamb_rest=lamb_rest)

D_rest_initial = D_rest.copy()


tic = time.time()



with np.load(evaluation_catalog) as fitting_dat:
    Ngal_evaluation = min(len(fitting_dat['z']), Ndat_evaluation)


# def main():
if __name__ == "__main__":

    # print(f"Algorithm = {algorithm}")
    print(f"Training Catalog:\t{training_catalog} ({Ngal} sources)")
    print(f"Evaluation Catalog:\t{evaluation_catalog} ({Ngal_evaluation} sources)")
    if dict_read_from_file:
        print(f"Dictionaries:\t\t{dict_read_from_file}")
    else:
        print(f"{num_EAZY_as_dict} of 7 EAZY templates used as initialized dictionaries")    
    print('')
    print(f"Ndict = {Ndict}")
    print(f"Add constant dictionary: {add_constant}")
    if fix_dicts > 0:
        print(f"Fix {fix_dicts} dictionaries from the end of list")
    print(f"Centering: {centering}")    
    print('')
    print(f"Fix z in training: {fix_z}")
    if use_DESI_flag:
        print(f"Ncalibrators = {Ncalibrators} (DESI Flags)")
    elif Ncalibrators>0:
        print(f"Ncalibrators = {Ncalibrators} (SNR > {calibrator_SNR51})")
    # print(f"Convolving filters: 1st stage:{convolve_filter}, 2nd stage:{last_stage_convolve_filter}, fitting:{fitting_convolve_filter}")
    if algorithm == 0:
        print(f"Learning rates = {learning_rate0}/{learning_rate_cali}")
    else:
        if larslasso:
            print(fr"LARSlasso: {larslasso} (alpha={lars_alpha} + {LARSlasso_alpha_sigma}sigma, Positive: {lars_positive})")
        else:
            print(f"LARSlasso: {larslasso}")
        print(f"Remove old AB info: {remove_old_ab_info}")

    if testing_zgrid:
        print('Testing Zgrid: On')

    # Initialize A and B matrices for dictionary updates
    A = np.zeros((D_rest.shape[0], D_rest.shape[0]))
    B = np.zeros((len(lamb_rest), D_rest.shape[0]))

    if remove_old_ab_info: # save previous A and B for later removal
        A_history = np.zeros((Ngal, D_rest.shape[0], D_rest.shape[0]))
        B_history = np.zeros((Ngal, len(lamb_rest), D_rest.shape[0]))

    # Training iterations
    for i_iter in range(Niterations):
        print(f"\n{i_iter+1} of {Niterations} iterations")

        # pick out calibration galaxies
        idx_other = np.setdiff1d(idx_original, idx_cali)
        np.random.shuffle(idx_cali)
        np.random.shuffle(idx_other)
        idx_shuffle = np.hstack((idx_cali, idx_other))

        for i_gal0 in range(Ngal): 
            i_gal = idx_shuffle[i_gal0]
            # update with number of galaxies processed
            print(f"\r\t{i_gal0+1}/{Ngal} spectra", end="")
            # if this is a calibrator galaxy
            if i_gal in idx_cali or fix_z:
                # use the known redshift
                zinput = ztrue[i_gal]
            else:
                # otherwise perform a best-fit for the redshift
                zinput = False

            # fit this spectrum and obtain the redshift
            z,zlow,zhigh,coefs,model,_,_ = fit_zgrid(lamb_obs, spec_obs[i_gal], err_obs[i_gal], lamb_D=lamb_rest, D_rest=D_rest, zinput=zinput, 
                                                                      zgrid=zgrid, filter_info=filter_info, **train_fit_kws)

            # update the spectral dictionary using the residuals between the model and data
            residual = spec_obs[i_gal] - model
            j_update = np.where((lamb_rest > np.min(lamb_obs)/(1+z)) & (lamb_rest < np.max(lamb_obs)/(1+z)))[0]     # Find overlap between this spectra in rest-frame and dictionary
            interpolated_residual = np.interp(lamb_rest[j_update], lamb_obs/(1+z), residual)                          # interpolate the residual to these values
            interpolated_spec_obs = np.interp(lamb_rest[j_update], lamb_obs/(1+z), spec_obs[i_gal])
            model_rest = (D_rest).T @ coefs                                             # construct the model from best fit coefficients and dictionaries
            model_rest[j_update] = interpolated_spec_obs                                                            # replace the overlapped part with interpolated observed spectra 
                                                                                                                      # this will be considered the observed spectra for update purpose; outside overlap range just use model (no update)
            # TESTING
            # error_rest = np.ones_like(model_rest)
            # error_rest = np.interp(lamb_rest, lamb_obs/(1+z), err_obs[i_gal])
            # residue_factor = 1/error_rest * min(error_rest)

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
                    # B_history[i_gal] = model_rest[:,None] @ coefs[:,None].T * residue_factor * AB_factor
                    B_history[i_gal] = model_rest[:,None] @ coefs[:,None].T
                    # B_history[i_gal] = model_rest[:,None] @ coefs[:,None].T * AB_factor
                    A += A_history[i_gal]
                    B += B_history[i_gal]
                else:
                    A += coefs[:,None] @ coefs[:,None].T
                    # A += coefs[:,None] @ coefs[:,None].T * AB_factor
                    B += model_rest.reshape((len(model_rest),1)) @ coefs[:,None].T
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
                            # uj = 1/diag_Aj * (B[:,j] - ((D_rest*residue_factor).T @ A[j])) + D_rest[j]
                            uj_norm = np.linalg.norm(uj)
                            D_rest[j] = uj/max(1, uj_norm)
                            score = np.linalg.norm(D_rest[j]-Dj_old)/np.linalg.norm(Dj_old)
                            abi += 1
                        else:
                            break
                        

    np.savez_compressed(output_dirname+'trained_templates.npz',lamb_rest=lamb_rest, D_rest=D_rest)

    # Make template plots and multiplots
    makefigs = fx.diagnostic_plots(output_dirname=output_dirname)
    makefigs.template_plots(lamb_rest=lamb_rest, D_rest=D_rest[:Ndict], D_rest_initial=D_rest_initial[:Ndict])


    # pre-redshift D_rest into all redshift in zgrid
    D_allz = fx.apply_redshift_all(D_rest, zgrid=zgrid, lamb_in=lamb_rest, lamb_out=lamb_obs, 
                                    filter_info=filter_info, conv=fitting_convolve_filter)
    D_allz_initial = fx.apply_redshift_all(D_rest_initial, zgrid=zgrid, lamb_in=lamb_rest, lamb_out=lamb_obs, 
                                    filter_info=filter_info, conv=fitting_convolve_filter)

    # Fitting training catalog if set to True
    if fit_training_catalog:
        # fit all galaxies with final template
        print('\nTraining Catalog Redshift Estimation')
        zbest_trained = np.zeros(Ngal)
        zlow_trained = np.zeros(Ngal)
        zhigh_trained = np.zeros(Ngal)
        coefs_trained = np.zeros((Ngal, D_rest.shape[0]))


        for i in range(Ngal):
            # update with number of galaxies processed
            print(f"\r\t{i+1}/{Ngal} spectra", end="")
            # fit this spectrum and obtain the redshift
            zinput = False
            z,zlow,zhigh,coefs,model,_,_ = fit_zgrid(lamb_obs, spec_obs[i], err_obs[i], lamb_rest, D_rest=D_rest, D_allz=D_allz, zinput=zinput, 
                                                    zgrid=zgrid, filter_info=filter_info, **validation_fit_kws)
            # store the redshift
            zbest_trained[i] = z
            zlow_trained[i] = zlow
            zhigh_trained[i] = zhigh
            coefs_trained[i] = coefs

        # for comparison, fit again with original template
        zbest_initial = np.zeros(Ngal)
        print('\nTraining Catalog Untrained Redshift Estimation')
        for i in range(Ngal):
            print(f"\r\t{i+1}/{Ngal} spectra", end="")
            # fit this spectrum and obtain the redshift
            zinput = False
            z_bi,zlow_bi,zhigh_bi,coefs_bi,model_bi,_,_ = fit_zgrid(lamb_obs, spec_obs[i], err_obs[i], lamb_rest, D_rest=D_rest_initial,
                                                D_allz=D_allz_initial, zinput=zinput, zgrid=zgrid, filter_info=filter_info, **validation_fit_kws)
            zbest_initial[i] = z_bi

        makefigs.zp_zs_plots(ztrue=ztrue, zbest_initial=zbest_initial, zbest_trained=zbest_trained, zmin=zmin, zmax=zmax, catalog='training')



    # Read evaluation catalog for final fitting
    cat = fx.Catalog(pathfile=evaluation_catalog, Ndat=Ndat_evaluation, centering=centering)
    lamb_obs = cat.lamb_obs
    ztrue = cat.ztrue
    spec_obs = cat.spec_obs
    spec_obs_original = cat.spec_obs_original
    err_obs = cat.err_obs
    desi_flag = cat.desi_flag
    snr = cat.snr_i

    Ngal = len(ztrue)

    print('\nFitting Catalog Redshift Estimation')
    zbest_trained = np.zeros(Ngal)
    zlow_trained = np.zeros(Ngal)
    zhigh_trained = np.zeros(Ngal)
    coefs_trained = np.zeros((Ngal, D_rest.shape[0]))


    for i in range(Ngal):
        print(f"\r\t{i+1}/{Ngal} spectra", end="")

        zinput = False
        z,zlow,zhigh,coefs,model,_,_ = fit_zgrid(lamb_obs, spec_obs[i], err_obs[i], lamb_rest, D_rest=D_rest, D_allz=D_allz, zinput=zinput, 
                                                    zgrid=zgrid, filter_info=filter_info, **validation_fit_kws)
        # store the redshift
        zbest_trained[i] = z
        zlow_trained[i] = zlow
        zhigh_trained[i] = zhigh
        coefs_trained[i] = coefs

    # for comparison, fit again with original template
    zbest_initial = np.zeros(Ngal)
    zlow_initial = np.zeros(Ngal)
    zhigh_initial = np.zeros(Ngal)
    print('\nFitting Catalog Untrained Redshift Estimation')

    for i in range(Ngal):
        print(f"\r\t{i+1}/{Ngal} spectra", end="")

        zinput = False
        z_bi,zlow_bi,zhigh_bi,coefs_bi,model_bi,_,_ = fit_zgrid(lamb_obs, spec_obs[i], err_obs[i], lamb_rest, D_rest=D_rest_initial,
                                                D_allz=D_allz_initial, zinput=zinput, zgrid=zgrid, filter_info=filter_info, **validation_fit_kws)
        # store the redshift
        zbest_initial[i] = z_bi
        zlow_initial[i] = zlow_bi
        zhigh_initial[i] = zhigh_bi

    makefigs.zp_zs_plots(ztrue=ztrue, zbest_initial=zbest_initial, zbest_trained=zbest_trained, zmin=zmin, zmax=zmax, catalog='fitting')

    makefigs.uncertainty_binplots(zs=ztrue, zp=zbest_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zbest_initial, zl0=zlow_initial, zh0=zhigh_initial)
    makefigs.hexbin_binplot(zs=ztrue, zp=zbest_trained, zl=zlow_trained, zh=zhigh_trained, \
                                   zp0=zbest_initial, zl0=zlow_initial, zh0=zhigh_initial)

    # save estimated redshifts
    np.savez_compressed(output_dirname+'estimated_redshifts.npz', ztrue=ztrue, zest=zbest_trained, zlow=zlow_trained, zhigh=zhigh_trained, 
             zest_initial=zbest_initial, zlow_initial=zlow_initial, zhigh_initial=zhigh_initial, coefs=coefs_trained, idx_cali=idx_cali)

    # Sparsity plot
    if larslasso:
        makefigs.sparsity_report(coefs_trained=coefs_trained)



    makefigs.example_seds(cat=cat, lamb_rest=lamb_rest, D_rest=D_rest, D_rest_initial=D_rest_initial, \
                          zgrid=zgrid, filter_info=filter_info, validation_fit_kws=validation_fit_kws)

    # fit EAZY with trained dictionaries
    # makefigs.fit_eazy_plots(lamb_rest, D_rest, templates_EAZY)


    print('\nElapsed Time = '+str(time.time()-tic)+' seconds')


