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
# from numba import jit, njit     # use numba to accelerate key functions in the algorithm (by roughly a factor of 3!)
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
error_method = config['Catalog']['error_method']
Ncalibrators = config['Catalog']['Ncalibrators']
calibrator_SNR51 = config['Catalog']['calibrator_SNR']
f_lambda_mode = config['Catalog']['f_lambda_mode']    # fitting in f_lambda or f_nu
SNR = config['Catalog']['SNR']
add_fluctuations = config['Catalog']['add_fluctuations']     # Add random fluctuations to fluxes based on error columns as gaussian
flux_fluctuation_scaling = config['Catalog']['flux_fluctuation_scaling']

# Dictionary input configurations
dict_read_from_file = config['Dictionary']['read_from_file']
add_constant = config['Dictionary']['add_constant']
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
AB_update_tolerance = config['Algorithm']['AB_update_tolerance']
max_AB_loops = config['Algorithm']['max_update_loops']            # if algorithm = 1, number of loops to run updates on dictionaries
replace_old_ab_info = config['Algorithm']['replace_old_ab_info']
NMF = config['Algorithm']['NMF']
NMF_tolerance = config['Algorithm']['NMF_tolerance']
NMF_cutoff = config['Algorithm']['NMF_cutoff']
learning_rate0 = config['Algorithm']['learning_rate0']
learning_rate_cali = config['Algorithm']['learning_rate_cali']
lassolars = config['Algorithm']['LassoLars']    # TESTING
lars_alpha = config['Algorithm']['Lars_alpha']
lars_positive = config['Algorithm']['Lars_positive']

# Fitting configurations
probline = config['Fitting']['probline']
fit_training_catalog = config['Fitting']['fit_training_catalog']
prov_z_Niterations = config['Fitting']['prov_z_Niterations']
prov_z_cold_restart = config['Fitting']['prov_z_cold_restart']
convolve_filters = config['Fitting']['convolve_filters']
convolve_filter = convolve_filters[0] # choose to convolve templates with filter or not in the first stage of optimized grid search
last_stage_convolve_filter = convolve_filters[1]   # whether to colvolve with filters in the last stage of grid search 
fitting_convolve_filter = convolve_filters[2] # Whether to convolve with filters in the end when fitting final redshifts

# Zgrid configurations
z_fitting_max = config['Zgrid']['z_fitting_max']
zgrid_seps = config['Zgrid']['zgrid_separation']                # optimized zgrid setting and initialization
zgrid_seps.append(z_fitting_max)
zgrid_seps = np.array(zgrid_seps)
zgrid_stepsizes = np.array(config['Zgrid']['zgrid_stepsizes'])  # This needs to be shorter than zgrid_seps by 1 element
zgrid_searchsize = max(max(zgrid_stepsizes), config['Zgrid']['min_zgrid_searchsize'])  # search grid size to the left and right should be at least 0.02 but equal to the largest step size
zgrid_errsearchsize = config['Zgrid']['zgrid_errsearchsize']
zgrid = fx.generate_zgrid(zgrid_seps, zgrid_stepsizes, z_fitting_max)

# Directory locations
eazy_templates_location = config['Directory_locations']['eazy_templates_location']
filter_location = config['Directory_locations']['filter_location']
output_dirname = config['Directory_locations']['OUTPUT']

# TESTING
if lassolars:
    fit_spectrum = fx.fit_spectrum1
else:
    fit_spectrum = fx.fit_spectrum



# Read input file and get all necessary information
ztrue, lamb_obs, spec_obs, spec_obs_original, err_obs = fx.read_file(training_catalog, Ndat=Ndat, 
                                    error_method=error_method, SNR=SNR, f_lambda_mode=f_lambda_mode,  
                                    add_fluctuations=add_fluctuations, flux_fluctuation_scaling=flux_fluctuation_scaling)

Ngal = len(ztrue)
if Ngal < Ndat:
    Ndat = Ngal

# 51th channel SNR require for calibration galaxies
snr_obs = spec_obs/err_obs
h_cali = snr_obs[:,51] > calibrator_SNR51
idx_original = np.arange(Ngal)  # index corresponding to read input
idx_cali = idx_original[h_cali][:Ncalibrators] # index for calibration galaxies within read input



# Make sure directory strings have "/" at the end; if output directory doesn't exist, create one
if len(output_dirname) > 0 and output_dirname[-1] != '/':
    output_dirname = output_dirname + '/'
output_dir = Path(output_dirname)
if len(output_dirname) > 0 and not output_dir.is_dir():
    output_dir.mkdir()
if eazy_templates_location[-1] != '/':
    eazy_templates_location = eazy_templates_location + '/'
if filter_location[-1] != '/':
    filter_location = filter_location + '/'
filter_infos = fx.read_filters(filter_location)



# add a tiny number to z=0.0 to get around zinput issue at z=0.0
for i in range(len(ztrue)):
    if ztrue[i][0] == 0.0:
        # print('hhh')
        ztrue[i][0] = 0.00000001


# For saving files, if using error_method=0, change SNR to a string for filenames
# but if input file doesn't have error column, switch to method 1 and use input SNR
# Not really maintained nor used
if error_method == 0:
    try:
        np.load(training_catalog)['error']
        print(f"Error method: Original")
        SNR = 'dat'
    except:
        print(f"Error column not found, switch to Error method 1 (SNR={SNR})")
else:
    print(f"Error method: {error_method} (SNR={SNR})")


# preparing to initialize dictionaries with lamb_rest and EAZY templates
if dict_read_from_file:
    D_rest_input = np.load(dict_read_from_file)
    lamb_rest = D_rest_input['lamb_rest']
    D_rest = D_rest_input['D_rest']
    if Ndict >= len(D_rest):
        Ndict = len(D_rest)
    else:
        D_rest = D_rest[:Ndict]
    if f_lambda_mode and not D_rest_input['save_flambda']:
        D_rest = fx.fnu2flambda(lamb_rest*1e5, D_rest)
        D_rest = D_rest/np.linalg.norm(D_rest,axis=1)[:,None]
    elif not f_lambda_mode and D_rest_input['save_flambda']:
        D_rest = fx.flambda2fnu(lamb_rest*1e5, D_rest)
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
    print(f"Ndict = {Ndict}")
    print(f"Add constant dictionary: {add_constant}")
    print(f"Ncalibrators = {Ncalibrators} (SNR > {calibrator_SNR51})")
    print(f"Fix z in training: {fix_z}")
    # print(f"Non-negative Matrix Factorization: {NMF}")
    # print(f"Convolving filters: 1st stage:{convolve_filter}, 2nd stage:{last_stage_convolve_filter}, fitting:{fitting_convolve_filter}")
    if algorithm == 0:
        print(f"Learning rates = {learning_rate0}/{learning_rate_cali}")
    else:
        if lassolars:
            print(f"LassoLars: {lassolars} (alpha={lars_alpha}, Positive: {lars_positive})")
        print(f"Replace old AB info: {replace_old_ab_info}")
    if dict_read_from_file:
        print(f"Dictionaries:\t{dict_read_from_file}")
    else:
        print(f"{num_EAZY_as_dict} of 7 EAZY templates used as initialized dictionaries")


    # Initialize A and B matrices for dictionary updates
    A = np.zeros((D_rest.shape[0], D_rest.shape[0]))
    B = np.zeros((len(lamb_rest), D_rest.shape[0]))

    if replace_old_ab_info: # save previous A and B for later removal
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
            # if np.mod(i_gal0,100) == 0:
                # print(f"\r\t{i_gal0}/{Ngal} spectra", end="")
            print(f"\r\t{i_gal0}/{Ngal} spectra", end="")   # TEMP
            # print(i_gal)
            # if this is a calibrator galaxy
            # if i_gal in i_calibrator_galaxies or fix_z:
            if i_gal in idx_cali or fix_z:
                # use the known redshift
                zinput = ztrue[i_gal][0]
            else:
                # otherwise perform a best-fit for the redshift
                zinput = False

            # fit this spectrum and obtain the redshift
            z,zlow,zhigh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[i_gal,:], err_obs[i_gal,:], lamb_rest, D_rest, zgrid=zgrid, 
                                                    filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive, 
                                                    NMF=NMF, NMF_tolerance=NMF_tolerance, NMF_cutoff=NMF_cutoff,
                                                    zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                    zinput=zinput, conv_first=convolve_filter, conv_last=last_stage_convolve_filter, error=False)

            # update the spectral dictionary using the residuals between the model and data
            residual = spec_obs[i_gal,:] - model
            j_update = np.where((lamb_rest > np.min(lamb_obs)/(1+z)) & (lamb_rest < np.max(lamb_obs)/(1+z)))[0]     # Find overlap between this spectra in rest-frame and dictionary
            interpolated_residual = np.interp(lamb_rest[j_update], lamb_obs/(1+z), residual)                          # interpolate the residual to these values
            interpolated_spec_obs = np.interp(lamb_rest[j_update], lamb_obs/(1+z), spec_obs[i_gal,:])
            model_rest = ((D_rest).T @ params).reshape(len(lamb_rest))                                              # construct the model from best fit coefficients and dictionaries
            model_rest[j_update] = interpolated_spec_obs                                                            # replace the overlapped part with interpolated observed spectra 
                                                                                                                    # this will be considered the observed spectra for update purpose; outside overlap range just use model (no update)

            # update each item in the dictionary (do not modify the DC offset term at the end)
            if algorithm == 0:  # pseudo-inverse vector method
                # set the learning rate
                learning_rate = learning_rate0
                # if this is a calibrator galaxy
                if i_gal in idx_cali:
                    # use a higher learning rate since we know the redshift is correct
                    learning_rate = learning_rate_cali
                for i in range(D_rest.shape[0]-add_constant):
                    update_factor = learning_rate * (params[i]/(np.sum(params**2)))
                    D_rest[i,j_update] = D_rest[i,j_update] + update_factor * interpolated_residual

            elif algorithm == 1:    # paper method with A and B matrices
                # update A and B
                if replace_old_ab_info:
                    A -= A_history[i_gal]
                    B -= B_history[i_gal]
                    A_history[i_gal] = params @ params.T
                    B_history[i_gal] = model_rest.reshape((len(model_rest),1)) @ params.T
                    A += A_history[i_gal]
                    B += B_history[i_gal]
                else:
                    A += params @ params.T
                    B += model_rest.reshape((len(model_rest),1)) @ params.T

                # iterative over A and B matrices until converge or max_AB_loop
                for j in range(D_rest.shape[0]-add_constant):
                    score = 1.0
                    Dj_old = D_rest[j].copy()
                    abi = 0
                    while score >= AB_update_tolerance and abi < max_AB_loops:
                        # print(np.diagonal(A))
                        diag_Aj = np.diagonal(A)[j]
                        if diag_Aj == 0:
                            diag_Aj = 1e-10 # Avoid NaN
                        uj = 1/diag_Aj * (B[:,j] - (D_rest.T @ A[j])) + D_rest[j]
                        uj_norm = np.linalg.norm(uj)
                        D_rest[j] = uj/max(1, uj_norm)
                        score = np.linalg.norm(D_rest[j]-Dj_old)/np.linalg.norm(Dj_old)
                        abi += 1

    # If requiring provisional z iteration, fit every training set with learned dictionaries, then used fitted redshift as "true" redshift to learn again
    if prov_z_Niterations > 0:
        print('\nProvisional Redshift Estimation with fixed fitted redshift:')
        zbest_provisional = np.zeros(Ngal)
        for i in range(Ngal):
            # update with number of galaxies processed
            # if np.mod(i,100)==0:
                # print('    '+str(i)+' of '+str(Ngal)+' spectra')
                # print(f"\r\t{i}/{Ngal} spectra", end="")
            print(f"\r\t{i}/{Ngal} spectra", end="")    # TEMP
            # fit this spectrum and obtain the redshift
            z,zlow,zhigh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[i,:], err_obs[i,:], lamb_rest, D_rest, zgrid=zgrid, 
                                                        filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive, 
                                                        NMF=NMF, NMF_tolerance=NMF_tolerance, NMF_cutoff=NMF_cutoff,
                                                        zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                        zinput=False, conv_first=convolve_filter, conv_last=last_stage_convolve_filter)
            # store the redshift
            zbest_provisional[i] = z

        # iterate again
        if prov_z_cold_restart:
            D_rest = D_rest_initial.copy()  # Reset the dictionaries to initial if cold restart
            A = np.zeros((D_rest.shape[0], D_rest.shape[0]))
            B = np.zeros((len(lamb_rest), D_rest.shape[0]))
            if replace_old_ab_info:
                A_history = np.zeros((Ngal, D_rest.shape[0], D_rest.shape[0]))
                B_history = np.zeros((Ngal, len(lamb_rest), D_rest.shape[0]))

        for i_iter in range(prov_z_Niterations):
            idx_shuffle = idx_original.copy()
            np.random.shuffle(idx_shuffle)
            print(f"{i_iter+1} of {prov_z_Niterations} re-iterations")
            for i_gal0 in np.arange(Ngal):
                i_gal = idx_shuffle[i_gal0]
                zinput = zbest_provisional[i_gal]
                if fix_z:
                    zinput = ztrue[i_gal][0]

                # fit this spectrum and obtain the redshift
                z,zlow,zhigh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[i_gal,:], err_obs[i_gal,:], lamb_rest, D_rest, zgrid=zgrid, 
                                                        filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive, 
                                                        NMF=NMF, NMF_tolerance=NMF_tolerance, NMF_cutoff=NMF_cutoff,
                                                        zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                        zinput=zinput, conv_first=convolve_filter, conv_last=last_stage_convolve_filter)
                

                # update the spectral dictionary using the residuals between the model and data
                residual = spec_obs[i_gal,:] - model
                j_update = np.where((lamb_rest > np.min(lamb_obs)/(1+z)) & (lamb_rest < np.max(lamb_obs)/(1+z)))[0]
                # interpolate the residual to these values
                interpolated_residual = np.interp(lamb_rest[j_update], lamb_obs/(1+z), residual)
                interpolated_spec_obs = np.interp(lamb_rest[j_update], lamb_obs/(1+z), spec_obs[i_gal,:])
                model_rest = ((D_rest).T @ params).reshape(len(lamb_rest))
                model_rest[j_update] = interpolated_spec_obs

                # update each item in the dictionary (do not modify the DC offset term at the end)
                if algorithm == 0:  # pseudo-inverse vector method
                    # set the learning rate
                    learning_rate = learning_rate0
                    for i in range(D_rest.shape[0]-add_constant):
                        update_factor = learning_rate*(params[i]/(np.sum(params**2)))
                        D_rest[i,j_update] = D_rest[i,j_update] + update_factor * interpolated_residual
                elif algorithm == 1:    # paper method with A and B matrices
                    # update A and B
                    if replace_old_ab_info:
                        A -= A_history[i_gal]
                        B -= B_history[i_gal]
                        A_history[i_gal] = params @ params.T
                        B_history[i_gal] = model_rest.reshape((len(model_rest),1)) @ params.T
                        A += A_history[i_gal]
                        B += B_history[i_gal]
                    else:
                        A += params @ params.T
                        B += model_rest.reshape((len(model_rest),1)) @ params.T

                    # iterative over A and B matrices until converge or max_AB_loop
                    for j in range(D_rest.shape[0]-add_constant):
                        score = 1.0
                        Dj_old = D_rest[j].copy()
                        abi = 0
                        while score >= AB_update_tolerance and abi < max_AB_loops:
                            uj = 1/np.diagonal(A)[j] * (B[:,j] - (D_rest.T @ A[j])) + D_rest[j]
                            uj_norm = np.linalg.norm(uj)
                            D_rest[j] = uj/max(1, uj_norm)
                            score = np.linalg.norm(D_rest[j]-Dj_old)/np.linalg.norm(Dj_old)
                            abi += 1
        
    # plot results
    plt.ion()
    plt.figure(1)
    plt.clf()

    # Create template plots and save learned template
    plt.plot(lamb_rest,D_rest[:-1].transpose(),'-', alpha=0.8)
    plt.plot(np.nan,np.nan,'k-',label='Trained Template')
    plt.xlabel('Wavelength [um]')
    plt.ylabel('Flux [arb]')
    plt.title('Estimating Redshift Templates from Data')
    plt.legend()
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(output_dirname+'trained_template.png',dpi=600)
    np.savez_compressed(output_dirname+'trained_template.npz',lamb_rest=lamb_rest,D_rest=D_rest)
    # np.savez_compressed(output_dirname+'initial_template.npz',lamb_rest=lamb_rest,D_rest=D_rest_initial)

    templates_figsize = (12,10)
    tick_fontsize = 6

    fig, axs = plt.subplots(Ndict, 2, figsize=templates_figsize, num=2)
    for i in range(Ndict):
        axs[i,0].plot(lamb_rest, D_rest_initial[i], alpha=0.8, linewidth=1)
        axs[i,0].plot(lamb_rest, D_rest[i], alpha=0.8, linewidth=1)
        axs[i,1].plot(lamb_rest, D_rest[i]-D_rest_initial[i], linewidth=1)
        axs[i,0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[i,1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    axs[0,0].set_title('Initial/Trained Dictionaries')
    axs[0,1].set_title('Trained-Initial')
    fig.supxlabel('Wavelength [um]')
    # fig.supylabel()
    fig.tight_layout()
    plt.savefig(output_dirname+f'trained_template_multiplot.png',dpi=600)

    # Fitting training catalog if set to True
    if fit_training_catalog:
        # fit all galaxies with final template
        print('\nTraining Catalog Redshift Estimation')
        zbest_trained = np.zeros(Ngal)
        zlow_trained = np.zeros(Ngal)
        zhigh_trained = np.zeros(Ngal)
        params_trained = np.zeros((Ngal, D_rest.shape[0], 1))

        for i in range(Ngal):
            # update with number of galaxies processed
            # if np.mod(i,100)==0:
                # print(f"\r\t{i}/{Ngal} spectra", end="")
            print(f"\r\t{i}/{Ngal} spectra", end="")    # TEMP
            # fit this spectrum and obtain the redshift
            zinput = False
            z,zlow,zhigh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[i,:], err_obs[i,:], lamb_rest, D_rest, zgrid=zgrid, 
                                                        filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive,
                                                        zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                        zinput=zinput, conv_first=convolve_filter, conv_last=fitting_convolve_filter, error=True)
            # store the redshift
            zbest_trained[i] = z
            zlow_trained[i] = zlow
            zhigh_trained[i] = zhigh
            params_trained[i] = params

        # for comparison, fit again with original template
        # turn off this part
        zbest_initial = np.zeros(Ngal)
        print('\nTraining Catalog Untrained Redshift Estimation')
        for i in range(Ngal):
            # if np.mod(i,100)==0:
                # print(f"\r\t{i}/{Ngal} spectra", end="")
            print(f"\r\t{i}/{Ngal} spectra", end="")    # TEMP
            # fit this spectrum and obtain the redshift
            zinput = False
            z_bi,zlow_bi,zhigh_bi,params_bi,model_bi,ztrials_bi,residues_bi = fit_spectrum(lamb_obs, spec_obs[i,:], err_obs[i,:], lamb_rest, D_rest_initial, zgrid=zgrid, 
                                                        filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive,
                                                        zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                        zinput=zinput, conv_first=convolve_filter, conv_last=fitting_convolve_filter, error=True)
            zbest_initial[i] = z_bi

        # correct dimsionality of ztrue
        ztrue = ztrue.flatten()

        # find % of catastrophic error and accuracy
        dz_initial = zbest_initial - ztrue
        eta_initial = np.mean(np.abs(dz_initial/(1+ztrue)) > 0.15) * 100
        dz_trained = zbest_trained - ztrue
        eta_trained = np.mean(np.abs(dz_trained/(1+ztrue)) > 0.15) * 100

        # NMAD
        nmad_initial = 1.48 * np.median(np.abs((dz_initial - np.median(dz_initial))/(1+ztrue)))
        nmad_trained = 1.48 * np.median(np.abs((dz_trained - np.median(dz_trained))/(1+ztrue)))

        # plot redshift reconstruction
        zpzs_figsize = (10,10)
        fig1, axs1 = plt.subplots(2,1, figsize=zpzs_figsize, num=3, gridspec_kw={'height_ratios': [3,1]})
        lim_offset = 0.05
        axs1[0].set_xlim(0-lim_offset, z_fitting_max+lim_offset)
        axs1[0].set_ylim(0-lim_offset, z_fitting_max+lim_offset)
        axs1[1].set_ylim(-0.25,0.25)
        axs1[1].set_xlim(0-lim_offset, z_fitting_max+lim_offset)
        axs1[0].grid()
        axs1[1].grid()

        labelfontsize = 18
        tickfontsize = 12
        legendfontsize = 14
        m0 = 'o'
        m1 = 'o'
        m0size = 4
        m1size = 4
        markeredgewidth = 0.2
        # m0edgec = 'tab:blue'
        # m1edgec = 'tab:orange'
        m0edgec = 'k'
        m1edgec = 'k'

        axs1[0].set_ylabel('Estimated Redshift', fontsize=labelfontsize)
        axs1[1].set_xlabel('True Redshift', fontsize=labelfontsize)
        axs1[1].set_ylabel(r'$\Delta z/(1+z_{True})$', fontsize=labelfontsize)
        axs1[0].plot(ztrue, zbest_initial, m0, markersize=m0size, markeredgecolor=m0edgec, markeredgewidth=markeredgewidth, alpha=0.65,
                    label=f'Initial, $\eta={eta_initial:.3f}$%, $\sigma_{{NMAD}}={100*nmad_initial:.3f}$%')
        axs1[0].plot(ztrue, zbest_trained, m1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth, alpha=0.8,
                    label=f'Trained, $\eta={eta_trained:.3f}$%, $\sigma_{{NMAD}}={100*nmad_trained:.3f}$%')
        axs1[0].plot([0-lim_offset,z_fitting_max+lim_offset],[0-lim_offset,z_fitting_max+lim_offset],'-',alpha=0.8, color='g', linewidth=2)
        axs1[1].plot(ztrue, (zbest_initial-ztrue)/(1+ztrue), m0, markersize=m0size, markeredgecolor=m0edgec, markeredgewidth=markeredgewidth, alpha=0.65)
        axs1[1].plot(ztrue, (zbest_trained-ztrue)/(1+ztrue), m1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth, alpha=0.8)
        axs1[1].plot([0-lim_offset,z_fitting_max+lim_offset],[0,0],'-',alpha=0.8, color='g', linewidth=2)
        axs1[0].tick_params(axis='both', which='major', labelsize=tickfontsize)
        axs1[1].tick_params(axis='both', which='major', labelsize=tickfontsize)
        axs1[0].legend(fontsize=legendfontsize, framealpha=0.9, loc='upper left')
        # axs[1].legend(fontsize=20, loc='lower right')
        fig1.suptitle('Training Catalog Redshift Estimation')
        fig1.tight_layout()
        plt.savefig(output_dirname+f'redshift_estimation_performance_training_catalog.png',dpi=600)
        # plt.savefig(output_dirname+'redshift_estimation_performance.png',dpi=600)

    # Fitting on fitting_catalog
    # Read the fitting catalog first
    ztrue, lamb_obs, spec_obs, spec_obs_original, err_obs = fx.read_file(evaluation_catalog, Ndat=Ndat_evaluation,
                                    error_method=error_method, SNR=SNR, f_lambda_mode=f_lambda_mode, 
                                    add_fluctuations=add_fluctuations, flux_fluctuation_scaling=flux_fluctuation_scaling)

    Ngal = len(ztrue)

    print('\nFitting Catalog Redshift Estimation')
    zbest_trained = np.zeros(Ngal)
    zlow_trained = np.zeros(Ngal)
    zhigh_trained = np.zeros(Ngal)
    params_trained = np.zeros((Ngal, D_rest.shape[0], 1))

    for i in range(Ngal):
        # update with number of galaxies processed
        # if np.mod(i,100)==0:
            # print(f"\r\t{i}/{Ngal} spectra", end="")
        print(f"\r\t{i}/{Ngal} spectra", end="")    # TEMP

        # fit this spectrum and obtain the redshift
        zinput = False
        z,zlow,zhigh,params,model,ztrials,residues = fit_spectrum(lamb_obs, spec_obs[i,:], err_obs[i,:], lamb_rest, D_rest, zgrid=zgrid, 
                                                    filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive, 
                                                    NMF=NMF, NMF_tolerance=NMF_tolerance, NMF_cutoff=NMF_cutoff,
                                                    zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                    zinput=zinput, conv_first=convolve_filter, conv_last=fitting_convolve_filter, error=True)
        # store the redshift
        zbest_trained[i] = z
        zlow_trained[i] = zlow
        zhigh_trained[i] = zhigh
        params_trained[i] = params

    # for comparison, fit again with original template
    # turn off this part
    zbest_initial = np.zeros(Ngal)
    zlow_initial = np.zeros(Ngal)
    zhigh_initial = np.zeros(Ngal)
    print('\nFitting Catalog Untrained Redshift Estimation')

    for i in range(Ngal):
        # update with number of galaxies processed
        # if np.mod(i,100)==0:
            # print(f"\r\t{i}/{Ngal} spectra", end="")
        print(f"\r\t{i}/{Ngal} spectra", end="")    # TEMP
        # fit this spectrum and obtain the redshift
        zinput = False
        z_bi,zlow_bi,zhigh_bi,params_bi,model_bi,ztrials_bi,residues_bi = fit_spectrum(lamb_obs, spec_obs[i,:], err_obs[i,:], lamb_rest, D_rest_initial, zgrid=zgrid, 
                                                    filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive, 
                                                    NMF=NMF, NMF_tolerance=NMF_tolerance, NMF_cutoff=NMF_cutoff,
                                                    zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                    zinput=zinput, conv_first=convolve_filter, conv_last=fitting_convolve_filter, error=True)
        # store the redshift
        zbest_initial[i] = z_bi
        zlow_initial[i] = zlow_bi
        zhigh_initial[i] = zhigh_bi


    # correct dimsionality of ztrue
    ztrue = ztrue.flatten()

    # find % of catastrophic error and accuracy
    dz_initial = zbest_initial - ztrue
    eta_initial = np.mean(np.abs(dz_initial/(1+ztrue)) > 0.15) * 100
    dz_trained = zbest_trained - ztrue
    eta_trained = np.mean(np.abs(dz_trained/(1+ztrue)) > 0.15) * 100

    # NMAD
    nmad_initial = 1.48 * np.median(np.abs((dz_initial - np.median(dz_initial))/(1+ztrue)))
    nmad_trained = 1.48 * np.median(np.abs((dz_trained - np.median(dz_trained))/(1+ztrue)))


    # plot redshift reconstruction
    zpzs_figsize = (10,10)
    fig2, axs2 = plt.subplots(2,1, figsize=zpzs_figsize, num=4, gridspec_kw={'height_ratios': [3,1]})
    lim_offset = 0.05
    axs2[0].set_xlim(0-lim_offset,z_fitting_max+lim_offset)
    axs2[0].set_ylim(0-lim_offset,z_fitting_max+lim_offset)
    axs2[1].set_ylim(-0.25,0.25)
    axs2[1].set_xlim(0-lim_offset,z_fitting_max+lim_offset)
    axs2[0].grid()
    axs2[1].grid()

    labelfontsize = 18
    tickfontsize = 12
    legendfontsize = 14
    m0 = 'o'
    m1 = 'o'
    m0size = 4
    m1size = 4
    markeredgewidth = 0.2
    # m0edgec = 'tab:blue'
    # m1edgec = 'tab:orange'
    m0edgec = 'k'
    m1edgec = 'k'

    axs2[0].set_ylabel('Estimated Redshift', fontsize=labelfontsize)
    axs2[1].set_xlabel('True Redshift', fontsize=labelfontsize)
    axs2[1].set_ylabel(r'$\Delta z/(1+z_{True})$', fontsize=labelfontsize)
    axs2[0].plot(ztrue, zbest_initial, m0, markersize=m0size, markeredgecolor=m0edgec, markeredgewidth=markeredgewidth, alpha=0.65,
                label=f'Initial, $\eta={eta_initial:.3f}$%, $\sigma_{{NMAD}}={100*nmad_initial:.3f}$%')
    axs2[0].plot(ztrue, zbest_trained, m1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth, alpha=0.8,
                label=f'Trained, $\eta={eta_trained:.3f}$%, $\sigma_{{NMAD}}={100*nmad_trained:.3f}$%')
    axs2[0].plot([0-lim_offset,z_fitting_max+lim_offset],[0-lim_offset,z_fitting_max+lim_offset],'-',alpha=0.8, color='g', linewidth=2)
    axs2[1].plot(ztrue, (zbest_initial-ztrue)/(1+ztrue), m0, markersize=m0size, markeredgecolor=m0edgec, markeredgewidth=markeredgewidth, alpha=0.65)
    axs2[1].plot(ztrue, (zbest_trained-ztrue)/(1+ztrue), m1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth, alpha=0.8)
    axs2[1].plot([0-lim_offset,z_fitting_max+lim_offset],[0,0],'-',alpha=0.8, color='g', linewidth=2)
    axs2[0].tick_params(axis='both', which='major', labelsize=tickfontsize)
    axs2[1].tick_params(axis='both', which='major', labelsize=tickfontsize)
    axs2[0].legend(fontsize=legendfontsize, framealpha=0.9, loc='upper left')
    # axs[1].legend(fontsize=20, loc='lower right')
    fig2.suptitle('Fitting Catalog Redshift Estimation')
    fig2.tight_layout()
    plt.savefig(output_dirname+f'redshift_estimation_performance_fitting_catalog.png',dpi=600)


    # save estimated redshifts
    np.savez_compressed(output_dirname+'estimated_redshifts.npz', ztrue=ztrue, zest=zbest_trained, zlow=zlow_trained, zhigh=zhigh_trained, 
             zest_initial=zbest_initial, zlow_initial=zlow_initial, zhigh_initial=zhigh_initial, params=params_trained, idx_cali=idx_cali)


    # for comparison, fit a single spectrum with the initial and trained template
    # select galaxy
    i = 67
    # i = 15
    # refit with the initial dictionary
    zbest_initial_ex,zlow_initial,zhigh_initial,params_initial,best_model_initial,ztrials_initial,residues_initial = fit_spectrum(lamb_obs, spec_obs[i,:], err_obs[i,:], lamb_rest, D_rest_initial, zgrid=zgrid, 
                                                    filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive, 
                                                    NMF=NMF, NMF_tolerance=NMF_tolerance, NMF_cutoff=NMF_cutoff,
                                                    zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                    zinput=False, conv_first=convolve_filter, conv_last=fitting_convolve_filter)
    # refit with the trained dictionary
    zbest_trained_ex,zlow_trained,zhigh_trained,params_trained,best_model,ztrials_best,residues_best = fit_spectrum(lamb_obs, spec_obs[i,:], err_obs[i,:], lamb_rest, D_rest, zgrid=zgrid, 
                                                    filter_infos=filter_infos, alpha=lars_alpha, lars_positive=lars_positive, 
                                                    NMF=NMF, NMF_tolerance=NMF_tolerance, NMF_cutoff=NMF_cutoff,
                                                    zgrid_searchsize=zgrid_searchsize, zgrid_errsearchsize=zgrid_errsearchsize, z_fitting_max=z_fitting_max, probline=probline,
                                                    zinput=False, conv_first=convolve_filter, conv_last=fitting_convolve_filter)
        
    # plot single fitted spectrum
    plt.figure(num=6)
    plt.clf()
    plt.errorbar(lamb_obs, spec_obs[i,:], err_obs[i,:], fmt='o', markersize=3, markerfacecolor='none', markeredgecolor='cadetblue', markeredgewidth=0.5, capsize=2, elinewidth=0.5, alpha=0.6, label="Photometry")
    plt.plot(lamb_obs, spec_obs_original[i,:], '.', color='green', markersize=3, alpha=0.7, label="Ground Truth")
    plt.plot(lamb_obs, best_model_initial, '-', linewidth=1.5, alpha=0.6, color='k', label=fr"Initial Template, $z_{{est}}$={zbest_initial_ex:.4}")
    plt.plot(lamb_obs, best_model, linewidth=1, color='salmon', alpha=0.9, label=fr"Trained Template, $z_{{est}}$={zbest_trained_ex:.4}")
    plt.xlabel('Observed Wavelength [$\mu$m]', fontsize=10)
    plt.ylabel('Flux [mJy]', fontsize=10)
    plt.title(fr"Idx={i}, $z_{{true}}$={ztrue[i]:.5f}", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(output_dirname+'single_spectrum_fitting.png',dpi=600)


    # compare learned template with input template
    # create main wavelength array
    lamb_um = np.arange(0.3,4.8,lamb_rest_resolution)
    h_lamb_um = (lamb_rest>=0.29999) & (lamb_rest<4.8)
    templates_EAZY = templates_EAZY[:,h_lamb_um]

    D_rest_interpolated_list = []
    for i in range(D_rest.shape[0]):
        D_rest_interpolated_list.append(np.interp(lamb_um,lamb_rest,D_rest[i,:]))
    D_rest_interpolated = np.vstack(tuple(D_rest_interpolated_list))

    plt.figure(num=5, figsize=templates_figsize)
    for i in range(7):
        # reconstruct this ground-truth template item with the learned template
        # params =  inv(D*D')*D*s'
        params = np.matmul(np.matmul(np.linalg.inv(np.matmul(D_rest_interpolated,D_rest_interpolated.transpose())),D_rest_interpolated),templates_EAZY[i,:])

        # evaluate model
        this_model = np.zeros_like(lamb_um)
        for j in range(D_rest.shape[0]):
            this_model += params[j]*D_rest_interpolated[j,:]
        
        # make a plot
        plt.subplot(7,1,i+1)
        plt.plot(lamb_um,templates_EAZY[i,:],'.',label='Ground-Truth')
        plt.plot(lamb_um,this_model,label='Learned')
        plt.ylabel('T'+str(i))
        plt.grid('on')
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        ax=plt.gca()
        ax.yaxis.get_offset_text().set_size(tick_fontsize)

    plt.xlabel('[um]')
    plt.subplot(7,1,1)
    plt.title('Reconstructing Ground-Truth Template with Learned Template')
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dirname+'reconstructing_ground_truth_template.png',dpi=600)

    print('\nElapsed Time = '+str(time.time()-tic)+' seconds')

    # np.savez_compressed(output_dirname+'fluctuated_input_cat.npz', z=ztrue.reshape(Ngal,1), wavelengths=lamb_obs, spectra=spec_obs, error=err_obs, spectra_original=spec_obs_original)
    # np.savez(output_dirname+'min_chi2_dz.npz', chi2=chisqs, zhl=zhl, dz=delta_z, diff=model_diff, rescale_factor=rescale_factor, peak=peak)

