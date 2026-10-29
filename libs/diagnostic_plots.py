import numpy as np
import pandas as pd
from numba import jit, njit
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
import yaml
from pathlib import Path
from . import lib_dictlearn as libs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


def template_plots(lamb_rest, 
                   D_rest, 
                   D_rest_initial, 
                   lamb_obs=None, 
                   D_sys=None, 
                   D_sys_initial=None, 
                   savefig=False, 
                   output_dirname='', 
                   dpi=300):
    output_dir = Path(output_dirname)

    fig1, ax1 = plt.subplots(figsize=(8,4))
    if D_rest is not None:
        ax1.plot(lamb_rest, D_rest.T, '-', linewidth=1, alpha=0.8)
    if lamb_obs is not None and D_sys is not None:
        ax1.plot(lamb_obs, D_sys.T, '--', linewidth=1, alpha=0.8)
    ax1.set_xlabel(r"Wavelength [$\mu m$]")
    ax1.set_ylabel('Flux [arb]')
    ax1.set_title('Learned Templates')
    ax1.grid()
    ax1.set_xlim(0,5.)
    fig1.tight_layout()
    if savefig:
        plt.savefig(output_dir / 'trained_templates.png', dpi=dpi)
    # else:
        # plt.show()
    # self.fig_num += 1
    templates_figsize = (12, 10)
    tick_fontsize = 6

    if D_rest is not None:
        Ndict = D_rest.shape[0]
    else:
        Ndict = 0
    if lamb_obs is not None and D_sys is not None:
        Ndict_sys = D_sys.shape[0]
    else:
        Ndict_sys = 0
    if D_rest is not None:
        D_diff = D_rest - D_rest_initial
    if (Ndict+Ndict_sys) == 1:
        Nrows = 2
    else:
        Nrows = Ndict+Ndict_sys
    fig2, ax2 = plt.subplots(Nrows, 2, sharex=True, figsize=(templates_figsize))            
    for i in range(Ndict_sys, Ndict+Ndict_sys):
        ax2[i,0].plot(lamb_rest, D_rest_initial[i-Ndict_sys], alpha=0.8, linewidth=1)
        ax2[i,0].plot(lamb_rest, D_rest[i-Ndict_sys], alpha=0.8, linewidth=1)
        ax2[i,1].plot(lamb_rest, D_diff[i-Ndict_sys], linewidth=1)
        ax2[i,1].plot([-1,6], [0,0], color='k', linewidth=0.5, linestyle='--', alpha=0.8)
        ax2[i,0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax2[i,1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax2[i,0].set_ylabel(i, fontsize=tick_fontsize)
    for i in range(Ndict_sys):
        if D_sys_initial is not None:
            ax2[i, 0].plot(lamb_obs, D_sys_initial[i], '--', alpha=0.8, linewidth=1)
            ax2[i, 1].plot(lamb_obs, D_sys[i]-D_sys_initial[i], '--', linewidth=1)
        ax2[i, 0].plot(lamb_obs, D_sys[i], alpha=0.8, linewidth=1)



    ax2[0,1].set_ylim(-0.1, 0.1)  # TEMP
    # if (D_diff==0).all():
        # ax2[0,1].set_ylim(-0.05, 0.05)
    # else:
        # ax2[0,1].set_ylim(np.min(D_diff), np.max(D_diff))
    
    for i in range(1, Ndict):
        ax2[i, 1].sharey(ax2[0,1])
    ax2[0,0].set_xlim(0,5.)

    ax2[0,0].set_title('Initial/Trained Dictionaries')
    ax2[0,1].set_title('Trained-Initial')
    fig2.supxlabel(r"Wavelength [$\mu m$]")
    # fig.supylabel()
    fig2.tight_layout()
    if savefig:
        plt.savefig(output_dir / f'trained_template_multiplot.png', dpi=dpi)
    # else:
        # plt.show()

def zp_zs_plots(ztrue, z_trained, ztrue_pit=None,
                z_initial=None, zl=None, zh=None, sig_1pz_cutoff=0.2,
                zmin=0.0, zmax=3.0, catalog='fitting', 
                gridsize=150, gridsize2=(150,30), fit_initial=False,
                savefig=False, 
                output_dirname='', 
                dpi=300):
    output_dir = Path(output_dirname)
    
    zpzs_figsize = (12,7)
    lim_offset = 0.05
    bottom_ylim = 0.25

    labelfontsize = 12
    tickfontsize = 10
    legendfontsize = 12
    markercolor0 = 'tab:blue'
    markercolor1 = 'tab:orange'
    m0 = '.'
    m1 = '.'
    m0size = 4
    m1size = 4
    markeredgewidth0 = 0.0
    markeredgewidth1 = 0.0
    m0alpha = 0.1
    m1alpha = 0.15
    # m0edgec = 'tab:blue'
    # m1edgec = 'tab:orange'
    m0edgec = 'grey'
    m1edgec = 'k'
    # gridsize = 100
    bins = 'log'

    # qq_linestyle = '-'
    # qq_linewidth=1.5
    # qq_linecolor='tab:red'
    # qq_linealpha=0.8
    # qq_refstyle = '--'
    # qq_refwidth=1.0
    # qq_refcolor='tab:green'
    # qq_refalpha = 0.6

    nmad_f, eta_f = libs.nmad_eta(zs=ztrue, zp=z_trained)
    if zl is not None and zh is not None:
        sig = (zh-zl)/2
        sig_1pz = sig/(1+z_trained)
        sig_mask = sig_1pz < sig_1pz_cutoff
        nmad_f1, eta_f1 = libs.nmad_eta(zs=ztrue[sig_mask], zp=z_trained[sig_mask])

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=zpzs_figsize, gridspec_kw={'height_ratios': [3,1], 'width_ratios': [1,1.25]})
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

    if fit_initial:
        nmad_i, eta_i = libs.nmad_eta(zs=ztrue, zp=z_initial)
        ax[0,0].plot(ztrue, z_initial, m0, color=markercolor0, markersize=m0size, markeredgecolor=m0edgec, markeredgewidth=markeredgewidth0, alpha=m0alpha,
                    label=fr'Initial, $\eta={eta_i*100:.3f}$%, $\sigma_{{NMAD}}={nmad_i*100:.3f}$%')
        ax[1,0].plot(ztrue, (z_initial-ztrue)/(1+ztrue), m0, color=markercolor0, markersize=m0size, markeredgecolor=m0edgec, \
                    markeredgewidth=markeredgewidth0, alpha=m0alpha)
    ax[0,0].plot(ztrue, z_trained, m1, color=markercolor1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth1, alpha=m1alpha,
                label=fr'Trained, $\eta={eta_f*100:.3f}$%, $\sigma_{{NMAD}}={nmad_f*100:.3f}$%')
    if zl is not None and zh is not None:
        ax[0,0].plot(ztrue, z_trained, m1, color=markercolor1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth1, alpha=0,
                label=fr'$\sigma_{{z}}/(1+z)<0.2$: $\eta={eta_f1*100:.3f}$%, $\sigma_{{NMAD}}={nmad_f1*100:.3f}$%')    
    ax[1,0].plot(ztrue, (z_trained-ztrue)/(1+ztrue), m1, color=markercolor1, markersize=m1size, markeredgecolor=m1edgec, \
                    markeredgewidth=markeredgewidth1, alpha=m1alpha)

    hb1 = ax[0,1].hexbin(ztrue, z_trained, gridsize=gridsize, extent=[zmin, zmax, zmin, zmax], bins=bins)
    cb1 = fig.colorbar(hb1, ax=ax[0,1])
    hb2 = ax[1,1].hexbin(ztrue, (z_trained-ztrue)/(1+ztrue), gridsize=gridsize2, extent=[zmin, zmax, -bottom_ylim, bottom_ylim], bins=bins)
    cb2 = fig.colorbar(hb2, ax=ax[1,1])

    ax[0,0].plot([zmin-lim_offset, zmax+lim_offset], [zmin-lim_offset, zmax+lim_offset], '-', alpha=0.8, color='g', linewidth=1)
    ax[0,1].plot([zmin-lim_offset, zmax+lim_offset], [zmin-lim_offset, zmax+lim_offset], linewidth=0.7, color='salmon', alpha=0.8)
    ax[1,0].plot([zmin-lim_offset, zmax+lim_offset], [0,0],'-', alpha=0.8, color='g', linewidth=1)
    ax[1,1].plot([zmin-lim_offset, zmax+lim_offset], [0,0],'-', linewidth=0.7, color='salmon', alpha=0.8)

    ax[0,0].tick_params(axis='both', which='major', labelsize=tickfontsize)
    ax[1,0].tick_params(axis='both', which='major', labelsize=tickfontsize)
    ax[0,0].legend(fontsize=legendfontsize, framealpha=0.9, loc='upper left')
    # axs[1].legend(fontsize=20, loc='lower right')
    fig.suptitle(catalog)
    fig.tight_layout()
    if savefig:
        plt.savefig(output_dir / f'redshift_estimation_performance_{catalog}.png', dpi=dpi)
    # else:
        # plt.show()
    # self.fig_num += 1

def sparsity_report(coefs_trained, 
                    ynorm=None, 
                    max_feature=None, 
                    add_constant=False, 
                    catalog='', 
                    savefig=False, 
                    output_dirname='', 
                    dpi=300):
    output_dir = Path(output_dirname)
    
    Ngal = coefs_trained.shape[0]
    Ndict = coefs_trained.shape[1]

    figsize = (8,8)
    fig, ax = plt.subplots(3, 2, figsize=figsize)

    coefs = coefs_trained
    coefs[np.isnan(coefs)] = 0.0
    coefs_nonzero_nums_in_each_source = np.sum((coefs!=0), axis=1)
    coefs_nonzeros = np.zeros(Ndict+1, dtype=int)
    for i in range(Ndict+1):
        coefs_nonzeros[i] = np.sum(coefs_nonzero_nums_in_each_source==i)
    ax[0,0].bar(np.arange(Ndict+1), coefs_nonzeros/Ngal)
    ax[0,0].set_title('Number of coefficients')
    # ax[0,0].set_xlabel('# of coefficients')
    ax[0,0].set_ylabel('Frequency')

    coef_counts = np.sum((coefs!=0), axis=0)
    coef_sums = np.sum(coefs, axis=0)
    coef_pos_sums = np.sum(np.fabs(coefs), axis=0)
    bar10 = ax[1,0].bar(np.arange(Ndict), coef_counts/Ngal, color='teal')
    ax[1,0].set_title('Dictionary activation frequency')

    bar20_1 = ax[2,0].bar(np.arange(Ndict), coef_pos_sums, color='brown', alpha=0.6)
    bar20_2 = ax[2,0].bar(np.arange(Ndict), coef_sums, color='salmon', alpha=1.0)
    if add_constant:
        bar10[-1].set_alpha(0.35)
        bar20_1[-1].set_alpha(0.35)
        bar20_2[-1].set_alpha(0.35)
    if max_feature is not None:
        max_line = max_feature + 1
        ymin, ymax = ax[0].get_ylim()
        ax[0,0].vlines(max_line, ymin=ymin, ymax=ymax, linewidth=1, color='tab:red')
        ax[0,0].set_ylim(ymin, ymax)

    ax[2,0].set_title('Sum of coefficients for each dictionary')

    if ynorm is not None:
        cmap_name = 'inferno'
        cmap = plt.get_cmap(cmap_name, Ndict)
        norm = mpl.colors.BoundaryNorm(boundaries=np.arange(0.5, Ndict+1.5, 1), ncolors=Ndict)
        ticks = [i for i in range(Ndict+1)]
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax[0,1])
        cax = divider.append_axes("right", size="3%", pad=0.03)
        # nanmask = ynorm >= 0
        # ax[0,1].scatter(1/ynorm[nanmask], coefs_nonzero_nums_in_each_source[nanmask], s=5, alpha=0.1)
        ynorm[ynorm==0] = np.min(ynorm[ynorm!=0])
        for di in range(1, Ndict+1):
            nonmask = ~np.isnan(ynorm)
            mask = (coefs_nonzero_nums_in_each_source == di)
            im01 = ax[0,1].hist(1/ynorm[mask & nonmask], bins=100, alpha=0.1+(0.6/Ndict)*di, color=cmap(norm(di)), range=(0, np.max(1/ynorm[nonmask])))
            cb = plt.colorbar(sm, cax=cax, ticks=ticks)
            # fig.colorbar(im01, cax=cax, ticks=ticks, label='# of dicts')

    divider11 = make_axes_locatable(ax[1,1])
    cax11 = divider11.append_axes('right', size='5%', pad=0.03)

    coefs_nonzeros_each_coef = (coefs != 0) * 1
    num_cov = coefs_nonzeros_each_coef.T @ coefs_nonzeros_each_coef
    im11 = ax[1,1].imshow(num_cov/Ngal)
    cb11 = fig.colorbar(im11, cax=cax11)
    ax[1,1].set_title('Co-activation frequency')

    divider21 = make_axes_locatable(ax[2,1])
    cax21 = divider21.append_axes('right', size='5%', pad=0.03)

    coefs_dot_product = coefs.T @ coefs
    im21 = ax[2,1].imshow(coefs_dot_product)
    cb21 = fig.colorbar(im21, cax=cax21)
    ax[2,1].set_title('Co-activation dot product')


    fig.tight_layout()
    if savefig:
        plt.savefig(output_dir / f'sparsity_reports{catalog}.png', dpi=dpi)
    # self.fig_num += 1

def example_seds(cat, 
                 lamb_rest, 
                 D_rest, 
                 D_rest_initial, 
                 zgrid, 
                 validation_fit_kws, 
                 D_rest_alphas=None,
                 D_rest_alphas_ndict=None,
                 D_sys=None, 
                 D_sys_initial=None, 
                 idx=None, 
                 filters=None, 
                 ztype='zpeak',
                 savefig=False, 
                 output_dirname='', 
                 dpi=300):
    output_dir = Path(output_dirname)
    
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

    fig, ax = plt.subplots(nrows, ncols, sharex=True, figsize=figsize)

    for i in range(len(idx)):
        row = row_idx[i]
        col = col_idx[i]
        idx_i = idx[i]

        # refit with the initial dictionary
        spec_obs_i = np.ascontiguousarray(spec_obs[idx_i])
        err_obs_i = np.ascontiguousarray(err_obs[idx_i])
        fit_initial = libs.fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest=lamb_rest, 
                                     D_rest=D_rest_initial, D_sys=D_sys_initial, zinput=False, zgrid=zgrid, 
                                     D_rest_alphas=D_rest_alphas, D_rest_alphas_ndict=D_rest_alphas_ndict,
                                     filters=filters, **validation_fit_kws)
        zpeak_initial_ex = fit_initial['zpeak']
        zbest_initial_ex = fit_initial['zbest']
        zlow_initial = fit_initial['zlower']
        zhigh_initial = fit_initial['zupper']
        coefs_initial = fit_initial['coef']
        best_model_initial = fit_initial['model']
        # refit with the trained dictionary
        fit_out = libs.fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest=lamb_rest, 
                                 D_rest=D_rest, D_sys=D_sys, zinput=False, zgrid=zgrid, 
                                 D_rest_alphas=D_rest_alphas, D_rest_alphas_ndict=D_rest_alphas_ndict,
                                 filters=filters, **validation_fit_kws)
        zpeak_trained_ex = fit_out['zpeak']
        zbest_trained_ex = fit_out['zbest']
        zlow_trained = fit_out['zlower']
        zhigh_trained = fit_out['zupper']
        coefs_trained = fit_out['coef']
        best_model = fit_out['model']
        if ztype == 'zpeak':
            z_trained_ex = zpeak_trained_ex
            z_initial_ex = zpeak_initial_ex
        elif ztype == 'zbest':
            z_trained_ex = zbest_trained_ex
            z_initial_ex = zbest_initial_ex
        nAi = np.sum(coefs_initial!=0)
        nAf = np.sum(coefs_trained!=0)
        # ymax = np.max(best_model) * 2.0
        # ymin = (np.min(best_model) - (np.max(best_model)-np.min(best_model))) * 1.5
        # ymax = np.max(spec_obs_original[idx_i]) * 2.0
        ymax = np.max(spec_obs_original[idx_i]) + (np.max(spec_obs_original[idx_i])-np.min(spec_obs_original[idx_i])) * 1.5
        ymin = np.min(spec_obs_original[idx_i]) - (np.max(spec_obs_original[idx_i])-np.min(spec_obs_original[idx_i])) * 1.5
        # data_max = spec_obs[i]+err_obs[i]
        # data_min = spec_obs[i]-err_obs[i]
        # ymax = np.max(data_max[:66])*1.2
        # ymin = np.min(data_min[:66])*1.2

        ax[row, col].errorbar(lamb_obs, spec_obs[idx_i], err_obs[idx_i], fmt='o', markersize=ms0, markerfacecolor='none', 
                            markeredgecolor=mec0, markeredgewidth=mew0, capsize=cs0, elinewidth=ew0, alpha=alpha0, label="Photometry")
        ax[row, col].plot(lamb_obs, spec_obs_original[idx_i], '.', color=c1, markersize=ms1, alpha=alpha1, 
                            label=fr"Ground Truth, $z_{{true}}={ztrue[idx_i]:.4f}$")
        # ax[row, col].plot(lamb_obs, best_model_initial, '-', linewidth=lw2, color=c2, alpha=alpha2, \
                        # label=fr"Initial Template, $z_{{est}}={z_initial_ex:.3f}^{{+{zhigh_initial-z_initial_ex:.3f}}}_{{-{z_initial_ex-zlow_initial:.3f}}}$")
        ax[row, col].plot(lamb_obs, best_model, linewidth=lw3, color=c3, alpha=alpha3, \
                        label=fr"Trained Template, $z_{{est}}={z_trained_ex:.3f}^{{+{zhigh_trained-z_trained_ex:.3f}}}_{{-{z_trained_ex-zlow_trained:.3f}}}$")
        # ax[row, col].set_xlabel('Observed Wavelength [$\mu$m]', fontsize=8)
        # ax[row, col].set_ylabel('Flux [mJy]', fontsize=8)
        ax[row, col].set_title(fr"Idx={idx_i}, $n_A$={nAf}", fontsize=8)
        ax[row, col].legend(fontsize=legendfontsize, loc='lower center')
        ax[row, col].grid()
        ax[row, col].tick_params(axis='both', which='major', labelsize=ticklabelsize)
        ax[row, col].set_ylim(ymin, ymax)
    fig.supxlabel(r'Observed Wavelength [$\mu$m]', fontsize=labelfontsize)
    fig.supylabel('Flux [mJy]', fontsize=labelfontsize)
    fig.tight_layout()
    if savefig:
        plt.savefig(output_dir / f'example_fitted_SEDs_{ztype}.png', dpi=dpi)
    # else:
        # plt.show()
    # self.fig_num += 1

# Plot fractional uncertainty and z-score in 6 uncertainty bins
def uncertainty_binplots(zs=None, 
                         zp=None, 
                         zl=None, 
                         zh=None, 
                         zstd=None,
                         zp0=None, 
                         zl0=None, 
                         zh0=None, 
                         zstd0=None,
                         dat=None, 
                         sigtype='std',
                         nbins=50, 
                         ztype='zbest', 
                         figsize=(12, 12),
                         fit_initial=False,
                         savefig=False, 
                         output_dirname='', 
                         dpi=300):
    output_dir = Path(output_dirname)
    
    if dat is not None:
        zs = dat['ztrue']
        # zp = dat['zest']
        zl = dat['zlow']
        zh = dat['zhigh']
        zstd = dat['zstd']

        # zp0 = dat['zest_initial']
        zl0 = dat['zlow_initial']
        zh0 = dat['zhigh_initial']
        zstd0 = dat['zstd_initial']
        if ztype == 'zpeak':
            zp = dat['zpeak']
            zp0 = dat['zpeak_initial']
        elif ztype == 'zbest':
            zp = dat['zbest']
            zp0 = dat['zbest_initial']

    if not fit_initial:
        zp0 = zp
        zl0 = zl
        zh0 = zh
        zstd0 = zstd

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

        'boxtextsize': 7.5
        }

    sigma_ranges = [
            (0, 0.003),
            (0.003, 0.01),
            (0.01, 0.03),
            (0.03, 0.1),
            (0.1, 0.2),
            # (0.2, 0.5)
            (0.0, 0.2)
        ]

    hist1_ranges = [
            [-0.01, 0.01],
            [-0.05, 0.05],
            [-0.1, 0.1],
            [-0.2, 0.2],
            [-0.50, 0.50],
            # [-1.0, 1.0]
            [-0.5, 0.5]
        ]
    
    hist2_range = [-5, 5]
    # figsize = (12,12)

    if sigtype == 'std':
        sig = zstd
    else:
        sig = (zh-zl)/2
    if sigtype == 'std':
        sig0 = zstd0
    else:
        sig0 = (zh0-zl0)/2
    frac_uncertainty = sig/(1+zp)
    frac_uncertainty0 = sig0/(1+zp0)

    fig, ax = plt.subplots(4, 3, figsize=figsize)
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

        h = (frac_uncertainty<high) & (frac_uncertainty>=low)
        frac_uncertainty_i = frac_uncertainty[h]
        zsi = zs[h]
        zpi = zp[h]
        sigi = sig[h]
        dzi = zpi - zsi
        err_dist_i = dzi/(1 + zsi)
        nmad_i, eta_i = libs.nmad_eta(zs=zsi, zp=zpi)
        bias_i = np.mean(err_dist_i)
        med_frac_i = np.median(frac_uncertainty_i)
        ngals_i = len(zpi)

        if fit_initial:
            h0 = (frac_uncertainty0<high) & (frac_uncertainty0>=low)
            frac_uncertainty0_i = frac_uncertainty0[h0]
            zs0i = zs[h0]
            zp0i = zp0[h0]     
            sig0i = sig0[h0]
            dz0i = zp0i - zs0i
            err_dist0_i = dz0i/(1 + zs0i)
            nmad0_i, eta0_i = libs.nmad_eta(zs=zs0i, zp=zp0i)
            bias0_i = np.mean(err_dist0_i)
            med_frac0_i = np.median(frac_uncertainty0_i)
            ngals0_i = len(zp0i)


        row1i = hist1_rows[i]
        col1i = hist1_cols[i]

        cts, bins_i, _ = ax[row1i, col1i].hist(err_dist_i, bins=nbins, range=hist1_ranges[i], color=color_args['hist1_color'], alpha=color_args['hist1_alpha'])

        g_pts = 100
        gaussian_xi = np.linspace(hist1_ranges[i][0], hist1_ranges[i][1], g_pts)
        gaussian_i = np.max(cts) * np.exp(-gaussian_xi**2/(2*med_frac_i**2))
        ax[row1i, col1i].plot(gaussian_xi, gaussian_i, linewidth=color_args['g1_lwidth'], color=color_args['g1_color'], alpha=color_args['g1_alpha'])

        if fit_initial:
            cts0, bins0_i, _ = ax[row1i, col1i].hist(err_dist0_i, bins=nbins, range=hist1_ranges[i], histtype='step', 
                                                linewidth=color_args['hist1_lwidth0'], color=color_args['hist1_color0'], alpha=color_args['hist1_alpha0'])
            gaussian0_i = np.max(cts0) * np.exp(-gaussian_xi**2/(2*med_frac0_i**2))
            ax[row1i, col1i].plot(gaussian_xi, gaussian0_i, '--', linewidth=color_args['g1_lwidth0'], color=color_args['g1_color0'], alpha=color_args['g1_alpha0'])
            
            stats_text = (
                fr"NMAD: {nmad0_i:.4f}$\rightarrow${nmad_i:.4f}"+'\n'
                fr"Bias: {bias0_i:.4f}$\rightarrow${bias_i:.4f}"+'\n'
                fr"$\sigma_{{z/(1+z)}}$: {med_frac0_i:.4f}$\rightarrow${med_frac_i:.4f}"+'\n'
                fr"$\eta$: {eta0_i*100:.2f}%$\rightarrow${eta_i*100:.2f}%"+'\n'
                fr"$N_g$: {ngals0_i}$\rightarrow${ngals_i}"
            )
        else:
            stats_text = (
                fr"NMAD: {nmad_i:.4f}"+'\n'
                fr"Bias: {bias_i:.4f}"+'\n'
                fr"$\sigma_{{z/(1+z)}}$: {med_frac_i:.4f}"+'\n'
                fr"$\eta$: {eta_i*100:.2f}%"+'\n'
                fr"$N_g$: {ngals_i}"
            )
        
        ax[row1i, col1i].text(0.02, 0.98, stats_text, ha='left', va='top', transform=ax[row1i, col1i].transAxes, fontsize=color_args['boxtextsize'],
                bbox=dict(facecolor='none', alpha=0.7, linewidth=0.0, edgecolor='black', boxstyle='square,pad=0.5'))

        ax[row1i, col1i].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
        ax[row1i, col1i].set_xlabel(r'($z_{sp}$-$z_s$)/(1+$z_s$)')
        ax[row1i, col1i].set_ylabel(r"$N_g$")
        ax[row1i, col1i].grid(alpha=0.4)

        # Plotting z-score histograms

        row2i = hist2_rows[i]
        col2i = hist2_cols[i]

        zscore_i = (zpi - zsi)/sigi
        mean_zscore_i = np.mean(zscore_i)
        sig_zscore_i = np.std(zscore_i)
        if fit_initial:
            zscore0_i = (zp0i - zs0i)/sig0i
            mean_zscore0_i = np.mean(zscore0_i)
            sig_zscore0_i = np.std(zscore0_i)

        cts2, bins2, _ = ax[row2i, col2i].hist(zscore_i, bins=nbins, range=hist2_range, color=color_args['hist2_color'], alpha=color_args['hist2_alpha'])
        gaussian_xi = np.linspace(hist2_range[0], hist2_range[1], g_pts)
        gaussian_zscore_i = np.max(cts2) * np.exp(-gaussian_xi**2/(2))
        ax[row2i, col2i].plot(gaussian_xi, gaussian_zscore_i, linewidth=color_args['g2_lwidth'], color=color_args['g2_color'], alpha=color_args['g2_alpha'])
        if fit_initial:
            cts0_2, bins0_2, _ = ax[row2i, col2i].hist(zscore0_i, bins=nbins, range=hist2_range, histtype='step', 
                                                linewidth=color_args['hist2_lwidth0'], color=color_args['hist2_color0'], alpha=color_args['hist2_alpha0'])
            gaussian_zscore0_i = np.max(cts0_2) * np.exp(-gaussian_xi**2/(2))
            ax[row2i, col2i].plot(gaussian_xi, gaussian_zscore0_i, '--', linewidth=color_args['g2_lwidth0'], color=color_args['g2_color0'], alpha=color_args['g2_alpha0'])
            stats_text2 = (
                fr"($\mu$, $\sigma$)=({mean_zscore0_i:.2f}, {sig_zscore0_i:.2f})"+'\n'+fr"$\rightarrow$({mean_zscore_i:.2f}, {sig_zscore_i:.2f})"
            )
        else:
            stats_text2 = (
                fr"($\mu$, $\sigma$)=({mean_zscore_i:.2f}, {sig_zscore_i:.2f})"
            )

        ax[row2i, col2i].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
        ax[row2i, col2i].set_xlabel(r'($z_{sp}$-$z_s$)/$\sigma_z$')
        ax[row2i, col2i].set_ylabel(r"$N_g$")
        ax[row2i, col2i].grid(alpha=0.4)


        ax[row2i, col2i].text(0.38, 0.98, stats_text2, ha='right', va='top', transform=ax[row2i, col2i].transAxes, fontsize=color_args['boxtextsize'],
                bbox=dict(facecolor='none', alpha=0.7, linewidth=0.0, edgecolor='black', boxstyle='square,pad=0.5'))
    fig.suptitle(f'Dictionary learning, {ztype}')
    fig.text(0.02, 0.73, 'Redshift error distribution', va='center', ha='center', rotation='vertical', fontsize=10)
    fig.text(0.02, 0.28, 'Z-Score', va='center', ha='center', rotation='vertical', fontsize=10)
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.98])
    if savefig:
        plt.savefig(output_dir / f'binplot_error_dist_zscore_{ztype}.png', dpi=dpi)
    # else:
        # plt.show()
    # self.fig_num += 1



def hexbin_binplot(zs=None, 
                   zp=None, 
                   zl=None, 
                   zh=None, 
                   zstd=None,
                   zp0=None, 
                   zl0=None, 
                   zh0=None, 
                   dat=None, 
                   gridsize=100, 
                   sigtype='highlow',
                   zlims=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                   ztype='zbest', 
                   fit_initial=False,
                   savefig=False, 
                   output_dirname='', 
                   cmap='viridis',
                   mincnt=1,
                   figure_width=12,
                   dpi=300):
    output_dir = Path(output_dirname)

    figsize = (figure_width,figure_width-(1-fit_initial)*(figure_width/2))
    cmap1 = cmap
    cmap2 = 'cividis'

    if dat is not None:
        zs = dat['ztrue']
        # zp = dat['zest']
        zl = dat['zlow']
        zh = dat['zhigh']
        zstd = dat['zstd']
        # zp0 = dat['zest_initial']
        zl0 = dat['zlow_initial']
        zh0 = dat['zhigh_initial']
        if ztype == 'zpeak':
            zp = dat['zpeak']
            zp0 = dat['zpeak_initial']
        elif ztype == 'zbest':
            zp = dat['zbest']
            zp0 = dat['zbest_initial']
    if not fit_initial:
        zp0 = zp
        zl0 = zl
        zh0 = zh

    fig1, ax1 = plt.subplots(4-(1-fit_initial)*2, 3, sharex=False, sharey=False, figsize=figsize)
    # zmin = 0.0
    # zmax = 3.0
    # gridsize = (80,100)
    hexbins_scale = 'log'

    sigma_ranges = [
            (0, 0.003),
            (0.003, 0.01),
            (0.01, 0.03),
            (0.03, 0.1),
            (0.1, 0.2),
            # (0.2, 0.5)
            (0.0, 0.2)
        ]

    p1_cols = [0,1,2,0,1,2]
    p1_rows = [0,0,0,1,1,1]
    p2_cols = [0,1,2,0,1,2]
    p2_rows = [2,2,2,3,3,3]

    if sigtype == 'highlow':
        sig = (zh-zl)/2
    else:
        sig = zstd
    frac_uncertainty = sig/(1+zp)
    if fit_initial:
        sig0 = (zh0-zl0)/2
        frac_uncertainty0 = sig0/(1+zp0)            

    for i, (low, high) in enumerate(sigma_ranges):
        zlim = zlims[i]
        h = (frac_uncertainty<high) & (frac_uncertainty>=low)
        zsi = zs[h]
        zpi = zp[h]
        dzi = zpi - zsi
        err_dist_i = dzi/(1 + zsi)
        nmad_i, eta_i = libs.nmad_eta(zs=zsi, zp=zpi)
        bias_i = np.mean(err_dist_i)
        ngals_i = len(zpi)

        if fit_initial:
            h0 = (frac_uncertainty0<high) & (frac_uncertainty0>=low)
            zs0i = zs[h0]
            zp0i = zp0[h0]
            dz0i = zp0i - zs0i
            err_dist0_i = dz0i/(1 + zs0i)
            nmad0_i, eta0_i = libs.nmad_eta(zs=zs0i, zp=zp0i)
            bias0_i = np.mean(err_dist0_i)
            ngals0_i = len(zp0i)

        row1i = p1_rows[i]
        col1i = p1_cols[i]
        extent = [0, zlim, 0, zlim]
        if sum(h) > 0:
            hb1 = ax1[row1i, col1i].hexbin(zsi, zpi, gridsize=gridsize, extent=extent, bins=hexbins_scale, cmap=cmap1, mincnt=mincnt)
            if mincnt > 1:
                hb11 = ax1[row1i, col1i].hexbin(zsi, zpi, gridsize=gridsize, extent=extent, bins=hexbins_scale, cmap=cmap1, alpha=0.1)
            try:
                cb1 = fig1.colorbar(hb1, ax=ax1[row1i, col1i])
            except:
                pass
        else:
            hb1 = ax1[row1i, col1i].hexbin([0], [0], gridsize=gridsize, extent=extent, bins=hexbins_scale, cmap=cmap1)
            cb1 = fig1.colorbar(hb1, ax=ax1[row1i, col1i])
        ax1[row1i, col1i].plot([0, zlim], [0, zlim], linewidth=0.7, color='salmon', alpha=0.8)

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
        ax1[row1i, col1i].set_xlim(0, zlim)
        ax1[row1i, col1i].set_ylim(0, zlim)

        # Plot z initial hexbins
        if fit_initial:    
            row2i = p2_rows[i]
            col2i = p2_cols[i]

            if sum(h0)>0:
                hb2 = ax1[row2i, col2i].hexbin(zs0i, zp0i, gridsize=gridsize, extent=extent, bins=hexbins_scale, cmap=cmap2)
                cb2 = fig1.colorbar(hb2, ax=ax1[row2i, col2i])
            else:
                hb2 = ax1[row2i, col2i].hexbin([0], [0], gridsize=gridsize, extent=extent, bins=hexbins_scale, cmap=cmap2)
                cb2 = fig1.colorbar(hb2, ax=ax1[row2i, col2i])
            ax1[row2i, col2i].plot([0, zlim], [0, zlim], linewidth=0.7, color='salmon', alpha=0.8)

            ax1[row2i, col2i].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
            ax1[row2i, col2i].set_xlabel(r'$z_{true}$')
            ax1[row2i, col2i].set_ylabel(r"$z_{est}$")
            # ax1[row2i, col2i].grid(alpha=0.4)
            ax1[row2i, col2i].set_xlim(0, zlim)
            ax1[row2i, col2i].set_ylim(0, zlim)

            stats_text2 = (
                fr"NMAD: {nmad0_i:.5f}"+'\n'
                fr"Bias: {bias0_i:.5f}"+'\n'
                fr"$\eta$: {eta0_i*100:.2f}%"+'\n'
                fr"$N_g$: {ngals0_i}"
            )
            ax1[row2i, col2i].text(0.62, 0.23, stats_text2, ha='left', va='top', transform=ax1[row2i, col2i].transAxes, fontsize=7,
                    bbox=dict(facecolor='w', alpha=0.7, linewidth=0.0, edgecolor='black', boxstyle='square,pad=0.5'))

    # fig.suptitle('Dictionary learning')
    fig1.text(0.02, 0.73*fit_initial+0.5*(1-fit_initial), f'Trained dictionary performance ({ztype})', va='center', ha='center', rotation='vertical', fontsize=10)
    if fit_initial:
        fig1.text(0.02, 0.28, f'Initial dictionary performance ({ztype})', va='center', ha='center', rotation='vertical', fontsize=10)
    fig1.tight_layout(rect=[0.02, 0.0, 1.0, 0.98])
    if savefig:
        plt.savefig(output_dir / f'binplot_hexbin_{ztype}.png', dpi=dpi)
    # else:
        # plt.show()
    # self.fig_num += 1

def qq_plots(ztrue_pit, 
             savefig=False, 
             output_dirname='', 
             output_filename='qq_plot.png',
             dpi=300, 
             line_kwargs=None, 
             ref_kwargs=None, 
             figsize=(5, 5),
             grid_alpha=0.3):
    line_kwargs = line_kwargs or {}
    ref_kwargs = ref_kwargs or {}

    line_kwargs.setdefault('linestyle', '-')
    line_kwargs.setdefault('color', 'tab:red')
    line_kwargs.setdefault('linewidth', 1.5)
    line_kwargs.setdefault('alpha', 0.8)

    ref_kwargs.setdefault('linestyle', '--')
    ref_kwargs.setdefault('color', 'tab:green')
    ref_kwargs.setdefault('linewidth', 1)
    ref_kwargs.setdefault('alpha', 0.6)

    ndat = len(ztrue_pit)
    output_dir = Path(output_dirname)
    fig, ax = plt.subplots(figsize=figsize)
    zpit_sorted = np.sort(ztrue_pit)
    zpit_th = (np.arange(ndat)-0.5) / ndat
    
    ax.plot(zpit_th, zpit_sorted, **line_kwargs)
    ax.plot([0,1], [0,1], **ref_kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=grid_alpha)

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    fig.tight_layout()
    if savefig:
        plt.savefig(output_dir / output_filename, dpi=dpi)

def qq_binplots(ztrue_pit, 
                dat=None, 
                zp=None, 
                zl=None, 
                zh=None, 
                zstd=None, 
                line_kwargs=None,
                ref_kwargs=None,
                figsize=(10, 5),
                grid_alpha=0.3,
                output_dirname='',
                output_filename='qq_binplots.png',
                savefig=False, 
                dpi=300):
    output_dir = Path(output_dirname)

    sigma_ranges = [
            (0, 0.003),
            (0.003, 0.01),
            (0.01, 0.03),
            (0.03, 0.1),
            (0.1, 0.2),
            # (0.2, 0.5)
            (0.0, 0.2)
        ]
    
    line_kwargs = line_kwargs or {}
    ref_kwargs = ref_kwargs or {}

    line_kwargs.setdefault('linestyle', '-')
    line_kwargs.setdefault('color', 'tab:red')
    line_kwargs.setdefault('linewidth', 1.5)
    line_kwargs.setdefault('alpha', 0.8)

    ref_kwargs.setdefault('linestyle', '--')
    ref_kwargs.setdefault('color', 'tab:green')
    ref_kwargs.setdefault('linewidth', 1)
    ref_kwargs.setdefault('alpha', 0.6)

    if dat is not None:
        zp = dat['zbest']
        zl = dat['zlow']
        zh = dat['zhigh']
    if zstd is not None:
        sig = zstd
    else:
        sig = (zh-zl)/2
    frac_uncertainty = sig/(1+zp)

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=figsize)

    for i, (low, high) in enumerate(sigma_ranges):
        h = (frac_uncertainty<high) & (frac_uncertainty>=low)
        zpit_i = ztrue_pit[h]
        ndat_i = len(zpit_i)
        zpit_i_sorted = np.sort(zpit_i)
        zpit_i_th = (np.arange(ndat_i)-0.5)/ndat_i

        rowi = i // 3
        coli = i % 3

        if sum(h) > 0:
            ax[rowi, coli].plot(zpit_i_th, zpit_i_sorted, **line_kwargs)
            ax[rowi, coli].plot([0,1], [0,1], **ref_kwargs)
            ax[rowi, coli].grid(alpha=grid_alpha)
            ax[rowi, coli].set_xlim(0, 1)
            ax[rowi, coli].set_ylim(0, 1)
            # ax[rowi, coli].set_xlabel('Theoretical Quantiles')
            # ax[rowi, coli].set_ylabel('Empirical Quantiles')
            ax[rowi, coli].set_title(fr"{low}$<\sigma_{{z/(1+z)}}<${high}")
    fig.supxlabel('Theoretical Quantiles')
    fig.supylabel('Empirical Quantiles')
    fig.tight_layout()
    if savefig:
        plt.savefig(output_dir / output_filename, dpi=dpi)

def fit_eazy_plots(lamb_rest, D_rest, templates_EAZY, savefig=False, output_dirname='', dpi=300):
    output_dir = Path(output_dirname)

    # lamb_rest_resolution = np.mean(np.diff(lamb_rest))
    lamb_rest_resolution = 0.01
    lamb_um = np.arange(0.3, 4.8, lamb_rest_resolution)
    h_lamb_um = (lamb_rest>=min(lamb_um)-0.000001) & (lamb_rest<max(lamb_um))
    templates_EAZY = templates_EAZY[:,h_lamb_um]
    templates_EAZY = templates_EAZY/np.linalg.norm(templates_EAZY, axis=1)[:,None]

    D_rest_interpolated = libs.multiInterp1(lamb_um, lamb_rest, D_rest)
    templates_figsize = (8, 8)
    tick_fontsize = 6
    fig, ax = plt.subplots(7, 1, figsize=templates_figsize)

    for i in range(7):
        # reconstruct this ground-truth template item with the learned template
        coefs, this_model = libs.fit_models_ols(D_rest_interpolated, templates_EAZY[i], np.ones_like(templates_EAZY[i]))
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
    if savefig:
        plt.savefig(output_dir / 'EAZY_reconstruction.png', dpi=dpi)
    # else:
        # plt.show()
    # self.fig_num += 1

def chi2_plots(ynorms, 
               reduced_chi2s=None, 
               reduced_chi2s_fixz=None, 
               last_alpha=None, 
               last_alpha_fixz=None, 
               Nbins=100, 
               N_perbin=20, 
               Nstd=4, 
               catalog='fitting', 
               savefig=False, 
               output_dirname='', 
               dpi=300):
    output_dir = Path(output_dirname)

    alpha_hist = 0.6
    alpha_scatter = 0.1
    s_scatter = 0.5

    if reduced_chi2s is not None:
        med = np.nanmedian(reduced_chi2s)
        # std = np.nanstd(reduced_chi2s)
        std = 1.48 * np.nanmedian(np.abs(reduced_chi2s-np.nanmedian(reduced_chi2s)))
    else:
        med = 0.0
        std = 0.0

    if reduced_chi2s_fixz is not None:
        med_fixz = np.nanmedian(reduced_chi2s_fixz)
        # std_fixz = np.nanstd(reduced_chi2s_fixz)
        std_fixz = 1.48 * np.nanmedian(np.abs(reduced_chi2s_fixz-np.nanmedian(reduced_chi2s_fixz)))

    else:
        med_fixz = 0.0
        std_fixz = 0.0

    max_rchi2 = np.max([(med+Nstd*std), (med_fixz+Nstd*std_fixz)])
    # min_rchi2 = 0.1
    # print(max_rchi2)
    # Nbins = len(ynorms[reduced_chi2s<max_rchi2]) // N_perbin
    fig, ax = plt.subplots(2, 1, figsize=(4,6))

    if reduced_chi2s is not None:
        mask = (reduced_chi2s<max_rchi2) & (~np.isnan(ynorms))
        ax[0].hist(reduced_chi2s[mask], bins=Nbins, alpha=alpha_hist, label='zpeak')
        ax[1].scatter(1/ynorms[mask], reduced_chi2s[mask], s=s_scatter, alpha=alpha_scatter, label='zpeak')

    if reduced_chi2s_fixz is not None:
        mask_fixz = (reduced_chi2s_fixz<max_rchi2) & (~np.isnan(ynorms))
        ax[0].hist(reduced_chi2s_fixz[mask_fixz], bins=Nbins, alpha=alpha_hist, label='ztrue')
        ax[1].scatter(1/ynorms[mask_fixz], reduced_chi2s_fixz[mask_fixz], s=s_scatter, alpha=alpha_scatter, label='ztrue')

    # calculate range of 1/ynorm
    alpha_sigma = 1/ynorms
    med_1_over_ynorm = np.nanmedian(alpha_sigma)
    std_1_over_ynorm = 1.48 * np.nanmedian(np.abs(alpha_sigma-np.nanmedian(alpha_sigma)))
    mask2 = alpha_sigma < (med_1_over_ynorm + Nstd*std_1_over_ynorm)

    # if last_alpha is not None:
        # ax[1].scatter(1/ynorms[mask2], last_alpha[mask2], s=s_scatter, alpha=alpha_scatter, label='zpeak')
    # if last_alpha_fixz is not None:
        # ax[1].scatter(1/ynorms[mask2], last_alpha_fixz[mask2], s=s_scatter, alpha=alpha_scatter, label='ztrue')
    # for sigma_i in range(3):
        # ax[1].plot(1/ynorms[mask2], 1/ynorms[mask2]*(sigma_i+1), linewidth=0.5, alpha=0.8-0.1*sigma_i, color='tab:green')

    # ax0_xmin, ax0_xmax = ax[0].get_xlim()
    ax0_ymin, ax0_ymax = ax[0].get_ylim()
    # ax[0].set_xlim(min_rchi2, max_rchi2)
    ax[0].set_ylim(ax0_ymin, ax0_ymax)
    ax[0].vlines(x=1, ymin=ax0_ymin, ymax=ax0_ymax, linestyle='--', color='k', linewidth=1, alpha=0.8)
    ax[0].set_xlabel(r'$\chi^2/N_{filt}$')
    # ax[1].set_ylim(min_rchi2, max_rchi2)
    ax[1].set_xlabel(r'$1/\|\frac{flux}{error}\|_2$')
    # ax[1].set_ylabel(r'Last $\alpha$')
    ax[1].set_ylabel(r'$\chi^2/N_{filt}$')
    # ax[1].set_xscale('log')
    # ax[1].set_yscale('log')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    fig.tight_layout()
    if savefig:
        plt.savefig(output_dir / f'chi2_plots_{catalog}.png', dpi=dpi)
    # else:
        # plt.show()

    # self.fig_num += 1



# coefficient path, correlation path and estimator path
def plot_fit_paths(idx=None,
                   catalog=None,
                   z=None,
                   lamb_obs=None,
                   spec_obs=None,
                   err_obs=None,
                   dictionary=None,
                   lamb_rest=None,
                   D_rest=None,
                   D_sys=None,
                   alpha=0.0,
                   alpha_sig=0.0,
                   larslasso_kwargs=None,
                   alpha_plot_yscale='linear',
                   estimator_plot_yscale='linear',
                   figsize=(8,10),
                   title_fontsize=12,
                   coef_plot_kwargs=None,
                   alpha_plot_kwargs=None,
                   estimator_plot_kwargs=None,
                   grid_alpha=0.4,
                   print_path=False,
                   output_dirname='',
                   filename='fit_paths.png',
                   dpi=300,
                   save=False
                   ):
    larslasso_kwargs = larslasso_kwargs or {}
    coef_plot_kwargs = coef_plot_kwargs or {}
    alpha_plot_kwargs = alpha_plot_kwargs or {}
    estimator_plot_kwargs = estimator_plot_kwargs or {}

    larslasso_kwargs.setdefault('positive', True)
    larslasso_kwargs.setdefault('center_Xy', False)
    larslasso_kwargs.setdefault('unit_X', True)
    larslasso_kwargs.setdefault('unit_y', True)
    larslasso_kwargs.setdefault('max_iter', 200)
    larslasso_kwargs.setdefault('decimals', 10)
    larslasso_kwargs.setdefault('max_feature', None)

    coef_plot_kwargs.setdefault('linewidth', 1)
    coef_plot_kwargs.setdefault('alpha', 0.8)

    alpha_plot_kwargs.setdefault('linewidth', 1)
    alpha_plot_kwargs.setdefault('markersize', 6)
    alpha_plot_kwargs.setdefault('alpha', 0.8)

    estimator_plot_kwargs.setdefault('linewidth', 1)
    estimator_plot_kwargs.setdefault('markersize', 6)
    estimator_plot_kwargs.setdefault('alpha', 0.6)

    if catalog is not None:
        if z is None:
            z = catalog.ztrue[idx]  # only use catalog's true redshift if z is not provided, otherwise use input z as evaluation redshift
        lamb_obs = catalog.lamb_obs
        spec_obs = catalog.spec_obs[idx]
        err_obs = catalog.err_obs[idx]

    if dictionary is not None:
        lamb_rest = dictionary.lamb_rest
        D_rest = dictionary.D_rest
        D_sys = dictionary.D_sys


    D_thisz = libs.apply_redshift(D_rest, z, lamb_rest, lamb_obs)
    if D_sys is not None:
        D_thisz = np.vstack((D_sys, D_thisz))
    X = D_thisz.T/err_obs[:,None]


    y = spec_obs/err_obs

    # calculate alpha according to 1 sigma and 3 sigma just for visualization purpose
    alpha_1sig = 1/np.linalg.norm(y)
    alpha_3sig = alpha_1sig * 3

    # coef, _, coef_path, alpha_path, _ = libs.larslasso(X, y, alpha=alpha, path=True, **larslasso_kwargs)
    coef, b, model, _, coef_path, alpha_path = libs.fit_model_larslasso(D_thisz, spec_obs, err_obs, alpha=alpha, alpha_sig=alpha_sig, **larslasso_kwargs)
    cp_path = np.zeros(len(coef_path))
    bic_path = np.zeros(len(coef_path))
    chi2_path = np.zeros(len(coef_path))
    ns = spec_obs.shape[0]

    for i in range(coef_path.shape[0]):
        coef_i = coef_path[i]
        model_i = (X @ coef_i + b) * err_obs
        chi2 = np.sum((spec_obs - model_i)**2/err_obs**2)
        chi2_path[i] = chi2
        cp_path[i] = chi2 - ns + 2*i
        bic_path[i] = chi2 + np.log(ns)*i
    
    idx_alpha_min_cp = np.argmin(cp_path)
    idx_alpha_min_bic = np.argmin(bic_path)
    alpha_cp = alpha_path[idx_alpha_min_cp]
    alpha_bic = alpha_path[idx_alpha_min_bic]

    ncoefs_nonzero = np.sum(coef!=0)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    x_coords = np.arange(coef_path.shape[0])
    alpha_now = alpha + alpha_1sig * alpha_sig

    ax[0].plot(x_coords, coef_path, **coef_plot_kwargs)
    ax[0].grid(alpha=grid_alpha)
    ax[0].set_title(fr'Coefficient path, $n_A$={ncoefs_nonzero}, $\alpha={alpha_now}$, positive={larslasso_kwargs['positive']}', fontsize=title_fontsize)
    ax[0].set_ylabel("Coefficients")
    # ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[1].plot(x_coords, alpha_path[:len(x_coords)], '.-', **alpha_plot_kwargs)
    ax[1].plot(x_coords, np.full(x_coords.shape, alpha_3sig), '--', linewidth=0.8, alpha=0.6, label=fr'3-$\sigma$')
    ax[1].plot(x_coords, np.full(x_coords.shape, alpha_1sig), linewidth=0.8, alpha=0.6, label=fr'1-$\sigma$')
    ax[1].plot(x_coords, np.full(x_coords.shape, alpha_now), '-', linewidth=1, alpha=0.8, label=f'{alpha_now:.4f}')
    ax[1].grid(alpha=grid_alpha)
    ax[1].set_yscale(alpha_plot_yscale)
    ax[1].set_title(fr'Correlation path', fontsize=title_fontsize)
    ax[1].legend()
    ax[1].set_ylabel(r"$\alpha_k$")
    # ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[2].plot(x_coords, cp_path, '.-', **estimator_plot_kwargs, label=r'$C_p$')
    ax[2].plot(x_coords, bic_path, '.-', **estimator_plot_kwargs, label='BIC')
    ax[2].set_yscale(estimator_plot_yscale)
    ax[2].grid(alpha=grid_alpha)
    ax[2].set_xlabel('Steps (k)')
    ax[2].set_title(fr"$\alpha_{{cp}}$={alpha_cp:.4f} ({idx_alpha_min_cp}), $\alpha_{{BIC}}$={alpha_bic:.4f} ({idx_alpha_min_bic})", fontsize=title_fontsize)
    ax[2].set_ylabel('Estimators')
    ax[2].legend()
    ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()

    if save:
        output_dir = Path(output_dirname)
        plt.savefig(output_dir / filename, dpi=dpi)

    if print_path:
        print('alpha path:\t', alpha_path)
        print('Cp path:\t', cp_path)
        print('BIC path:\t', bic_path)




# plotting best fit SED, chi2 & likelihood vs zgrid 
def plot_bestfit_zgrid(idx=None,
                       catalog=None,
                       zinput=False,
                       lamb_obs=None,
                       spec_obs=None,
                       spec_obs_original=None,
                       err_obs=None,
                       dictionary=None,
                       lamb_rest=None,
                       D_rest=None,
                       D_rest_alphas=None,
                       D_rest_alphas_ndict=None,
                       D_sys=None,
                       alpha=0.0,
                       alpha_sig=0.0,
                       fit_zgrid_kwargs=None,
                       figsize=(10, 10),
                       title_fontsize=12,
                       label_fontsize=10,
                       legend_fontsize=10,
                       sed_errorbar_kwargs=None,
                       sed_plot_kwargs=None,
                       sed_ymin_ymax=None,
                       cost_plot_kwargs=None,
                    #    likelihood_plot_kwargs=None,
                       divider_kwargs=None,
                       cost_likelihood_scatter_kwargs=None,
                       cost_likelihood_line_kwargs=None,
                       vlines_kwargs=None,
                       vspan_kwargs=None,
                       cmap_name='viridis',
                       sed_ylabel='Flux [uJy]',
                       grid_alpha=0.3,
                       plot_chi2=True,
                       save=False,
                       output_dirname='',
                       filename='bestfit_zgrid.png',
                       dpi=300,
                       ):
    fit_zgrid_kwargs = fit_zgrid_kwargs or {}
    sed_errorbar_kwargs = sed_errorbar_kwargs or {}
    sed_plot_kwargs = sed_plot_kwargs or {}
    cost_plot_kwargs = cost_plot_kwargs or {}
    # likelihood_plot_kwargs = likelihood_plot_kwargs or {}
    divider_kwargs = divider_kwargs or {}
    cost_likelihood_scatter_kwargs = cost_likelihood_scatter_kwargs or {}
    cost_likelihood_line_kwargs = cost_likelihood_line_kwargs or {}
    vlines_kwargs = vlines_kwargs or {}
    vspan_kwargs = vspan_kwargs or {}

    fit_zgrid_kwargs.setdefault('positive', True)
    fit_zgrid_kwargs.setdefault('center_Xy', False)
    fit_zgrid_kwargs.setdefault('unit_X', True)
    fit_zgrid_kwargs.setdefault('unit_y', True)
    fit_zgrid_kwargs.setdefault('max_feature', None)
    fit_zgrid_kwargs.setdefault('local_finegrid', False)

    sed_errorbar_kwargs.setdefault('fmt', 'o')
    sed_errorbar_kwargs.setdefault('markersize', 4)
    sed_errorbar_kwargs.setdefault('markerfacecolor', 'none')
    sed_errorbar_kwargs.setdefault('markeredgecolor', 'cadetblue')
    sed_errorbar_kwargs.setdefault('markeredgewidth', 0.8)
    sed_errorbar_kwargs.setdefault('capsize', 3)
    sed_errorbar_kwargs.setdefault('elinewidth', 0.5)
    sed_errorbar_kwargs.setdefault('alpha', 0.6)

    sed_plot_kwargs.setdefault('linewidth', 1.5)
    sed_plot_kwargs.setdefault('alpha', 0.7)

    divider_kwargs.setdefault('size', "3%")
    divider_kwargs.setdefault('pad', 0.05)

    cost_likelihood_scatter_kwargs.setdefault('s', 4)
    cost_likelihood_scatter_kwargs.setdefault('alpha', 0.8)
    # cost_likelihood_scatter_kwargs.setdefault('cmap', cmap_name)
    cost_likelihood_line_kwargs.setdefault('linewidth', 1)
    cost_likelihood_line_kwargs.setdefault('linestyle', '-')
    cost_likelihood_line_kwargs.setdefault('alpha', 0.2)
    cost_likelihood_line_kwargs.setdefault('color', 'grey')

    vlines_kwargs.setdefault('linestyle', '--')
    vlines_kwargs.setdefault('linewidth', 1)
    vlines_kwargs.setdefault('alpha', 0.7)
    vspan_kwargs.setdefault('alpha', 0.2)
    vspan_kwargs.setdefault('color', 'salmon')

    if catalog is not None:
        ztrue = catalog.ztrue[idx]
        lamb_obs = catalog.lamb_obs
        spec_obs = catalog.spec_obs[idx]
        err_obs = catalog.err_obs[idx]
        if zinput:
            zinput = ztrue
        try: 
            spec_obs_original = catalog.spec_obs_original[idx]
        except:
            spec_obs_original = None
    else:
        ztrue = zinput

    if dictionary is not None:
        lamb_rest = dictionary.lamb_rest
        D_rest = dictionary.D_rest
        D_sys = dictionary.D_sys
        D_rest_alphas = dictionary.D_rest_alphas
        D_rest_alphas_ndict = dictionary.D_rest_alphas_ndict

    fit_out = libs.fit_zgrid(lamb_obs=lamb_obs,
                             spec_obs=spec_obs,
                             err_obs=err_obs,
                             lamb_rest=lamb_rest,
                             D_rest=D_rest,
                             D_rest_alphas=D_rest_alphas,
                             D_rest_alphas_ndict=D_rest_alphas_ndict,
                             D_sys=D_sys,
                             zinput=zinput,
                             alpha=alpha,
                             alpha_sig=alpha_sig,
                             output_all_coefs=True,
                             error=True,
                             **fit_zgrid_kwargs)
    zpeak = fit_out['zpeak']
    zbest = fit_out['zbest']
    zlow = fit_out['zlower']
    zhigh = fit_out['zupper']
    coefs_zgrid = fit_out['coef']
    b = fit_out['b']
    model_zpeak = fit_out['model']
    zgrid = fit_out['zgrid']
    chi2_zgrid = fit_out['cost_zgrid']
    alpha_path_zpeak = fit_out['alpha_path']

    ncoefs = np.sum(coefs_zgrid!=0, axis=1)
    likelihood_zgrid = np.exp(-(chi2_zgrid-np.min(chi2_zgrid))/2)
    nA_best = ncoefs[np.where(zgrid==zpeak)[0][0]]

    # cmap = mpl.colormaps[cmap_name]
    cmap = plt.get_cmap(cmap_name, max(ncoefs))
    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(0.5, max(ncoefs)+1.5, 1), ncolors=max(ncoefs))
    ticks = [i for i in range(max(ncoefs)+1)]

    if plot_chi2:
        nrows = 3
    else:
        nrows = 2

    fig, ax = plt.subplots(nrows, 1, figsize=figsize, constrained_layout=False)
    nonzero_mask = err_obs != 50000
    ax[0].errorbar(lamb_obs[nonzero_mask], spec_obs[nonzero_mask], err_obs[nonzero_mask], **sed_errorbar_kwargs)
    if spec_obs_original is not None:
        ax[0].plot(lamb_obs, spec_obs_original, **sed_plot_kwargs, label='Truth')
    ax[0].plot(lamb_obs, model_zpeak, **sed_plot_kwargs, label='Best fit model')
    ax[0].grid(alpha=grid_alpha)
    if sed_ymin_ymax is not None:
        ax[0].set_ylim(sed_ymin_ymax)
    ax[0].legend(fontsize=legend_fontsize)
    ax[0].set_ylabel(sed_ylabel, fontsize=label_fontsize)
    ax[0].set_xlabel(r'wavelength [$\mu m$]', fontsize=label_fontsize)

    row_i = 1
    if plot_chi2:
        divider1 = make_axes_locatable(ax[row_i])
        cax_chi2 = divider1.append_axes("right", **divider_kwargs)
        scatter_chi2 = ax[row_i].scatter(zgrid, chi2_zgrid, c=ncoefs, **cost_likelihood_scatter_kwargs, cmap=cmap, norm=norm)
        cbar_chi2 = fig.colorbar(scatter_chi2, cax=cax_chi2, ticks=ticks, label=r'$N_{dicts}$')
        cbar_chi2.ax.tick_params(labelsize=label_fontsize)
        ax[row_i].plot(zgrid, chi2_zgrid, **cost_likelihood_line_kwargs)
        ax[row_i].grid(alpha=grid_alpha)
        ymin_chi2, ymax_chi2 = ax[row_i].get_ylim()
        ax[row_i].vlines(ztrue, ymin=ymin_chi2, ymax=ymax_chi2, **vlines_kwargs, color='k')
        ax[row_i].set_ylim(ymin_chi2, ymax_chi2)
        ax[row_i].set_ylabel('Cost', fontsize=label_fontsize)
        row_i = row_i + 1

    ymin = -0.05
    ymax = 1.05
    divider = make_axes_locatable(ax[row_i])
    cax_ll = divider.append_axes("right", **divider_kwargs)
    scatter_ll = ax[row_i].scatter(zgrid, likelihood_zgrid, c=ncoefs, **cost_likelihood_scatter_kwargs, cmap=cmap, norm=norm)
    ax[row_i].plot(zgrid, likelihood_zgrid, **cost_likelihood_line_kwargs)
    cbar = fig.colorbar(scatter_ll, cax=cax_ll, ticks=ticks, label=r'$N_{dicts}$')
    cbar.ax.tick_params(labelsize=label_fontsize)
    ax[row_i].vlines(ztrue, ymin, ymax, **vlines_kwargs, color='k', label=r'$z_{true}$')
    ax[row_i].vlines(zbest, ymin, ymax, **vlines_kwargs, color='tab:green', label=r'$z_{best}$')
    ax[row_i].vlines(zpeak, ymin, ymax, **vlines_kwargs, color='tab:red', label=r'$z_{peak}$')
    ax[row_i].axvspan(zlow, zhigh, **vspan_kwargs, label="68% confidence level")
    ax[row_i].set_ylim(ymin, ymax)
    ax[row_i].set_ylabel('Likelihood', fontsize=label_fontsize)
    ax[row_i].grid(alpha=grid_alpha)
    ax[row_i].legend(fontsize=legend_fontsize)
    ax[row_i].set_xlabel('redshift', fontsize=label_fontsize)
    if row_i == 2:
        ax[row_i].sharex(ax[row_i-1])

    fig.suptitle(fr"$z_{{true}}$={ztrue:.5f}, $z_{{peak}}={zpeak:.3f}^{{+{zhigh-zpeak:.3f}}}_{{-{zpeak-zlow:.3f}}}$, $z_{{best}}={zbest:.3f}$, $n_A$={nA_best}", fontsize=title_fontsize)
    fig.tight_layout()

    if save:
        output_dir = Path(output_dirname)
        plt.savefig(output_dir / filename, dpi=dpi)
