import numpy as np
import pandas as pd
from numba import jit, njit
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
import yaml
from pathlib import Path
from . import lib_dictlearn as fx



class diagnostic_plots:
    def __init__(self, output_dirname=None):
        if output_dirname is not None:
            self.savefig = True
        else:
            self.savefig = False
            output_dirname = ''
        self.dpi = 300
        self.output_dirname = output_dirname
        self.output_dir = Path(output_dirname)
        self.fig_num = 0


    def template_plots(self, lamb_rest, D_rest, D_rest_initial):
        
        fig1, ax1 = plt.subplots(num=self.fig_num)
        ax1.plot(lamb_rest, D_rest.T, '-', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel(r"Wavelength [$\mu m$]")
        ax1.set_ylabel('Flux [arb]')
        ax1.set_title('Learned Templates')
        ax1.grid()
        fig1.tight_layout()
        if self.savefig:
            plt.savefig(self.output_dir / 'trained_templates.png', dpi=self.dpi)
        else:
            plt.show()
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
        if self.savefig:
            plt.savefig(self.output_dir / f'trained_template_multiplot.png', dpi=self.dpi)
        else:
            plt.show()
        self.fig_num += 1

    def zp_zs_plots(self, ztrue, zbest_initial, zbest_trained, zmin=0.0, zmax=3.0, catalog='fitting', gridsize=150, gridsize2=(150,30)):
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

        nmad_i, eta_i = fx.nmad_eta(zs=ztrue, zp=zbest_initial)
        nmad_f, eta_f = fx.nmad_eta(zs=ztrue, zp=zbest_trained)

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
        ax[0,0].plot(ztrue, zbest_initial, m0, color=markercolor0, markersize=m0size, markeredgecolor=m0edgec, markeredgewidth=markeredgewidth0, alpha=m0alpha,
                    label=f'Initial, $\eta={eta_i*100:.3f}$%, $\sigma_{{NMAD}}={nmad_i*100:.3f}$%')
        ax[0,0].plot(ztrue, zbest_trained, m1, color=markercolor1, markersize=m1size, markeredgecolor=m1edgec, markeredgewidth=markeredgewidth1, alpha=m1alpha,
                    label=f'Trained, $\eta={eta_f*100:.3f}$%, $\sigma_{{NMAD}}={nmad_f*100:.3f}$%')
        ax[1,0].plot(ztrue, (zbest_initial-ztrue)/(1+ztrue), m0, color=markercolor0, markersize=m0size, markeredgecolor=m0edgec, \
                     markeredgewidth=markeredgewidth0, alpha=m0alpha)
        ax[1,0].plot(ztrue, (zbest_trained-ztrue)/(1+ztrue), m1, color=markercolor1, markersize=m1size, markeredgecolor=m1edgec, \
                     markeredgewidth=markeredgewidth1, alpha=m1alpha)

        hb1 = ax[0,1].hexbin(ztrue, zbest_trained, gridsize=gridsize, extent=[zmin, zmax, zmin, zmax], bins=bins)
        cb1 = fig.colorbar(hb1, ax=ax[0,1])
        hb2 = ax[1,1].hexbin(ztrue, (zbest_trained-ztrue)/(1+ztrue), gridsize=gridsize2, extent=[zmin, zmax, -bottom_ylim, bottom_ylim], bins=bins)
        cb2 = fig.colorbar(hb2, ax=ax[1,1])

        ax[0,0].plot([zmin-lim_offset, zmax+lim_offset], [zmin-lim_offset, zmax+lim_offset], '-', alpha=0.8, color='g', linewidth=1)
        ax[0,1].plot([zmin-lim_offset, zmax+lim_offset], [zmin-lim_offset, zmax+lim_offset], linewidth=0.7, color='salmon', alpha=0.8)
        ax[1,0].plot([zmin-lim_offset, zmax+lim_offset], [0,0],'-', alpha=0.8, color='g', linewidth=1)
        ax[1,1].plot([zmin-lim_offset, zmax+lim_offset], [0,0],'-', linewidth=0.7, color='salmon', alpha=0.8)

        ax[0,0].tick_params(axis='both', which='major', labelsize=tickfontsize)
        ax[1,0].tick_params(axis='both', which='major', labelsize=tickfontsize)
        ax[0,0].legend(fontsize=legendfontsize, framealpha=0.9, loc='upper left')
        # axs[1].legend(fontsize=20, loc='lower right')
        # fig.suptitle('Fitting Catalog Redshift Estimation')
        fig.tight_layout()
        if self.savefig:
            plt.savefig(self.output_dir / f'redshift_estimation_performance_{catalog}_catalog.png', dpi=self.dpi)
        else:
            plt.show()
        self.fig_num += 1

    def sparsity_report(self, coefs_trained, max_feature=None, add_constant=False):
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

        bar1 = ax[1].bar(np.arange(Ndict), p_counts, color='teal')
        ax[1].set_title('# of times dictionary have been used')
        bar21 = ax[2].bar(np.arange(Ndict), p_pos_sums, color='brown', alpha=0.6)
        bar22 = ax[2].bar(np.arange(Ndict), p_sums, color='salmon', alpha=1.0)
        if add_constant:
            bar1[-1].set_alpha(0.35)
            bar21[-1].set_alpha(0.35)
            bar22[-1].set_alpha(0.35)
        if max_feature is not None:
            max_line = max_feature + 1
            ymin, ymax = ax[0].get_ylim()
            ax[0].vlines(max_line, ymin=ymin, ymax=ymax, linewidth=1, color='tab:red')
            ax[0].set_ylim(ymin, ymax)

        ax[2].set_title('Sum of coefficients for each dictionary')

        fig.tight_layout()
        plt.savefig(self.output_dir / 'sparsity_reports.png')
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
            spec_obs_i = np.ascontiguousarray(spec_obs[idx_i])
            err_obs_i = np.ascontiguousarray(err_obs[idx_i])
            zbest_initial_ex,zlow_initial,zhigh_initial, \
            coefs_initial,b_initial,best_model_initial,_,_ = fx.fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest, 
                                                            D_rest=D_rest_initial, zinput=False, zgrid=zgrid, 
                                                            filter_info=filter_info, **validation_fit_kws)
            # refit with the trained dictionary
            zbest_trained_ex,zlow_trained,zhigh_trained, \
            coefs_trained,b_best,best_model,_,_ = fx.fit_zgrid(lamb_obs, spec_obs_i, err_obs_i, lamb_rest, 
                                                    D_rest=D_rest, zinput=False, zgrid=zgrid, 
                                                    filter_info=filter_info, **validation_fit_kws)
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
            ax[row, col].plot(lamb_obs, best_model_initial, '-', linewidth=lw2, color=c2, alpha=alpha2, \
                            label=fr"Initial Template, $z_{{est}}={zbest_initial_ex:.3f}^{{+{zhigh_initial-zbest_initial_ex:.3f}}}_{{-{zbest_initial_ex-zlow_initial:.3f}}}$")
            ax[row, col].plot(lamb_obs, best_model, linewidth=lw3, color=c3, alpha=alpha3, \
                            label=fr"Trained Template, $z_{{est}}={zbest_trained_ex:.3f}^{{+{zhigh_trained-zbest_trained_ex:.3f}}}_{{-{zbest_trained_ex-zlow_trained:.3f}}}$")
            # ax[row, col].set_xlabel('Observed Wavelength [$\mu$m]', fontsize=8)
            # ax[row, col].set_ylabel('Flux [mJy]', fontsize=8)
            ax[row, col].set_title(fr"Idx={idx_i}, $n_A$={nAf}", fontsize=8)
            ax[row, col].legend(fontsize=legendfontsize, loc='lower center')
            ax[row, col].grid()
            ax[row, col].tick_params(axis='both', which='major', labelsize=ticklabelsize)
            ax[row, col].set_ylim(ymin, ymax)
        fig.supxlabel('Observed Wavelength [$\mu$m]', fontsize=labelfontsize)
        fig.supylabel('Flux [mJy]', fontsize=labelfontsize)
        fig.tight_layout()
        if self.savefig:
            plt.savefig(self.output_dir / 'example_fitted_SEDs.png', dpi=self.dpi)
        else:
            plt.show()
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

            nmad_i, eta_i = fx.nmad_eta(zs=zsi, zp=zpi)
            nmad0_i, eta0_i = fx.nmad_eta(zs=zs0i, zp=zp0i)

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
        if self.savefig:
            plt.savefig(self.output_dir / 'binplot_error_dist_zscore.png', dpi=self.dpi)
        else:
            plt.show()
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

            nmad_i, eta_i = fx.nmad_eta(zs=zsi, zp=zpi)
            nmad0_i, eta0_i = fx.nmad_eta(zs=zs0i, zp=zp0i)
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
        if self.savefig:
            plt.savefig(self.output_dir / 'binplot_hexbin.png', dpi=self.dpi)
        else:
            plt.show()
        self.fig_num += 1


    def fit_eazy_plots(self, lamb_rest, D_rest, templates_EAZY):

        # lamb_rest_resolution = np.mean(np.diff(lamb_rest))
        lamb_rest_resolution = 0.01
        lamb_um = np.arange(0.3, 4.8, lamb_rest_resolution)
        h_lamb_um = (lamb_rest>=min(lamb_um)-0.000001) & (lamb_rest<max(lamb_um))
        templates_EAZY = templates_EAZY[:,h_lamb_um]
        templates_EAZY = templates_EAZY/np.linalg.norm(templates_EAZY, axis=1)[:,None]

        D_rest_interpolated = fx.multiInterp1(lamb_um, lamb_rest, D_rest)
        templates_figsize = (8, 8)
        tick_fontsize = 6
        fig, ax = plt.subplots(7, 1, figsize=templates_figsize, num=self.fig_num)

        for i in range(7):
            # reconstruct this ground-truth template item with the learned template
            coefs, this_model = fx.fit_models_ols(D_rest_interpolated, templates_EAZY[i], np.ones_like(templates_EAZY[i]))
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
        if self.savefig:
            plt.savefig(self.output_dir / 'EAZY_reconstruction.png', dpi=self.dpi)
        else:
            plt.show()
        self.fig_num += 1

