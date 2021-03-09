# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for re-plotting saved data as well as generating overlay plots with the observational structure functions.
"""

from astropy.modeling import fitting
from astropy.modeling.powerlaws import PowerLaw1D, BrokenPowerLaw1D
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

n_samples = 50  # number of samples to plot the figures for.
kernel = 'RQ'  # ['Matern' 'RQ'] are the options

if __name__ == '__main__':

    # 5.3 is the resolution, multiplied by 10 to enable file saving.

    if kernel == 'Matern':
        tag = f'Matern_53_days_averaged_{n_samples}_samples'
    else:
        tag = f'RQ_53_days_averaged_{n_samples}_samples'

    # Discard the first two points because tau < 10

    uv_gp_tao_plot = np.loadtxt(f'sf_gp_data/uv/{kernel}_{n_samples}_times')[2:]
    uv_gp_mean_structure_function = np.loadtxt(f'sf_gp_data/uv/mean_structure_function_{n_samples}_{kernel}')[2:]
    # Divide by the sqrt of the sample size to convert to standard error
    uv_gp_std_structure_function = np.loadtxt(f'sf_gp_data/uv/std_structure_function_{n_samples}_{kernel}')[2:]/np.sqrt(n_samples)

    # Observational Data

    uv_tao_plot = np.loadtxt(f'sf_data/uv/times_most_recent')[2:]
    uv_mean_structure_function = np.loadtxt(f'sf_data/uv/structure_function_most_recent')[2:]
    uv_std_structure_function = np.loadtxt(f'sf_data/uv/errors_most_recent')[2:]

    # Fit the GP UVW2 structure functions using a power law or broken power law
    power_law_model = PowerLaw1D()
    broken_power_law_model = BrokenPowerLaw1D(alpha_1=1, alpha_2=1, x_break=100)
    fitter_plm = fitting.SimplexLSQFitter()
    fitter_bplm = fitting.SimplexLSQFitter()
    power_law_tao_vals = uv_gp_tao_plot
    uv_gp_mean_structure_function *= 1e29  # multiply by a large number to prevent error in fitting power laws.
    uv_gp_std_structure_function *= 1e29
    power_law_sf_vals = uv_gp_mean_structure_function

    # Large number of iterations required for optimiser convergence (maxiter argument)
    # Use uncertainties to improve the fit as per this tutorial:
    # https://docs.astropy.org/en/stable/modeling/example-fitting-line.html#fit-using-uncertainties

    plm = fitter_plm(power_law_model, power_law_tao_vals, power_law_sf_vals, weights=1/uv_gp_std_structure_function,
                      maxiter=5000)

    plm_index1 = plm.alpha.value
    print(plm_index1)

    # plot on the same axis

    fig, ax = plt.subplots(1)
    color = 'tab:blue'
    plt.xscale('log')
    #ax.scatter(uv_tao_plot, uv_mean_structure_function, s=10, marker='+', color=color, label='obs')
    ax.errorbar(uv_tao_plot, uv_mean_structure_function, yerr=uv_std_structure_function, fmt='o', color=color,
                markersize=3, linewidth=0.5, label='Observational')
    ax.set_yscale('log')
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    ax.set_ylabel('Observational SF', color=color, fontsize=12)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    plt.yticks([0.1, 1], color=color)
    plt.xlim([10, 600])

    color = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gaussian Process SF', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax2.plot(power_law_tao_vals, plm(power_law_tao_vals), label='Power Law', color='k')
    #ax2.scatter(uv_gp_tao_plot, uv_gp_mean_structure_function, s=10, marker='x', color=color, label='gp')
    ax2.errorbar(uv_gp_tao_plot, uv_gp_mean_structure_function, yerr=uv_gp_std_structure_function.flatten(),
                 fmt='x', color=color, markersize=3, linewidth=0.5, label='GP')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.minorticks_off()
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    plt.tight_layout()
    fig.legend(loc=4, bbox_to_anchor=[0.25, 0.25, 0.6, 0])
    plt.savefig(f'cosmetic_figures/gp_structure_function_uv_{tag}_on_same_axis_'
                f'alpha_is_{str(100*plm_index1)}.png')
    plt.close()

    x_ray_gp_tao_plot = np.loadtxt(f'sf_gp_data/xray/{kernel}_{n_samples}_times')[2:]
    x_ray_gp_mean_structure_function = np.loadtxt(f'sf_gp_data/xray/mean_structure_function_{n_samples}_{kernel}')[2:]
    # Divide by sqrt the sample size to convert standard deviation to standard error
    x_ray_gp_std_structure_function = np.loadtxt(f'sf_gp_data/xray/std_structure_function_{n_samples}_{kernel}')[2:]/np.sqrt(n_samples)

    x_ray_tao_plot = np.loadtxt(f'sf_data/xray/times_most_recent')[2:]
    x_ray_mean_structure_function = np.loadtxt(f'sf_data/xray/structure_function_most_recent')[2:]
    x_ray_std_structure_function = np.loadtxt(f'sf_data/xray/errors_most_recent')[2:]

    # Fit the GP X-ray structure functions using a power law or broken power law
    power_law_model = PowerLaw1D()
    broken_power_law_model = BrokenPowerLaw1D(x_break=120)
    fitter_plm = fitting.SimplexLSQFitter()
    fitter_bplm = fitting.SimplexLSQFitter()
    power_law_tao_vals_xray = x_ray_gp_tao_plot
    power_law_sf_vals_xray = x_ray_gp_mean_structure_function

    # Large number of iterations required for optimiser convergence (maxiter argument). Use weights equal to the
    # uncertainties to improve the fit as per this tutorial:
    # https://docs.astropy.org/en/stable/modeling/example-fitting-line.html#fit-using-uncertainties

    if kernel == 'RQ':
        bplm_xray = fitter_plm(power_law_model, power_law_tao_vals_xray, power_law_sf_vals_xray,
                              weights=1/x_ray_gp_std_structure_function, maxiter=5000)
        bplm_index_xray = bplm_xray.alpha.value
        print(bplm_index_xray)

    else:  # Use broken power law for Matern kernel
        bplm_xray = fitter_bplm(broken_power_law_model, power_law_tao_vals_xray, power_law_sf_vals_xray,
                                weights=1/x_ray_gp_std_structure_function, maxiter=5000)

        bplm_index_xray_1 = bplm_xray.alpha_1.value
        bplm_index_xray_2 = bplm_xray.alpha_2.value
        bplm_break_xray = bplm_xray.x_break.value
        print(bplm_index_xray_1)
        print(bplm_index_xray_2)
        print(bplm_break_xray)

    # plot on the same axis

    fig, ax = plt.subplots(1)
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    color = 'tab:blue'
    ax.set_ylabel('Observational SF', fontsize=12, color=color)
    ax.errorbar(x_ray_tao_plot, x_ray_mean_structure_function, yerr=x_ray_std_structure_function,
                fmt='o', markersize=3, linewidth=0.5, color=color, label='Observational')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([0.5, 3.5])
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.set_yticks([0.6, 1, 2, 3, 4])
    plt.xticks([0.1, 1])
    ax.minorticks_off()
    if kernel == 'Matern':
        scale_factor = 0.9
    else:
        scale_factor = 1.35
    color = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gaussian Process SF', color=color, fontsize=12)  # we already handled the x-label with ax1
    if kernel == 'RQ':
        ax2.plot(power_law_tao_vals_xray, scale_factor*bplm_xray(power_law_tao_vals_xray), color='k', label='Power Law')
    else:
        ax2.plot(power_law_tao_vals_xray, scale_factor*bplm_xray(power_law_tao_vals_xray), color='k', label='Broken Power Law')
    ax.errorbar(x_ray_gp_tao_plot, scale_factor*x_ray_gp_mean_structure_function, yerr=x_ray_gp_std_structure_function, fmt='x', color=color, markersize=3, linewidth=0.5, label='GP')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.5, 3.5])
    # ax2.set_yticks([1])
    plt.xlim([10, 700])
    plt.tight_layout()
    fig.legend(loc=4, bbox_to_anchor=[0.275, 0.75, 0.15, 0])
    #ax2.minorticks_off()
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    if kernel == 'Matern':
        plt.savefig(f'cosmetic_figures/gp_structure_function_xray_{tag}_on_same_axis_{int(scale_factor*100)}_'
                    f'_alpha_1_is{str(100*bplm_index_xray_1)}_alpha_2_is{str(100*bplm_index_xray_2)}'
                    f'_x_break_is{str(100*bplm_break_xray)}.png')
    else:
        plt.savefig(f'cosmetic_figures/gp_structure_function_xray_{tag}_on_same_axis_{int(scale_factor*100)}_'
                    f'_alpha_1_is{str(100*bplm_index_xray)}.png')

    plt.close()
