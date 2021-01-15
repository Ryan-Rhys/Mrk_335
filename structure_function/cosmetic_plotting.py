# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for re-plotting saved data as well as generating overlay plots with the observational structure functions.
"""

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
    uv_gp_std_structure_function = np.loadtxt(f'sf_gp_data/uv/std_structure_function_{n_samples}_{kernel}')[2:]

    # Observational Data

    uv_tao_plot = np.loadtxt(f'sf_data/uv/times_most_recent')[2:]
    uv_mean_structure_function = np.loadtxt(f'sf_data/uv/structure_function_most_recent')[2:]
    uv_std_structure_function = np.loadtxt(f'sf_data/uv/errors_most_recent')[2:]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.scatter(uv_tao_plot, uv_mean_structure_function, s=10, marker='+', label='obs')
    ax2.scatter(uv_gp_tao_plot, uv_gp_mean_structure_function, s=10, marker='+', label='gp')
    ax1.errorbar(uv_tao_plot, uv_mean_structure_function, yerr=uv_std_structure_function, fmt='o', markersize=3, linewidth=1, label='obs')
    ax2.errorbar(uv_gp_tao_plot, uv_gp_mean_structure_function, yerr=uv_gp_std_structure_function, fmt='o', markersize=3, linewidth=1, label='gp')
    plt.xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.xlabel(r'$\tau$' + ' (days)')
    ax1.set_ylabel('SF')
    ax2.set_ylabel(f'GP-{kernel} SF')
    plt.xlim([10, 700])
    fig.suptitle('UVW2 Structure Function')
    plt.savefig(f'cosmetic_figures/gp_structure_function_uv_{tag}')
    plt.close()

    # plot on the same axis

    fig, ax = plt.subplots(1)
    color = 'tab:blue'
    plt.xscale('log')
    #ax.scatter(uv_tao_plot, uv_mean_structure_function, s=10, marker='+', color=color, label='obs')
    ax.errorbar(uv_tao_plot, uv_mean_structure_function, yerr=uv_std_structure_function, fmt='o', color=color, markersize=3, linewidth=0.5, label='Observational')
    ax.set_yscale('log')
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    ax.set_ylabel('Observational SF', color=color, fontsize=12)
    plt.yticks([0.1, 1], color=color)
    plt.xlim([10, 600])

    color = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gaussian Process SF', color=color, fontsize=12)  # we already handled the x-label with ax1
    #ax2.scatter(uv_gp_tao_plot, uv_gp_mean_structure_function, s=10, marker='x', color=color, label='gp')
    ax2.errorbar(uv_gp_tao_plot, uv_gp_mean_structure_function, yerr=uv_gp_std_structure_function, fmt='x', color=color, markersize=3, linewidth=0.5, label='GP')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.006, 0.1)
    #plt.title(f'UVW2 Structure Function {kernel} Kernel', fontsize=16)
    plt.tight_layout()
    fig.legend(loc=4, bbox_to_anchor=[0.25, 0.25, 0.6, 0])
    plt.savefig(f'cosmetic_figures/gp_structure_function_uv_{tag}_on_same_axis')
    plt.close()

    x_ray_gp_tao_plot = np.loadtxt(f'sf_gp_data/xray/{kernel}_{n_samples}_times')[2:]
    x_ray_gp_mean_structure_function = np.loadtxt(f'sf_gp_data/xray/mean_structure_function_{n_samples}_{kernel}')[2:]
    x_ray_gp_std_structure_function = np.loadtxt(f'sf_gp_data/xray/std_structure_function_{n_samples}_{kernel}')[2:]

    x_ray_tao_plot = np.loadtxt(f'sf_data/xray/times_most_recent')[2:]
    x_ray_mean_structure_function = np.loadtxt(f'sf_data/xray/structure_function_most_recent')[2:]
    x_ray_std_structure_function = np.loadtxt(f'sf_data/xray/errors_most_recent')[2:]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.scatter(x_ray_tao_plot, x_ray_mean_structure_function, s=10, marker='+', label='obs')
    ax2.scatter(x_ray_gp_tao_plot, x_ray_gp_mean_structure_function, s=10, marker='+', label='gp')
    ax1.errorbar(x_ray_tao_plot, x_ray_mean_structure_function, yerr=x_ray_std_structure_function, fmt='o', markersize=3, linewidth=1, label='obs')
    ax2.errorbar(x_ray_gp_tao_plot, x_ray_gp_mean_structure_function, yerr=x_ray_gp_std_structure_function, fmt='o', markersize=3, linewidth=1, label='gp')
    plt.xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.xlabel(r'$\tau$' + ' (days)')
    ax1.set_ylabel('SF')
    ax2.set_ylabel(f'GP-{kernel} SF')
    plt.xlim([10, 700])
    plt.subplots_adjust(left=0.17, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.suptitle('XRT Structure Function')
    plt.savefig(f'cosmetic_figures/gp_structure_function_xray_{tag}')
    plt.close()

    # plot on the same axis

    fig, ax = plt.subplots(1)
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    color = 'tab:blue'
    ax.set_ylabel('Observational SF', fontsize=12, color=color)
    ax.errorbar(x_ray_tao_plot, x_ray_mean_structure_function, yerr=x_ray_std_structure_function, fmt='o', markersize=3, linewidth=0.5, color=color, label='Observational')
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
    ax2.minorticks_off()
    plt.savefig(f'cosmetic_figures/gp_structure_function_xray_{tag}_on_same_axis_{int(scale_factor*100)}')
    plt.close()
