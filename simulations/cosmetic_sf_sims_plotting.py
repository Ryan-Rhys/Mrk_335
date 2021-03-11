# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
Cosmetic plotting for the structure function simulations.
"""

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

kernel = 'Matern'  # ['Matern', 'RQ']
i = 5  # simulation number to plot

if __name__ == '__main__':

    gapped_tao_plot = np.loadtxt(f'saved_sf_values/xray/_gapped_tao_plot_{i}.txt')
    gapped_structure_function_vals = np.loadtxt(f'saved_sf_values/xray/gapped_structure_function_vals_{i}.txt')
    gp_tao_plot = np.loadtxt(f'saved_sf_values/xray/{kernel}_gp_tao_plot_{i}.txt')
    ground_truth_structure_function_vals = np.loadtxt(f'saved_sf_values/xray/ground_truth_structure_function_vals_{i}.txt')
    ground_truth_tao_plot = np.loadtxt(f'saved_sf_values/xray/ground_truth_tao_plot_{i}.txt')
    gp_structure_function_vals = np.loadtxt(f'saved_sf_values/xray/{kernel}_gp_structure_function_vals_{i}.txt')

    # plot on the same axis

    fig, ax = plt.subplots(1)
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    color = 'tab:red'
    ax.set_ylabel('Observational SF', fontsize=12, color=color)
    ax.scatter(gapped_tao_plot, gapped_structure_function_vals, s=10, marker='+', color=color, label='Observational')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([0.01, 0.5])
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.set_yticks([0.6, 1, 2, 3, 4])
    plt.xticks([0.1, 1])
    ax.minorticks_off()
    color = 'tab:blue'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Ground Truth SF', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax.scatter(ground_truth_tao_plot, ground_truth_structure_function_vals, s=10, color=color, marker='+', label='Ground Truth')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.01, 0.5])
    # ax2.set_yticks([1])
    plt.xlim([10, 700])
    plt.tight_layout()
    fig.legend(loc=4, bbox_to_anchor=[0.275, 0.75, 0.15, 0])
    ax2.yaxis.set_minor_locator(mticker.NullLocator())  # set y-axis tick labels off only. Found by looking at the API for the minorticks_off() method.
    plt.savefig(f'cosmetic_figures/ground_truth_function_xray_on_same_axis_')
    plt.close()

    # Plot GP structure function against ground truth.

    if kernel == 'Matern':
        scale_factor = 0.1
    else:
        scale_factor = 1.35

    fig, ax = plt.subplots(1)
    plt.xlabel(r'$\tau$' + '(days)', fontsize=12)
    color = 'tab:red'
    ax.set_ylabel('GP SF', fontsize=12, color=color)
    ax.scatter(gp_tao_plot, scale_factor*gp_structure_function_vals, s=10, marker='+', color=color, label='Gaussian Process')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([0.01, 0.5])
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.set_yticks([0.6, 1, 2, 3, 4])
    plt.xticks([0.1, 1])
    ax.minorticks_off()
    color = 'tab:blue'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Ground Truth SF', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax.scatter(ground_truth_tao_plot, ground_truth_structure_function_vals, s=10, color=color, marker='+', label='Ground Truth')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.01, 0.5])
    # ax2.set_yticks([1])
    plt.xlim([10, 700])
    plt.tight_layout()
    fig.legend(loc=4, bbox_to_anchor=[0.275, 0.75, 0.15, 0])
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    plt.savefig(f'cosmetic_figures/gp_function_xray_{kernel}_on_same_axis_{int(scale_factor * 100)}')
    plt.close()