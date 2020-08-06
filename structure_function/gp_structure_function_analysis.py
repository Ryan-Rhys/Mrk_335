# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
This script performs a structure function analysis of the GP lightcurves for the UW2 and X-ray bands of Mrk-335.
"""

import numpy as np
from matplotlib import pyplot as plt

from structure_function_utils import compute_gp_structure_function

use_matern = True
n_samples = 50
resolution = 5.3


if __name__ == '__main__':

    if use_matern:
        tag = f'Matern_53_days_averaged_{n_samples}_samples'
        xray_handle = 'xray_samples_Matern_12_Kernel_noise_0.0001.txt'
        uv_handle = 'uv_samples_Matern_12_Kernel_noise_0.036907630522088355.txt'
    else:
        tag = f'RQ_53_days_averaged_{n_samples}_samples'
        xray_handle = 'xray_samples_Rational_Quadratic_Kernel_noise_0.0001.txt'
        uv_handle = 'uv_samples_Rational_Quadratic_Kernel_noise_0.036907630522088355.txt'

    time_grid = np.arange(54236, 58630, 1).reshape(-1, 1)

    # x_ray_structure_function_samples = []
    #
    # for iteration in range(n_samples):
    #
    #     x_ray_band_count_rates = np.loadtxt(fname=f'../gp_fit_real_data/samples/xray/{xray_handle}', skiprows=iteration, max_rows=1)
    #     x_ray_tao_plot, x_ray_structure_function_vals = compute_gp_structure_function(x_ray_band_count_rates, time_grid, resolution=resolution)
    #     x_ray_structure_function_samples.append(x_ray_structure_function_vals)
    #
    # x_ray_structure_function_samples = np.array(x_ray_structure_function_samples)
    # x_ray_mean_structure_function = np.mean(x_ray_structure_function_samples, axis=0)
    # x_ray_std_structure_function = np.std(x_ray_structure_function_samples, axis=0)
    #
    # fig, ax = plt.subplots(1)
    # plt.scatter(x_ray_tao_plot, x_ray_mean_structure_function, s=10, marker='+')
    # plt.errorbar(x_ray_tao_plot, x_ray_mean_structure_function, yerr=x_ray_std_structure_function, fmt='o', markersize=3, linewidth=1)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'$\tau$' + ' (days)')
    # plt.ylabel('SF')
    # plt.title('XRT Structure Function')
    # plt.savefig(f'figures/gp_structure_function_xray_{tag}')
    # plt.close()

    uv_structure_function_samples = []

    for iteration in range(n_samples):

        uv_band_count_rates = np.loadtxt(fname=f'../gp_fit_real_data/samples/uv/{uv_handle}', skiprows=iteration, max_rows=1)
        uv_tao_plot, uv_structure_function_vals = compute_gp_structure_function(uv_band_count_rates, time_grid, resolution=resolution)
        uv_structure_function_samples.append(uv_structure_function_vals)

    uv_structure_function_samples = np.array(uv_structure_function_samples)
    uv_mean_structure_function = np.mean(uv_structure_function_samples, axis=0)
    uv_std_structure_function = np.std(uv_structure_function_samples, axis=0)

    fig, ax = plt.subplots(1)
    plt.scatter(uv_tao_plot, uv_mean_structure_function, s=10, marker='+')
    plt.errorbar(uv_tao_plot, uv_mean_structure_function, yerr=uv_std_structure_function, fmt='o', markersize=3, linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\tau$' + ' (days)')
    plt.ylabel('SF')
    plt.title('UVW2 Structure Function')
    plt.savefig(f'figures/gp_structure_function_uv_{tag}')
    plt.close()
