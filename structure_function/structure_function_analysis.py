# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
This script performs a structure function analysis of the UW2 and X-ray bands of Mrk-335.
"""

import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from structure_function_utils import compute_structure_function


repeat_gallo = False  # Use the data from Gallo et al. 2018 https://arxiv.org/pdf/1805.00300.pdf


if __name__ == '__main__':

    if repeat_gallo:
        x_ray_end_of_range = 402
        uv_end_of_range = 391
    else:
        x_ray_end_of_range = 509
        uv_end_of_range = 498

    with open('../processed_data/xray/x_ray_times.pickle', 'rb') as handle:
        x_ray_time = pickle.load(handle).reshape(-1, 1)[0:x_ray_end_of_range]
    with open('../processed_data/xray/x_ray_band_count_rates.pickle', 'rb') as handle:
        x_ray_band_count_rates = pickle.load(handle).reshape(-1, 1)[0:x_ray_end_of_range]
    with open('../processed_data/xray/x_ray_band_count_errors.pickle', 'rb') as handle:
        x_ray_band_count_errors = pickle.load(handle).reshape(-1, 1)[0:x_ray_end_of_range]

    x_ray_lightcurve_variance = np.mean((x_ray_band_count_rates - np.mean(x_ray_band_count_rates))**2)
    x_ray_mean_noise_variance = np.mean((x_ray_band_count_errors - np.mean(x_ray_band_count_errors))**2)

    x_ray_tao_plot, x_ray_structure_function_vals, x_ray_structure_function_errors = compute_structure_function(x_ray_band_count_rates, x_ray_time, x_ray_band_count_errors)

    x_ray_structure_function_vals = x_ray_structure_function_vals/(x_ray_lightcurve_variance)
    x_ray_structure_function_errors = x_ray_structure_function_errors/(x_ray_lightcurve_variance)

    fig, ax = plt.subplots(1)
    plt.scatter(x_ray_tao_plot, x_ray_structure_function_vals, s=10, marker='+')
    plt.errorbar(x_ray_tao_plot, x_ray_structure_function_vals, yerr=x_ray_structure_function_errors, fmt='o', markersize=3, linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, 700)
    plt.ylim(0.5, 5)
    plt.xticks([10, 100])
    plt.yticks([1, 2, 3, 4])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # no decimal places
    plt.xlabel(r'$\tau$' + ' (days)')
    plt.ylabel('SF')
    plt.title('XRT Structure Function')

    if repeat_gallo:
        plt.savefig('figures/Structure_function_xray_from_Gallo_et_al_2018')
    else:
        plt.savefig('figures/Structure_function_xray')
    plt.close()

    with open('../processed_data/uv/uv_times.pickle', 'rb') as handle:
        uv_time = pickle.load(handle).reshape(-1, 1)[0:uv_end_of_range]
    with open('../processed_data/uv/uv_band_count_rates.pickle', 'rb') as handle:
        uv_band_count_rates = pickle.load(handle).reshape(-1, 1)[0:uv_end_of_range]
    with open('../processed_data/uv/uv_band_count_errors.pickle', 'rb') as handle:
        uv_band_count_errors = pickle.load(handle).reshape(-1, 1)[0:uv_end_of_range]

    uv_lightcurve_variance = np.mean((uv_band_count_rates - np.mean(uv_band_count_rates))**2)
    uv_mean_noise_variance = np.mean((uv_band_count_errors - np.mean(uv_band_count_errors))**2)

    uv_tao_plot, uv_structure_function_vals, uv_structure_function_errors = compute_structure_function(uv_band_count_rates, uv_time, uv_band_count_errors)

    uv_structure_function_vals = uv_structure_function_vals/(uv_lightcurve_variance)
    uv_structure_function_errors = uv_structure_function_errors/(uv_lightcurve_variance)

    fig, ax = plt.subplots(1)
    plt.scatter(uv_tao_plot, uv_structure_function_vals, s=10, marker='+')
    plt.errorbar(uv_tao_plot, uv_structure_function_vals, yerr=uv_structure_function_errors, fmt='o', markersize=3, linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, 700)
    plt.ylim(0.2, 5)
    plt.xticks([10, 100])
    plt.yticks([1, 2, 3, 4])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # no decimal places
    plt.xlabel(r'$\tau$' + ' (days)')
    plt.ylabel('SF')
    plt.title('UVW2 Structure Function')

    if repeat_gallo:
        plt.savefig('figures/Structure_function_uv_from_Gallo_et_al_2018')
    else:
        plt.savefig('figures/Structure_function_uv')
    plt.close()
