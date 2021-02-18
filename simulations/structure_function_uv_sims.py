# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
Comparison of GP-interpolated UVW2 and true structure functions where the GP interpolated
structure functions are computed following the introduction of gaps into lightcurves.
"""

import numpy as np
from matplotlib import pyplot as plt

from simulation_utils import load_sim_data
from structure_function_utils import compute_gp_structure_function

resolution = 5.3
nsims = 1000  # number of simulated curves i.e length of gapped_file
kernel = 'Matern'  # ['Matern', 'RQ']

TIMINGS_FILE = '../processed_data/uv_simulations/uv_sim_times.pickle'
GAPPED_FILE = 'sim_curves/w2_lightcurves.dat'
GROUND_TRUTH_FILE = 'sim_curves/w2_lightcurves_no_gaps.dat'

if __name__ == '__main__':

    if kernel == 'Matern':
        tag = 'Matern_12'
    else:
        kernel = 'RQ'
        tag = 'Rational Quadratic'

    # Load the times for gap points, times for full curves, count rates for gap points and count rates for full curves
    # Matrix because second dimension corresponds to nsims.
    time, test_times, gapped_count_rates_matrix, ground_truth_count_rates_matrix = load_sim_data(TIMINGS_FILE,
                                                                                   GAPPED_FILE,
                                                                                   GROUND_TRUTH_FILE)

    for i in range(15):

        # file handle for GP lightcurve
        handle = f'SF_uv_samples_{tag} Kernel_iteration_{i}.txt'

        gapped_count_rates = np.reshape(gapped_count_rates_matrix[i, :], (-1, 1))
        count_rates = np.reshape(ground_truth_count_rates_matrix[i, :], (-1, 1))
        gp_count_rates = np.reshape(np.loadtxt(fname=f'SF_samples/uv/{handle}'), (-1, 1))

        gapped_tao_plot, gapped_structure_function_vals = compute_gp_structure_function(gapped_count_rates, time, resolution=resolution)
        ground_truth_tao_plot, ground_truth_structure_function_vals = compute_gp_structure_function(count_rates, test_times, resolution=resolution)
        gp_tao_plot, gp_structure_function_vals = compute_gp_structure_function(gp_count_rates, test_times, resolution=resolution)

        np.savetxt(f'saved_sf_values/uv/gapped_tao_plot_{i}.txt', gapped_tao_plot, fmt='%.2f')
        np.savetxt(f'saved_sf_values/uv/gapped_structure_function_vals_{i}.txt', gapped_structure_function_vals, fmt='%.2f')
        np.savetxt(f'saved_sf_values/uv/{kernel}_gp_tao_plot_{i}.txt', gp_tao_plot, fmt='%.2f')
        np.savetxt(f'saved_sf_values/uv/ground_truth_structure_function_vals_{i}.txt', ground_truth_structure_function_vals, fmt='%.2f')
        np.savetxt(f'saved_sf_values/uv/ground_truth_tao_plot_{i}.txt', ground_truth_tao_plot, fmt='%.2f')
        np.savetxt(f'saved_sf_values/uv/{kernel}_gp_structure_function_vals_{i}.txt', gp_structure_function_vals, fmt='%.2f')

        fig, ax = plt.subplots(1)
        plt.scatter(gapped_tao_plot, gapped_structure_function_vals, s=10, marker='+', label='Gapped')
        plt.scatter(ground_truth_tao_plot, ground_truth_structure_function_vals, s=10, marker='+', label='Ground Truth')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\tau$' + ' (days)')
        plt.ylabel('SF')
        plt.xlim([10, 700])
        plt.title('UVW2 Gapped Structure Function')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'SF_sims_figures/uv/gapped_structure_function_{i}')
        plt.close()

        fig, ax = plt.subplots(1)
        plt.scatter(gp_tao_plot, gp_structure_function_vals, s=10, marker='+', label='GP')
        plt.scatter(ground_truth_tao_plot, ground_truth_structure_function_vals, s=10, marker='+', label='Ground Truth')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\tau$' + ' (days)')
        plt.ylabel('SF')
        plt.xlim([10, 700])
        plt.title(f'UVW2 GP {kernel} Structure Function')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'SF_sims_figures/uv/gp_{kernel}_structure_function_{i}')
        plt.close()
