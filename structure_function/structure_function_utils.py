"""
Utility functions for structure function analysis
"""

import numpy as np


def compute_resolution(time_grid):
    """
    Compute the resolution given the time grid at which measurements are made.

    :param time_grid: The time grid on which the sample is taken.
    :return delta: the resolution
    """

    count = len(time_grid)
    delta = 0
    for i in range(len(time_grid) - 1):
        diff = time_grid[i + 1] - time_grid[i]
        delta += diff

    delta = delta/count

    return delta


def compute_structure_function(sample, time_grid, errors):
    """
    Computes the structure function for a lightcurve.

    :param sample: The flux observations for the data
    :param time_grid: The time values
    :param errors: The experimental errors
    :return: tao_range, structure_function_vals, structure_function_errors
    """

    # Median temporal sampling from Gallo et al. (2018) https://arxiv.org/pdf/1805.00300.pdf used instead of delta as
    # the resolution

    median_sampling_rate = 5.3
    tao_min = 0
    tao_max = 600
    tao_range = np.arange(tao_min, tao_max, median_sampling_rate)
    structure_function_vals = np.zeros((len(tao_range) - 1, 1))  # We pre-allocate the structure function value array.
    structure_function_errors = np.zeros((len(tao_range) - 1, 1))  # We pre-allocate the structure function uncertainty array
    tao_plot = []  # For plotting the structure function.

    # We iterate over each bin
    for i in range(len(tao_range) - 1):

        bin_start_val = tao_range[i]
        bin_end_val = tao_range[i + 1]
        structure_function_pairs = []
        bin_errors = []

        # For a given bin we take all pairs of observations separated by a tao in the bin.

        for time in time_grid:

            time_index = list(time_grid).index(time)
            running_index = time_index + 1

            while running_index < len(time_grid):

                time_difference = time_grid[running_index] - time_grid[time_index]

                if time_difference >= bin_start_val and time_difference <= bin_end_val:

                    structure_function_val = np.square(sample[running_index] - sample[time_index])
                    structure_function_pairs.append(structure_function_val)
                    bin_errors.append(errors[i])
                    bin_errors.append(errors[i + 1])

                running_index += 1

        if len(structure_function_pairs) < 6:
            continue
        else:

            tao_plot.append((bin_end_val + bin_start_val) / 2)  # We append the bin to the plot if it has more than 6 pairs

            mean_noise_variance = np.std(bin_errors)**2  # Compute the mean noise variance in the bin

            structure_function_average = np.mean(structure_function_pairs)
            structure_function_vals[i] = structure_function_average - 2*mean_noise_variance
            structure_function_error = np.std(structure_function_pairs)/(np.sqrt(len(structure_function_pairs)/2.0))
            structure_function_errors[i] = structure_function_error

    return np.array(tao_plot), structure_function_vals, structure_function_errors


def compute_gp_structure_function(sample, time_grid, resolution):
    """
    Computes the structure function for a GP lightcurve

    :param sample: The GP sample
    :param time_grid: The time grid on which the sample is taken.
    :param resolution: The resolution  at which to sample the lightcurve
    :return: tao_range, structure_function_vals
    """

    tao_min = 0
    tao_max = 600
    tao_range = np.arange(tao_min, tao_max, resolution)
    structure_function_vals = np.zeros((len(tao_range) - 1, 1))  # We pre-allocate the structure function value array.
    tao_plot = []  # For plotting the structure function.

    # We iterate over each bin
    for i in range(len(tao_range) - 1):

        import time
        start = time.time()

        print(i)

        bin_start_val = tao_range[i]
        bin_end_val = tao_range[i + 1]
        structure_function_pairs = []

        # For a given bin we take all pairs of observations separated by a tao in the bin.

        for times in time_grid:

            time_index = list(time_grid).index(times)
            running_index = time_index + 1

            while running_index < len(time_grid):

                time_difference = time_grid[running_index] - time_grid[time_index]

                if time_difference >= bin_start_val and time_difference <= bin_end_val:

                    structure_function_val = np.square(sample[running_index] - sample[time_index])
                    structure_function_pairs.append(structure_function_val)

                # Can induce a speed-up

                elif time_difference > bin_end_val:
                    break

                running_index += 1

        if len(structure_function_pairs) < 6:
            continue
        else:

            tao_plot.append((bin_end_val + bin_start_val) / 2)  # We append the bin to the plot if it has more than 6 pairs

            structure_function_average = np.mean(structure_function_pairs)
            structure_function_vals[i] = structure_function_average

        end = time.time()
        print("Iteration time is " + str(end - start) + ' seconds')

    return np.array(tao_plot), structure_function_vals
