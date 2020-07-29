# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains utilities for the mrk-335 simulation experiments.
"""

import pickle

import numpy as np


def load_sim_data(timings_file, gapped_count_rate_file, ground_truth_count_rate_file):
    """
    Load the simulation data. Return

    :param timings_file: The timings of the actual dataset
    :param gapped_count_rate_file: The count rate of the simulated light curves with gaps
    :param ground_truth_count_rate_file: The count rate of the full simulated light curves
    :return: timings, test_times, gapped_count_rates, ground_truth_count_rates
    """
    with open(timings_file, 'rb') as handle:
        timings = pickle.load(handle).reshape(-1, 1)

    test_times = np.arange(int(timings[0]), int(timings[-1]) + 1, 1).reshape(-1, 1)

    with open(gapped_count_rate_file, 'rb') as handle:
        # array of shape (NSIMS, timings.shape[0])
        gapped_count_rates = pickle.load(handle)

    with open(ground_truth_count_rate_file, 'rb') as handle:
        # array of shape (NSIMS, test_times.shape[0])
        ground_truth_count_rates = pickle.load(handle)

    return timings, test_times, gapped_count_rates, ground_truth_count_rates


def rss_func(mean, ground_truth):
    """
    Compute the Residual Sum of Squares between the GP predictions and the ground truth values of the simulated
    lightcurves.
    :param mean: GP predictive mean
    :param ground_truth: ground truth values of the simulated lightcurve
    :return: residual sum of squares
    """

    rss = np.sum((mean - ground_truth)**2)

    return rss
