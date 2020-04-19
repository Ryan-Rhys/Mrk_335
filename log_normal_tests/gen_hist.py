# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for generating histograms of the UVOT and XROT observations from Mrk-335.
"""

import pickle

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF


pp_normal = False  # Whether to construct PP-plot for Gaussian or Log-Gaussian distributions

if __name__ == '__main__':

    with open('../processed_data/uv/uv_band_count_rates.pickle', 'rb') as handle:
        uv_band_count_rates = np.sort(pickle.load(handle).reshape(-1, 1).squeeze())

    # plot the histogram
    plt.hist(uv_band_count_rates, bins=20)
    plt.xlabel('UV Band Count Rates')
    plt.ylabel('Frequency')
    plt.title('Histogram of UV Band Count Rates')
    plt.savefig('test_figures/uv_histogram.png')
    plt.clf()

    # plot the empirical cdf (ecdf)
    ecdf_uv = ECDF(uv_band_count_rates)
    plt.plot(ecdf_uv.x, ecdf_uv.y)
    plt.xlabel('UV Band Count Rate')
    plt.title('Empirical Cumulative Distribution Function for the UV Band Count Rate')
    plt.savefig('test_figures/uv_ecdf')
    plt.clf()

    # plot the probability plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if pp_normal:
        res = stats.probplot(uv_band_count_rates, dist=stats.norm, plot=ax)
        ax.set_title("UV Band Probability Plot for Gaussian Distribution")
        plt.savefig('test_figures/uv_prob_plot')

    else:
        res = stats.probplot(np.log(uv_band_count_rates), dist=stats.norm, plot=ax)
        ax.set_title("UV Band Probability Plot for Log-Gaussian Distribution")
        plt.savefig('test_figures/uv_log_prob_plot')

    plt.clf()

    with open('../processed_data/xray/x_ray_band_count_rates.pickle', 'rb') as handle:
        xray_band_count_rates = np.sort(pickle.load(handle).reshape(-1, 1).squeeze())

    plt.hist(xray_band_count_rates, bins=20)
    plt.xlabel('X-ray Band Count Rates')
    plt.ylabel('Frequency')
    plt.title('Histogram of X-ray Band Count Rates')
    plt.savefig('test_figures/xray_histogram')
    plt.clf()

    # plot the empirical cdf (ecdf)
    ecdf_xray = ECDF(xray_band_count_rates)
    plt.plot(ecdf_xray.x, ecdf_xray.y)
    plt.xlabel('X-ray Band Count Rate')
    plt.title('Empirical Cumulative Distribution Function for the X-ray Band Count Rate')
    plt.savefig('test_figures/xray_ecdf')
    plt.clf()

    # plot the probability plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if pp_normal:

        res = stats.probplot(xray_band_count_rates, dist=stats.norm, plot=ax)
        ax.set_title("X-ray Band Probability Plot for Gaussian Distribution")
        plt.savefig('test_figures/xray_prob_plot')

    else:
        res = stats.probplot(np.log(xray_band_count_rates), dist=stats.norm, plot=ax)
        ax.set_title("X-ray Band Probability Plot for Log-Gaussian Distribution")
        plt.savefig('test_figures/xray_log_prob_plot')

    plt.clf()

    print(stats.kstest(uv_band_count_rates, 'norm', args=(np.mean(uv_band_count_rates), np.std(uv_band_count_rates))))  # null vs alternative hypothesis for sample1. Dont reject equal distribution against alternative hypothesis: greater
    print(stats.kstest(np.log(xray_band_count_rates), 'norm', args=(np.mean(np.log(xray_band_count_rates)), np.std(np.log(xray_band_count_rates)))))
    print(stats.ks_2samp(uv_band_count_rates, uv_band_count_rates))
    print('hi')
