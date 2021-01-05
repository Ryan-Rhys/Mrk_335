# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for generating histograms of the UVOT and XROT observations from Mrk-335. Utilities for plotting the
Empirical Cumulative Distribution Function (ECDF) and PP plots are also provided.
"""

import pickle

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF


def main():
    """
    Generate histograms, ECDFs and PP plots.
    """

    with open('../processed_data/uv/uv_band_count_rates.pickle', 'rb') as handle:
        uv_band_count_rates = np.sort(pickle.load(handle).reshape(-1, 1).squeeze())

    # plot the histogram
    plt.hist(uv_band_count_rates, bins=20, color="#00b764", alpha=0.5, density=True)
    density = gaussian_kde(uv_band_count_rates)
    xs = np.linspace(np.min(uv_band_count_rates), np.max(uv_band_count_rates), 200)
    plt.plot(xs, density(xs), color='k')
    plt.xticks(fontsize=12)
    plt.yticks([0, 1, 2], fontsize=12)
    #plt.xlabel('Swift Observed UVOT Magnitudes', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Density', fontsize=16, fontname='Times New Roman')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('figures/new_uv_histogram.png')
    plt.clf()

    # plot the empirical cdf (ecdf) for the UV magnitudes. Must take negative sign in order to reverse.
    ecdf_uv = ECDF(-uv_band_count_rates)
    plt.plot(ecdf_uv.x, ecdf_uv.y)
    plt.xticks(fontsize=12, labels=['13.8', '13.3', '12.8'],
               ticks=[-13.8, -13.3, -12.8])
    plt.yticks(fontsize=12)
    plt.xlabel('UVW2 Magnitudes', fontsize=16, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('figures/uv_ecdf')
    plt.clf()

    # plot the UV band probability plot for Gaussian distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Theoretical Quantiles', fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Ordered Values', fontsize=16, fontname='Times New Roman')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.title.set_visible(False)
    res = stats.probplot(-uv_band_count_rates, dist=stats.norm, plot=ax)
    ax.set_yticks(ticks=[-14.0, -13.8, -13.6, -13.4, -13.2, -13.0, -12.8, -12.6, -12.4])
    ax.set_yticklabels(labels=['14.0', '13.8', '13.6', '13.4', '13.2', '13.0', '12.8', '12.6', '12.4'])
    plt.tight_layout()
    plt.savefig('figures/uv_prob_plot')

    plt.clf()

    with open('../processed_data/xray/x_ray_band_count_rates.pickle', 'rb') as handle:
        xray_band_count_rates = np.sort(pickle.load(handle).reshape(-1, 1).squeeze())

        # We log the x-ray band count rates for hte histogram
        xray_band_count_rates = np.log(xray_band_count_rates)

    plt.hist(xray_band_count_rates, bins=20, color="#e65802", alpha=0.5, density=True)
    density = gaussian_kde(xray_band_count_rates)
    xs = np.linspace(np.min(xray_band_count_rates), np.max(xray_band_count_rates), 200)
    plt.plot(xs, density(xs), color='k')
    plt.xticks(fontsize=12)
    plt.yticks([0, 0.2, 0.4], fontsize=12)
    #plt.xlabel('Swift Observed X-ray Log Count Rates', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Density', fontsize=16, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('figures/new_xray_histogram')
    plt.clf()

    # plot the empirical cdf (ecdf) for the x-ray band count rate
    ecdf_xray = ECDF(xray_band_count_rates)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.plot(ecdf_xray.x, ecdf_xray.y)
    plt.xlabel('Log Count Rate', fontsize=16, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('figures/xray_ecdf')
    plt.clf()

    # plot the probability plot for Log-Gaussian Distribution (x-ray vals have been logged previously)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Theoretical Quantiles', fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Ordered Values', fontsize=16, fontname='Times New Roman')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.title.set_visible(False)
    res = stats.probplot(xray_band_count_rates, dist=stats.norm, plot=ax)
    plt.tight_layout()
    plt.savefig('figures/xray_prob_plot')

    plt.clf()

    print(stats.kstest(uv_band_count_rates, 'norm', args=(np.mean(uv_band_count_rates), np.std(uv_band_count_rates))))  # null vs alternative hypothesis for sample1. Dont reject equal distribution against alternative hypothesis: greater
    print(stats.kstest(np.log(xray_band_count_rates), 'norm', args=(np.mean(np.log(xray_band_count_rates)), np.std(np.log(xray_band_count_rates)))))
    print(stats.ks_2samp(uv_band_count_rates, uv_band_count_rates))


if __name__ == '__main__':

    main()
