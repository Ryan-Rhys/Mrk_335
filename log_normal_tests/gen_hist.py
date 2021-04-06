# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for generating histograms of the UVOT and XROT observations from Mrk-335. Utilities for plotting the
Empirical Cumulative Distribution Function (ECDF) and PP plots are also provided in addition to Kolmogorov-Smirnov
tests.
"""

import pickle

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF

f_mag = False  # If true, plot UVW2 in magnitudes instead of flux


def main():
    """
    Generate histograms, ECDFs and PP plots and run Kolmogorov-Smirnov tests.
    """

    if f_mag:
        with open('../processed_data/uv/uv_band_magnitudes.pickle', 'rb') as handle:
            uv_band = np.sort(pickle.load(handle).reshape(-1, 1).squeeze())
    else:
        with open('../processed_data/uv/uv_band_flux.pickle', 'rb') as handle:
            uv_band = np.sort(pickle.load(handle).reshape(-1, 1).squeeze())

    # plot the histogram. Flux values multiplied by 1e14 for scaling.
    plt.hist(uv_band*1e14, bins=20, color="#00b764", alpha=0.5, density=True)
    density = gaussian_kde(uv_band*1e14)
    xs = np.linspace(np.min(uv_band*1e14), np.max(uv_band*1e14), 200)
    plt.plot(xs, density(xs), color='k')
    plt.xticks(fontsize=12)
    if f_mag:
        plt.yticks([0, 1, 2], fontsize=12)
    else:
        plt.yticks([0.1, 0.2, 0.3])
    plt.ylabel('Density', fontsize=16, fontname='Times New Roman')
    if f_mag:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'figures/new_uv_histogram_mags_is_{f_mag}.png')
    plt.clf()

    # plot the empirical cdf (ecdf) for the UV magnitudes or count rates. Must take negative sign in order to reverse
    # for magnitude units.
    if f_mag:
        ecdf_uv = ECDF(-uv_band)
    else:
        ecdf_uv = ECDF(uv_band)
        gaussian_samples = np.std(uv_band) * np.random.randn(5000) + np.mean(uv_band)
        cdf_gaussian_uv = ECDF(gaussian_samples)
    plt.plot(ecdf_uv.x, ecdf_uv.y, label='ECDF', linewidth='3')
    plt.plot(cdf_gaussian_uv.x, cdf_gaussian_uv.y, '--', color='r', label='Gaussian CDF')
    if f_mag:
        plt.xticks(fontsize=12, labels=['13.8', '13.3', '12.8'],
                   ticks=[-13.8, -13.3, -12.8])
    plt.yticks(fontsize=12)
    if f_mag:
        plt.xlabel('UVW2 Magnitudes', fontsize=16, fontname='Times New Roman')
    else:
        plt.xlabel('UVW2 Flux', fontsize=16, fontname='Times New Roman')
        plt.legend(fontsize=14)
        plt.xlim(0.3e-13, 0.95e-13)
    plt.tight_layout()
    plt.savefig(f'figures/uv_ecdf_mags_is_{f_mag}.png')
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
    if f_mag:
        res = stats.probplot(-uv_band, dist=stats.norm, plot=ax)
    else:
        res = stats.probplot(uv_band, dist=stats.norm, plot=ax)
    if f_mag:
        ax.set_yticks(ticks=[-14.0, -13.8, -13.6, -13.4, -13.2, -13.0, -12.8, -12.6, -12.4])
        ax.set_yticklabels(labels=['14.0', '13.8', '13.6', '13.4', '13.2', '13.0', '12.8', '12.6', '12.4'])
    else:
        ax.set_yticks(ticks=[3e-14, 5e-14, 7e-14, 9e-14])
        ax.set_yticklabels(labels=['3e-14', '5e-14', '7e-14', '9e-14'])
    plt.tight_layout()
    plt.savefig(f'figures/uv_prob_plot_mags_is_{f_mag}.png')

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
    plt.ylabel('Density', fontsize=16, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('figures/new_xray_histogram')
    plt.clf()

    # plot the empirical cdf (ecdf) for the x-ray band count rate
    ecdf_xray = ECDF(xray_band_count_rates)
    gaussian_cdf_xray = ECDF(np.std(xray_band_count_rates) * np.random.randn(5000) + np.mean(xray_band_count_rates))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.plot(ecdf_xray.x, ecdf_xray.y, linewidth='3', label='ECDF')
    plt.plot(gaussian_cdf_xray.x, gaussian_cdf_xray.y, '--', color='r', label='Gaussian CDF')
    plt.xlabel('Log Count Rate', fontsize=16, fontname='Times New Roman')
    plt.legend(fontsize='14')
    plt.xlim(-4, 1)
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

    print(stats.kstest(uv_band, 'norm', args=(np.mean(uv_band), np.std(uv_band))))  # null vs alternative hypothesis for sample1. Dont reject equal distribution against alternative hypothesis: greater
    print(stats.kstest(xray_band_count_rates, 'norm', args=(np.mean(xray_band_count_rates), np.std(xray_band_count_rates))))
    print(stats.ks_2samp(uv_band, uv_band))


if __name__ == '__main__':

    main()
