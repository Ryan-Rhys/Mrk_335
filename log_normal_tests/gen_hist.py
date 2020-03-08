# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
Script for generating histograms of the UVOT and XROT observations from Mrk-335.
"""

import pickle

from matplotlib import pyplot as plt
from scipy import stats

if __name__ == '__main__':

    with open('../processed_data/uv/uv_band_count_rates.pickle', 'rb') as handle:
        uv_band_count_rates = pickle.load(handle).reshape(-1, 1)

    plt.hist(uv_band_count_rates, bins=20)
    plt.xlabel('UV Band Count Rates')
    plt.ylabel('Frequency')
    plt.title('Histogram of UV Band Count Rates')
    plt.savefig('test_figures/uv_histogram.png')
    plt.clf()

    with open('../processed_data/xray/x_ray_band_count_rates.pickle', 'rb') as handle:
        xray_band_count_rates = pickle.load(handle).reshape(-1, 1)

    plt.hist(xray_band_count_rates, bins=20)
    plt.xlabel('X-ray Band Count Rates')
    plt.ylabel('Frequency')
    plt.title('Histogram of X-ray Band Count Rates')
    plt.savefig('test_figures/xray_histogram')

    stats.kstest(uv_band_count_rates, 'norm', alternative='greater')  # null vs alternative hypothesis for sample1. Dont reject equal distribution against alternative hypothesis: greater
    stats.kstest(xray_band_count_rates, 'norm', alternative='greater')
