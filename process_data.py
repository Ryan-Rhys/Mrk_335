# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script processes the light curve .dat file for Mrk 335. mkn335_xrt_uvot_lc.dat contains the UVW2 data in
magnitudes whilst mkn335_xrt_w2_lc.dat contains the UVW2 data in count rates. UVW2 Count rate data goes up to
59234.47 days whereas magnitude data goes up to 58626.2199 days
"""

import pickle

from matplotlib import pyplot as plt
import numpy as np

f_plot = True  # Whether to plot the data on processing
f_plot_errors = True  # Whether to plot the errors on processing

if __name__ == '__main__':

    with open('raw_data/mkn335_xrt_w2_lc.dat', 'rb') as file:

        # Observational data for the UVW2 band in count rates.

        uv_cr_times = []
        uv_band_count_rates = []
        uv_band_count_errors = []

        i = 1  # used simply to skip the first line

        for line in file:
            if i > 1:

                # processing from bit format

                data = line.split()

                # Skip NA values

                if data[-2].decode('utf-8') == 'NA' or data[-1].decode('utf-8') == 'NA':
                    continue

                # Only use count rates up to 58626.2199 days.

                if float(data[0].decode('utf-8')) > 58626.2199:
                    break

                time = float(data[0].decode("utf-8"))
                uv_band_count_rate = float(data[-2].decode('utf-8'))
                uv_band_count_error = float(data[-1].decode('utf-8'))

                if uv_band_count_rate > 0:
                    uv_cr_times.append(time)
                    uv_band_count_rates.append(uv_band_count_rate)
                    uv_band_count_errors.append(uv_band_count_error)

            i += 1

    with open('raw_data/mkn335_xrt_uvot_lc.dat', 'rb') as file:

        # Observational data for the x-ray and UV bands of Mrk-335 including measurement errors.

        x_ray_times = []
        x_ray_band_count_rates = []
        x_ray_band_count_errors = []

        uv_mag_times = []
        uv_band_magnitudes = []
        uv_band_magnitudes_errors = []

        i = 1  # Used simply to skip the first line.

        for line in file:
            if i > 1:

                # Processing from bit format

                data = line.split()
                time = float(data[0].decode("utf-8"))
                x_ray_band_count = float(data[1].decode("utf-8"))
                x_ray_band_count_error = float(data[2].decode("utf-8"))
                uv_band_magnitude = float(data[-2].decode("utf-8"))
                uv_band_magnitude_error = float(data[-1].decode("utf-8"))

                if x_ray_band_count > 0:
                    x_ray_times.append(time)
                    x_ray_band_count_rates.append(x_ray_band_count)
                    x_ray_band_count_errors.append(x_ray_band_count_error)
                if uv_band_magnitude > 0:
                    uv_mag_times.append(time)
                    uv_band_magnitudes.append(uv_band_magnitude)
                    uv_band_magnitudes_errors.append(uv_band_magnitude_error)

            i += 1

    # typecast to float64.

    x_ray_times = np.array(x_ray_times, dtype=np.float64)
    x_ray_band_count_rates = np.array(x_ray_band_count_rates, dtype=np.float64)
    x_ray_band_count_errors = np.array(x_ray_band_count_errors, dtype=np.float64)
    uv_cr_times = np.array(uv_cr_times, dtype=np.float64)
    uv_mag_times = np.array(uv_mag_times, dtype=np.float64)
    uv_band_count_rates = np.array(uv_band_count_rates, dtype=np.float64)
    uv_band_count_errors = np.array(uv_band_count_errors, dtype=np.float64)
    uv_band_magnitudes = np.array(uv_band_magnitudes, dtype=np.float64)
    uv_band_magnitudes_errors = np.array(uv_band_magnitudes_errors, dtype=np.float64)

    # Visualise the data

    if f_plot:
        plt.scatter(x_ray_times, x_ray_band_count_rates, s=15, marker='.')
        plt.errorbar(x_ray_times, x_ray_band_count_rates, yerr=x_ray_band_count_errors, elinewidth=0.1, capsize=0.5, barsabove=True, capthick=1, linestyle="None", ecolor='r')
        plt.xlabel('Time')
        plt.ylabel('X-ray Band Count Rate')
        plt.title('X-ray Data for Mrk 335')
        plt.savefig('plots/xray_data_for_mrk_335')
        plt.show()
        plt.clf()
        plt.scatter(uv_cr_times, uv_band_count_rates, s=15, marker='.')
        plt.errorbar(uv_cr_times, uv_band_count_rates, yerr=uv_band_count_errors, elinewidth=0.1, capsize=0.5, barsabove=True, capthick=1, linestyle="None", ecolor='r')
        plt.xlabel('Time')
        plt.ylabel('UVW2 Band Count Rates')
        plt.title('UVW2 Data for Mrk 335')
        plt.savefig('plots/uv_cr_data_for_mrk_335')
        plt.show()
        plt.clf()
        plt.scatter(uv_mag_times, uv_band_magnitudes, s=15, marker='.')
        plt.errorbar(uv_mag_times, uv_band_magnitudes, yerr=uv_band_magnitudes_errors, elinewidth=0.1, capsize=0.5, barsabove=True, capthick=1, linestyle="None", ecolor='r')
        plt.xlabel('Time')
        plt.ylabel('UVW2 Band Magnitudes')
        plt.title('UVW2 Data for Mrk 335')
        plt.savefig('plots/uv_mag_data_for_mrk_335')
        plt.show()

    if f_plot_errors:
        plt.scatter(x_ray_times, x_ray_band_count_errors, s=10)
        plt.xlabel('Time')
        plt.ylabel('X-ray Band Count Errors')
        plt.title('X-ray Errors for Mrk 335')
        plt.savefig('plots/xray_errors_for_mrk_335')
        plt.show()
        plt.clf()
        plt.scatter(uv_cr_times, uv_band_count_errors, s=10)
        plt.xlabel('Time')
        plt.ylabel('UVW2 Band Count Errors')
        plt.title('UVW2 Count Rate Errors for Mrk 335')
        plt.savefig('plots/uv_cr_errors_for_mrk_335')
        plt.show()
        plt.clf()
        plt.scatter(uv_mag_times, uv_band_magnitudes_errors, s=10)
        plt.xlabel('Time')
        plt.ylabel('UVW2 Magnitudes Errors')
        plt.title('UVW2 Magnitudes Errors for Mrk 335')
        plt.savefig('plots/uv_mag_errors_for_mrk_335')
        plt.show()

    # Save the data in the processed_data folder.

    with open('processed_data/xray/x_ray_times.pickle', 'wb') as handle:
        pickle.dump(x_ray_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/xray/x_ray_band_count_rates.pickle', 'wb') as handle:
        pickle.dump(x_ray_band_count_rates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/xray/x_ray_band_count_errors.pickle', 'wb') as handle:
        pickle.dump(x_ray_band_count_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/uv/uv_fl_times.pickle', 'wb') as handle:
        pickle.dump(uv_cr_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/uv/uv_mag_times.pickle', 'wb') as handle:
        pickle.dump(uv_mag_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/uv/uv_band_flux.pickle', 'wb') as handle:
        pickle.dump(uv_band_count_rates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/uv/uv_band_flux_errors.pickle', 'wb') as handle:
        pickle.dump(uv_band_count_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/uv/uv_band_magnitudes.pickle', 'wb') as handle:
        pickle.dump(uv_band_magnitudes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_data/uv/uv_band_magnitudes_errors.pickle', 'wb') as handle:
        pickle.dump(uv_band_magnitudes_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
