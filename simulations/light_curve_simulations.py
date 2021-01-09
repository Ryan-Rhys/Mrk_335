# Authors: Douglas Buisson, Ryan-Rhys Griffiths
"""
This script generates simulated light curves. Timmer and Konig's algorithm generates Gaussian flux distributions.
"""

import pickle

from astropy.io import ascii
import numpy as np
from matplotlib import pyplot as p

from fourier_methods import psd
from ts_gen import ts_gen

RAW_DATA_PATH = '../raw_data/mkn335_xrt_uvot_lc.dat'
PROCESSED_DATA_PATH = '../processed_data/'

empirical_correction = False


if __name__ == '__main__':

    if empirical_correction:
        tag = 'empirical_correction_'
    else:
        tag = ''

    # Save the gapped and full lightcurves in these files.

    gapped_output_xray_dat = 'sim_curves/xray_lightcurves_new.dat'
    full_output_xray_dat = 'sim_curves/xray_lightcurves_no_gaps_new.dat'

    gapped_output_uv_dat = f'sim_curves/{tag}w2_lightcurves.dat'
    full_output_uv_dat = f'sim_curves/{tag}w2_lightcurves_no_gaps.dat'

    # Read data

    a = ascii.read(RAW_DATA_PATH, data_start=1)

    t = np.array(a.columns[0])  # timings
    xr = np.array(a.columns[1])  # x-ray band count rates
    wr = np.array(a.columns[15])  # w2 band magnitudes

    # Filter the flux values.

    goodx = np.where(xr > -1e-16)[0]
    goodw = np.where(wr > 3)[0]

    xrg = xr[goodx]  # good x-ray count rates
    xtg = t[goodx]  # are the timings for the simulations

    # save the filtered timings
    with open(PROCESSED_DATA_PATH + 'xray_simulations/x_ray_sim_times.pickle', 'wb') as handle:
        pickle.dump(xtg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # X-ray PSD

    splits = np.where(xtg[1:] - xtg[:-1] > 16)[0] + 1
    tregs = np.array([])
    xregs = np.array([])
    prev_split = 0
    for split in splits:
        if split - prev_split > 10:
            tsec = xtg[prev_split:split]
            xsec = xrg[prev_split:split]
            treg = np.arange(tsec[0], tsec[-1], 4)
            xreg = np.interp(treg, tsec, xsec)
            tregs = np.concatenate([tregs, treg])
            xregs = np.concatenate([xregs, xreg])
            #p.psd(xreg, Fs=0.25, color='k')
            #p.plot(treg, xreg, 'k')
            prev_split = split

    xf, xp, xe = psd(tregs, np.log(xregs))  # calculate psd in log space

    # W2 PSD

    # Filter the UVW2 magnitudes

    wrg = wr[goodw]
    wtg = t[goodw]

    # Convert the UVW2 magnitudes to count rates following equation 2 of:
    # https://swift.gsfc.nasa.gov/results/publist/Breeveld_SPIE_5898_379_2005.pdf

    Z_pt = 17.77

    wrg = 10**((Z_pt - wrg)/2.5)  # Range is 33 - 110 ct s^-1

    if empirical_correction:

        # Equations 3,4 and 5 of this document: https://swift.gsfc.nasa.gov/results/publist/Breeveld_SPIE_5898_379_2005.pdf

        ft = 0.011088
        df = 0.0155844

        c_theory = -np.log(1 - wrg*ft)/(ft*(1 - df))

        x_var = wrg*ft
        f_of_x = 1.0 + 0.2966*x_var - 0.492*x_var**2 - 0.4183*x_var**3 + 0.2668*x_var**4

        wrg = c_theory*f_of_x

    # save the filtered timings
    with open(PROCESSED_DATA_PATH + '/uv_simulations/uv_sim_times.pickle', 'wb') as handle:
        pickle.dump(wtg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    splits = np.where(wtg[1:] - wtg[:-1] > 16)[0]+1
    tregs = np.array([])
    wregs = np.array([])
    prev_split = 0
    for split in splits:
        if split - prev_split > 10:
            tsec = wtg[prev_split:split]
            wsec = wrg[prev_split:split]
            treg = np.arange(tsec[0], tsec[-1], 4)
            wreg = np.interp(treg, tsec, wsec)
            tregs = np.concatenate([tregs, treg])
            wregs = np.concatenate([wregs, wreg])
            # p.psd(wreg, Fs=0.25, color='b')
            # p.plot(treg, wreg, 'b')
            prev_split = split

    wf, wp, we = psd(tregs, wregs)

    #p.xscale('log')
    #p.show()

    # Fit PSDs

    from scipy import optimize

    def linear_res(mc, freq, pow):
        return (mc[0]*freq + mc[1]) - pow

    xresult = optimize.least_squares(linear_res, [2, -3], args=(np.log(xf), np.log(xp)))
    wresult = optimize.least_squares(linear_res, [2, -3], args=(np.log(wf), np.log(wp)))

    # Plot PSDs

    p.errorbar(wf/86400, wf*wp, wf*we, fmt='b')
    p.errorbar(xf/86400, xf*xp, xf*xe, fmt='k')
    p.xscale('log')
    p.yscale('log')
    p.xlabel('Frequency (Hz)')
    p.ylabel(r'Power ($fP(f)$)')
    #p.show()

    # Simulate lightcurves

    # X-ray

    nsims = 1000

    # Shape of power spectrum
    freq = np.array([1e-15, 1e5])
    pow = freq**xresult.x[0]*np.exp(xresult.x[1])

    # Size/scale of real data
    mean_rate = np.mean(np.log(xrg))
    var_rate = np.std(np.log(xrg))

    # Time of each observation (nearest day/10)
    time = np.round(xtg*10)/10
    time -= min(time)

    # Exposure of data
    exposure = 1000.

    # Work with timesteps of tenths of days

    xlightcurves = np.empty([nsims, len(xrg)])  # preallocate the array of count rates
    xlightcurves_no_gaps = np.empty([nsims, 4390])  # obtained by checking size of rates

    for i in range(nsims):

        rate_inc = (ts_gen(2**19, freq=freq, pow=pow, loginterp=1))[500:(50000+500)]
        rates_all = np.exp((rate_inc - np.mean(rate_inc)) / np.std(rate_inc)*var_rate + mean_rate)

        gap_rates = rates_all[(time*10).astype(int)]
        np.random.seed(i)
        gap_rateso = np.float_(np.random.poisson(gap_rates * exposure)) / exposure
        xlightcurves[i, :] = gap_rates

        full_rates = rates_all[:(time[-1].astype(int) * 10 + 1)]
        np.random.seed(i)
        full_rates = np.float_(np.random.poisson(full_rates * exposure)) / exposure
        xlightcurves_no_gaps[i, :] = full_rates[:43891: 10]  # output in units of band count rate

        print(i)

    gap_file = open(gapped_output_xray_dat, 'wb')
    pickle.dump(xlightcurves, gap_file)
    gap_file.close()
    #ascii.write(xlightcurves, output_xray_dat)
    no_gap_file = open(full_output_xray_dat, 'wb')
    pickle.dump(xlightcurves_no_gaps, no_gap_file)
    no_gap_file.close()
    #ascii.write(xlightcurves_no_gaps, output_xray_dat)

    # W2 band

    # Shape of power spectrum
    freq = np.array([1e-15, 1e5])
    pow = freq**wresult.x[0]*np.exp(wresult.x[1])

    # Size/scale of real data
    mean_rate = np.mean(np.log(wrg))
    var_rate = np.std(np.log(wrg))

    # Time of each observation (nearest day/10)
    time = np.round(wtg*10)/10
    time -= min(time)

    # Exposure of data
    exposure = 1000.

    # Work with timesteps of tenths of days

    wlightcurves = np.empty([nsims, len(wrg)])
    wlightcurves_no_gaps = np.empty([nsims, 4390])

    for i in range(nsims):
        rate_inc = (ts_gen(2**19, freq=freq, pow=pow, loginterp=1))[500:(50000+500)]
        rates_all = np.exp((rate_inc - np.mean(rate_inc)) / np.std(rate_inc)*var_rate + mean_rate)

        rates = rates_all[(time*10).astype(int)]
        np.random.seed(i)
        rates = np.float_(np.random.poisson(rates * exposure)) / exposure
        wlightcurves[i, :] = rates

        rates = rates_all[:(time[-1].astype(int) * 10 + 1)]
        np.random.seed(i)
        rates = np.float_(np.random.poisson(rates * exposure)) / exposure
        wlightcurves_no_gaps[i, :] = rates[:43891: 10]  # every day, units of count rate, 4390 values

        print(i)

    gap_file = open(gapped_output_uv_dat, 'wb')

    pickle.dump(wlightcurves, gap_file)
    gap_file.close()
    #ascii.write(wlightcurves, output_uv_dat)
    # we use pickle because ascii can't deal with large files
    no_gap_file = open(full_output_uv_dat, 'wb')
    pickle.dump(wlightcurves_no_gaps, no_gap_file)
    no_gap_file.close()
