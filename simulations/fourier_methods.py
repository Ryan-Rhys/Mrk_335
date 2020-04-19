#! /usr/bin/python
# Authors: Douglas Buisson and Ryan-Rhys Griffiths
"""
Fourier methods.
"""

import numpy as np


#######################################################################

### PSDs

#############################################
def psd(time, rate, binsize=None, binfac=20, sec_length=None, min_length=10, gap_threshold=1.5, verbose=False):

    if binsize == None:
        binsize = min(time[1:] - time[0:-1])

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    freqall = np.array([])
    powerall = np.array([])
    n_pgs = 0

    for j in range(n_sections):

        t_sec = time[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]
        r_sec = rate[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            pgs_sec = len(t_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(len(t_sec) // sec_length):
                r_subsec = r_sec[(sec_length * k):(sec_length * (k + 1))]

                power = abs((np.fft.fft(r_subsec - np.mean(r_subsec)))[1:size + 1]) ** 2 * 2. * binsize / np.mean(
                    r_subsec) ** 2 / sec_length

                freqall = np.concatenate([freqall, freq])
                powerall = np.concatenate([powerall, power])

        else:
            size = (len(t_sec) + 1) // 2

            if size >= min_length:
                freq = (np.arange((size)) + 1.) / len(t_sec) / binsize

                power = abs((np.fft.fft(r_sec - np.mean(r_sec)))[1:size + 1]) ** 2 * 2. * binsize / np.mean(
                    r_sec) ** 2 / len(r_sec)

                freqall = np.concatenate([freqall, freq])
                powerall = np.concatenate([powerall, power])

            binfac_use = binfac
            n_pgs = 1

        # p.plot(freq,power,'k')

    ### Average /bin up periodograms

    if n_pgs == 0:
        if verbose:
            print('No lightcurve sections long enough to create periodograms.')
        return np.array([]), np.array([]), np.array([])

    # round binfac UP to multiple of number of periodograms
    if verbose: print('Number of periodograms: ' + str(n_pgs))
    binfac_use = max(binfac, (int((binfac - 1) / n_pgs) + 1) * n_pgs)
    if verbose: print('Binning factor:         ' + str(binfac_use))

    # This sorting is slow but necessary to allow possibility of different periodogram lengths
    sortind = np.argsort(freqall)

    freqall = freqall[sortind]
    powerall = powerall[sortind]

    length = (len(freqall) - 1) // binfac_use

    freqbin = np.zeros(length)
    powerbin = np.zeros(length)
    errorbin = np.zeros(length)

    for k in range(length):
        freqbin[k] = np.mean(freqall[(binfac_use * k):(binfac_use * (k + 1))])
        powerbin[k] = np.mean(powerall[(binfac_use * k):(binfac_use * (k + 1))])
        errorbin[k] = np.sqrt(np.var(powerall[(binfac_use * k):(binfac_use * (k + 1))]) / float(binfac_use))

    return freqbin, powerbin, errorbin


#############################################

#############################################
def co_psd(timea, ratea, timeb, rateb, binsize=None, binfac=20, sec_length=None, min_length=10, gap_threshold=1.5,
           verbose=False):
    if binsize == None: binsize = min(timea[1:] - timea[0:-1])

    time = np.intersect1d(timea, timeb)

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    freqall = np.array([])
    powerall = np.array([])
    n_pgs = 0

    for j in range(n_sections):

        a_start = (np.where(time[sec_ends[j] + 1] == timea)[0])[0]
        a_end = a_start + sec_ends[j + 1] - sec_ends[j]
        ta_sec = timea[a_start:a_end]
        ra_sec = ratea[a_start:a_end]

        b_start = (np.where(time[sec_ends[j] + 1] == timeb)[0])[0]
        b_end = b_start + sec_ends[j + 1] - sec_ends[j]
        tb_sec = timeb[b_start:b_end]
        rb_sec = rateb[b_start:b_end]

        assert np.all(ta_sec == tb_sec), 'Times for each lightcurve section should match' + str(
            np.mean(ta_sec - tb_sec))

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            pgs_sec = len(ta_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(pgs_sec):
                ra_subsec = ra_sec[(sec_length * k):(sec_length * (k + 1))]
                rb_subsec = rb_sec[(sec_length * k):(sec_length * (k + 1))]

                power = np.real((np.fft.fft(ra_subsec - np.mean(ra_subsec)))[1:size + 1] * np.conj(
                    (np.fft.fft(rb_subsec - np.mean(rb_subsec)))[1:size + 1])) * 2. * binsize / np.mean(
                    ra_subsec) / np.mean(rb_subsec) / sec_length

                freqall = np.concatenate([freqall, freq])
                powerall = np.concatenate([powerall, power])

        else:
            size = (len(ta_sec) + 1) // 2

            if size >= min_length:
                freq = (np.arange((size)) + 1.) / len(ta_sec) / binsize

                power = np.real((np.fft.fft(ra_sec - np.mean(ra_sec)))[1:size + 1] * np.conj(
                    (np.fft.fft(rb_sec - np.mean(rb_sec)))[1:size + 1])) * 2. * binsize / np.mean(ra_sec) / np.mean(
                    rb_sec) / len(ra_sec)

                freqall = np.concatenate([freqall, freq])
                powerall = np.concatenate([powerall, power])

            binfac_use = binfac
            n_pgs = 0

        # p.plot(freq,power,'k')

    ### Average /bin up periodograms

    if n_pgs == 0:
        if verbose:
            print('No lightcurve sections long enough to create periodograms.')
        return [], [], []

    # round binfac UP to multiple of number of periodograms
    if verbose:
        print('Number of periodograms: ' + str(n_pgs))
    binfac_use = max(binfac, (int((binfac - 1) / n_pgs) + 1) * n_pgs)
    if verbose:
        print('Binning factor:         ' + str(binfac))

    # This sorting is slow but necessary to allow possibility of different periodogram lengths
    sortind = np.argsort(freqall)

    freqall = freqall[sortind]
    powerall = powerall[sortind]

    length = (len(freqall) - 1) // binfac_use

    freqbin = np.zeros(length)
    powerbin = np.zeros(length)
    errorbin = np.zeros(length)

    for k in range(length):
        freqbin[k] = np.mean(freqall[(binfac_use * k):(binfac_use * (k + 1))])
        powerbin[k] = np.mean(powerall[(binfac_use * k):(binfac_use * (k + 1))])
        errorbin[k] = np.sqrt(np.var(powerall[(binfac_use * k):(binfac_use * (k + 1))]) / float(binfac_use))

    return freqbin, powerbin, errorbin


#############################################

from scipy.ndimage.filters import gaussian_filter1d


#############################################
def fad_cpsd(timea, ratea, timeb, rateb, binsize=None, binfac=20, sec_length=None, min_length=10, gap_threshold=1.5,
             verbose=False, smoothing=None, smoothing_length=None):
    ## Method of Bachetti et al. 2017

    time = np.intersect1d(timea, timeb)
    if binsize == None:
        binsize = min(time[1:] - time[0:-1])

    if smoothing_length == None:
        smoothing_length = 3 * int(sec_length * binsize)  # ????
        # print ('Smoothing FADs by '+str(smoothing_length))
    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    freqall = np.array([])
    powerall = np.array([])
    n_pgs = 0

    for j in range(n_sections):

        a_start = (np.where(time[sec_ends[j] + 1] == timea)[0])[0]
        a_end = a_start + sec_ends[j + 1] - sec_ends[j]
        ta_sec = timea[a_start:a_end]
        ra_sec = ratea[a_start:a_end]

        b_start = (np.where(time[sec_ends[j] + 1] == timeb)[0])[0]
        b_end = b_start + sec_ends[j + 1] - sec_ends[j]
        tb_sec = timeb[b_start:b_end]
        rb_sec = rateb[b_start:b_end]

        assert np.all(ta_sec == tb_sec), 'Times for each lightcurve section should match' + str(
            np.mean(ta_sec - tb_sec))

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            pgs_sec = len(ta_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(pgs_sec):

                ra_subsec = ra_sec[(sec_length * k):(sec_length * (k + 1))]
                rb_subsec = rb_sec[(sec_length * k):(sec_length * (k + 1))]

                # power = np.real( (np.fft.fft(ra_subsec - np.mean(ra_subsec)))[1:size+1] * np.conj((np.fft.fft(rb_subsec - np.mean(rb_subsec)))[1:size+1]) ) * 2. * binsize / np.mean(ra_subsec) / np.mean(rb_subsec) / sec_length

                ## Fourier amplitudes of each lightcurve
                fta = np.fft.fft(ra_subsec - np.mean(ra_subsec))[1:size + 1]
                ftb = np.fft.fft(rb_subsec - np.mean(rb_subsec))[1:size + 1]

                ## Subtract to give FADs
                fad = np.absolute(fta - ftb)

                ## Smooth FADs
                if smoothing:
                    fad2 = gaussian_filter1d(fad.real,
                                             smoothing_length) ** 2
                    # fad = smoothing.function(fad)
                else:
                    fad2 = fad.real ** 2

                ## Correct FTs by FADs and normalise
                power = np.real(
                    fta * np.conj(fta)) * 2. * binsize ** 2 / fad2  # / np.sqrt(np.mean(ra_subsec) * np.mean(rb_subsec))

                # *2./(fad**2 * 2/N_ph)

                freqall = np.concatenate([freqall, freq])
                powerall = np.concatenate([powerall, power])

        else:
            raise ValueError('Please enter a section length.')

        # p.plot(freq,power,'k')

    ### Average /bin up periodograms
    if n_pgs == 0:
        if verbose: print('No lightcurve sections long enough to create periodograms.')
        return [], [], []

    # round binfac UP to multiple of number of periodograms
    if verbose: print('Number of periodograms: ' + str(n_pgs))
    binfac_use = max(binfac, (int((binfac - 1) / n_pgs) + 1) * n_pgs)
    if verbose: print('Binning factor:         ' + str(binfac))

    # This sorting is slow but necessary to allow possibility of different periodogram lengths
    sortind = np.argsort(freqall)

    freqall = freqall[sortind]
    powerall = powerall[sortind]

    length = (len(freqall) - 1) // binfac_use

    freqbin = np.zeros(length)
    powerbin = np.zeros(length)
    errorbin = np.zeros(length)

    for k in range(length):
        freqbin[k] = np.mean(freqall[(binfac_use * k):(binfac_use * (k + 1))])
        powerbin[k] = np.mean(powerall[(binfac_use * k):(binfac_use * (k + 1))])
        errorbin[k] = np.sqrt(np.var(powerall[(binfac_use * k):(binfac_use * (k + 1))]) / float(binfac_use))

    return freqbin, powerbin, errorbin


#############################################

## Coherence

#############################################
def coherence2(timea, ratea, timeb, rateb, binsize=None, binfac=20, sec_length=None, min_length=10, gap_threshold=1.5,
               verbose=False):
    assert sec_length is not None, TypeError('Please supply a section length')

    if binsize == None: binsize = min(timea[1:] - timea[0:-1])

    time = np.intersect1d(timea, timeb)

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    size = (int(sec_length) + 1) // 2

    freq = (np.arange((size)) + 1.) / sec_length / binsize
    cross = np.zeros(size, dtype='complex')
    powera = np.zeros(size)
    powerb = np.zeros(size)

    n_pgs = 0

    for j in range(n_sections):

        a_start = (np.where(time[sec_ends[j] + 1] == timea)[0])[0]
        a_end = a_start + sec_ends[j + 1] - sec_ends[j]
        ta_sec = timea[a_start:a_end]
        ra_sec = ratea[a_start:a_end]

        b_start = (np.where(time[sec_ends[j] + 1] == timeb)[0])[0]
        b_end = b_start + sec_ends[j + 1] - sec_ends[j]
        tb_sec = timeb[b_start:b_end]
        rb_sec = rateb[b_start:b_end]

        assert np.all(ta_sec == tb_sec), 'Times for each lightcurve section should match' + str(
            np.mean(ta_sec - tb_sec))

        pgs_sec = len(ta_sec) // sec_length
        n_pgs += pgs_sec

        for k in range(pgs_sec):
            ra_subsec = ra_sec[(sec_length * k):(sec_length * (k + 1))]
            rb_subsec = rb_sec[(sec_length * k):(sec_length * (k + 1))]

            ## Cross-spectrum
            cross += np.conj((np.fft.fft(ra_subsec - np.mean(ra_subsec)))[1:size + 1]) * (np.fft.fft(
                rb_subsec - np.mean(rb_subsec)))[1:size + 1]
            powera += np.abs((np.fft.fft(ra_subsec - np.mean(ra_subsec)))[1:size + 1]) ** 2
            powerb += np.abs((np.fft.fft(ra_subsec - np.mean(ra_subsec)))[1:size + 1]) ** 2

    ### Average /bin up periodograms

    if n_pgs == 0:
        if verbose: print('No lightcurve sections long enough to create periodograms.')
        return [], [], []

    # round binfac UP to multiple of number of periodograms
    if verbose: print('Number of periodograms: ' + str(n_pgs))
    binfac_use = int((binfac - 1) / n_pgs) + 1
    if verbose: print('Binning factor:         ' + str(binfac))

    coherence2 = np.abs(cross) ** 2 / powera / powerb

    err_coherence = np.sqrt(2. / n_pgs / coherence2) * (1 - coherence2)

    length = len(freq) // binfac_use

    freqbin = np.zeros(length)
    coh2bin = np.zeros(length)
    errorbin = np.zeros(length)

    for k in range(length):
        freqbin[k] = (freq[binfac_use * k] + freq[binfac_use * (k + 1) - 1]) / 2.
        coh2bin[k] = np.mean(coherence2[(binfac_use * k):(binfac_use * (k + 1))])
        errorbin[k] = np.sqrt(np.mean(err_coherence[(binfac_use * k):(binfac_use * (
                    k + 1))] ** 2 / binfac_use))  ### np.sqrt(np.var(lagall[(binfac_use*k):(binfac_use*(k+1))])/float(binfac_use))

    return freqbin, coh2bin, errorbin


#############################################

#######################################################################

### Lags

#############################################
def lag_freq(timea, ratea, timeb, rateb, binsize=None, binfac=20, sec_length=None, min_length=10, gap_threshold=1.5,
             verbose=False):
    if binsize == None: binsize = min(timea[1:] - timea[0:-1])

    time = np.intersect1d(timea, timeb)

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    freqall = np.array([])
    lagall = np.array([])
    n_pgs = 0

    for j in range(n_sections):

        a_start = (np.where(time[sec_ends[j] + 1] == timea)[0])[0]
        a_end = a_start + sec_ends[j + 1] - sec_ends[j]
        ta_sec = timea[a_start:a_end]
        ra_sec = ratea[a_start:a_end]

        b_start = (np.where(time[sec_ends[j] + 1] == timeb)[0])[0]
        b_end = b_start + sec_ends[j + 1] - sec_ends[j]
        tb_sec = timeb[b_start:b_end]
        rb_sec = rateb[b_start:b_end]

        assert np.all(ta_sec == tb_sec), 'Times for each lightcurve section should match' + str(
            np.mean(ta_sec - tb_sec))

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            pgs_sec = len(ta_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(pgs_sec):
                ra_subsec = ra_sec[(sec_length * k):(sec_length * (k + 1))]
                rb_subsec = rb_sec[(sec_length * k):(sec_length * (k + 1))]

                ## phase difference over -2pi to 2pi
                dphase = np.angle((np.fft.fft(ra_subsec - np.mean(ra_subsec)))[1:size + 1]) - np.angle(
                    (np.fft.fft(rb_subsec - np.mean(rb_subsec)))[1:size + 1])

                lag = (((dphase + np.pi) % (2 * np.pi)) - np.pi) / freq / 2. / np.pi

                freqall = np.concatenate([freqall, freq])
                lagall = np.concatenate([lagall, lag])

        else:
            raise TypeError('Please supply a section length')
            # size = (len(ta_sec)+1)//2

            # if size >=min_length:
            # freq = (np.arange((size))+1.) / len(ta_sec) / binsize

            # power = np.real( (np.fft.fft(ra_sec - np.mean(ra_sec)))[1:size+1] * np.conj((np.fft.fft(rb_sec - np.mean(rb_sec)))[1:size+1]) ) * 2. * binsize / np.mean(ra_sec) / np.mean(rb_sec) / len(ra_sec)

            # freqall  = np.concatenate([freqall, freq])
            # powerall = np.concatenate([powerall, power])

            # binfac_use=binfac
            # n_pgs=0

        # p.plot(freq,power,'k')

    ### Average /bin up periodograms

    if n_pgs == 0:
        if verbose: print('No lightcurve sections long enough to create periodograms.')
        return [], [], []

    # round binfac UP to multiple of number of periodograms
    if verbose: print('Number of periodograms: ' + str(n_pgs))
    binfac_use = max(binfac, (int((binfac - 1) / n_pgs) + 1) * n_pgs)
    if verbose: print('Binning factor:         ' + str(binfac))

    # This sorting is slow but necessary to allow possibility of different periodogram lengths
    sortind = np.argsort(freqall)

    freqall = freqall[sortind]
    lagall = lagall[sortind]

    length = (len(freqall) - 1) // binfac_use

    freqbin = np.zeros(length)
    lagbin = np.zeros(length)
    errorbin = np.zeros(length)

    for k in range(length):
        freqbin[k] = np.mean(freqall[(binfac_use * k):(binfac_use * (k + 1))])
        lagbin[k] = np.mean(lagall[(binfac_use * k):(binfac_use * (k + 1))])
        errorbin[k] = np.sqrt(np.var(lagall[(binfac_use * k):(binfac_use * (k + 1))]) / float(binfac_use))

    return freqbin, lagbin, errorbin


#############################################

#######################################################################

### Phase folding

#############################################
def phase_profile(time, rate, frequency=None, binsize=None, binfac=20, sec_length=None, min_length=10,
                  gap_threshold=1.5, verbose=False, max_harm=np.inf, plot=False):
    if plot: from matplotlib import pyplot as p

    #############################################
    ## Define frequency range for fundamental

    if frequency == None:
        ## For no frquency given, allow any frequency
        freq_use = [0, np.inf]

    elif type(frequency) == float or type(frequency) == int:
        ## For specified frequency, use closest available in PSD
        f, po, er = psd(time, rate, binsize=binsize, binfac=binfac, sec_length=sec_length, min_length=min_length,
                        gap_threshold=gap_threshold, verbose=verbose)

        freq_use = f[np.where((f - frequency) ** 2 == np.min((f - frequency) ** 2))[0][0]]
        freq_use = [freq_use, freq_use * 1.00000001]

    elif len(frequency) == 2:
        ## For range of frequencies, use frequency with greatest power in each periodogram
        freq_use = frequency

    else:
        raise TypeError('frequency must have length at most 2')

    #############################################

    ### Make sections as for PSD

    #############################################
    if binsize == None: binsize = min(time[1:] - time[0:-1])

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    # freqall  = np.array([])
    # powerall = np.array([])
    harm_all = np.array([])
    phase_diff_all = np.array([])
    pulses = []
    n_pgs = 0

    for j in range(n_sections):

        t_sec = time[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]
        r_sec = rate[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            ## Indices of frequencies in specified range for fundamental
            ind_f_range = np.intersect1d(np.where(freq >= freq_use[0])[0], np.where(freq <= freq_use[1])[0])

            ## Periodograms in this section
            pgs_sec = len(t_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(len(t_sec) // sec_length):

                ### Rates of each section from which to produce periodogram
                r_subsec = r_sec[(sec_length * k):(sec_length * (k + 1))]

                ### Find frequency with most power in given range
                ft_of_section = (np.fft.fft(r_subsec - np.mean(r_subsec)))[1:size + 1]
                power = abs(ft_of_section) ** 2 * 2. * binsize / np.mean(r_subsec) ** 2 / sec_length

                print(len(power[ind_f_range] * freq[ind_f_range]))

                ind_fundamental = ind_f_range[
                    np.where(power[ind_f_range] * freq[ind_f_range] == np.max(power[ind_f_range] * freq[ind_f_range]))[
                        0]]

                ### Find phases for harmonics

                phase = np.angle(ft_of_section)

                harm = (np.arange(size // (ind_fundamental + 1)) + 1)
                max_harm_use = int(np.minimum(max_harm, len(harm)))

                ind_harm = (ind_fundamental + 1) * harm - 1

                phase_diff = (phase[ind_harm] - phase[ind_fundamental] * harm) % (2 * np.pi)

                ft_of_pulse = np.zeros(sec_length, dtype=np.complex)
                ft_of_pulse[harm[0:max_harm_use]] = (ft_of_section[ind_harm[0:max_harm_use]]) * np.exp(
                    -1j * phase[ind_fundamental[0:max_harm_use]] * harm[0:max_harm_use])

                pulse = np.fft.ifft(ft_of_pulse)
                pulses = pulses + [pulse]

                if plot: p.plot(np.arange(len(pulse)) * 2 * np.pi / len(pulse), (pulse), color='grey', alpha=0.1)
                phase_diff_all = np.concatenate([phase_diff_all, phase_diff])
                harm_all = np.concatenate([harm_all, harm])

                # freqall  = np.concatenate([freqall, freq])
                # powerall = np.concatenate([powerall, power])

        else:
            raise TypeError('Please supply a section length')

    if plot: p.plot(np.arange(len(pulse)) * 2 * np.pi / len(pulse), np.mean((np.array(pulses)), axis=0))

    return np.array(np.arange(len(pulse)) * 2 * np.pi / len(pulse)), np.array(
        np.mean(np.array(pulses), axis=0)), np.array(np.std(np.array(pulses), axis=0) / np.sqrt(n_pgs))
    #############################################

    #############################################


#######################################################################


def phase_dist_harm(time, rate, frequency=None, binsize=None, binfac=20, sec_length=None, min_length=10,
                    gap_threshold=1.5, verbose=False, max_harm=np.inf, plot=False):
    if plot: from matplotlib import pyplot as p

    #############################################
    ## Define frequency range for fundamental

    if frequency == None:
        ## For no frquency given, allow any frequency
        freq_use = [0, np.inf]

    elif type(frequency) == float or type(frequency) == int:
        ## For specified frequency, use closest available in PSD
        f, po, er = psd(time, rate, binsize=binsize, binfac=binfac, sec_length=sec_length, min_length=min_length,
                        gap_threshold=gap_threshold, verbose=verbose)

        freq_use = f[np.where((f - frequency) ** 2 == np.min((f - frequency) ** 2))[0][0]]
        freq_use = [freq_use, freq_use * 1.00000001]

    elif len(frequency) == 2:
        ## For range of frequencies, use frequency with greatest power in each periodogram
        freq_use = frequency

    else:
        raise TypeError('frequency must have length at most 2')

    #############################################

    ### Make sections as for PSD

    #############################################
    if binsize == None: binsize = min(time[1:] - time[0:-1])

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    # freqall  = np.array([])
    # powerall = np.array([])
    harm_all = []
    phase_diff_all = []
    pulses = []
    n_pgs = 0

    for j in range(n_sections):

        t_sec = time[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]
        r_sec = rate[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            ## Indices of frequencies in specified range for fundamental
            ind_f_range = np.intersect1d(np.where(freq >= freq_use[0])[0], np.where(freq <= freq_use[1])[0])

            ## Periodograms in this section
            pgs_sec = len(t_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(len(t_sec) / sec_length):
                ### Rates of each section from which to produce periodogram
                r_subsec = r_sec[(sec_length * k):(sec_length * (k + 1))]

                ### Find frequency with most power in given range
                ft_of_section = (np.fft.fft(r_subsec - np.mean(r_subsec)))[1:size + 1]
                power = abs(ft_of_section) ** 2 * 2. * binsize / np.mean(r_subsec) ** 2 / sec_length

                ind_fundamental = ind_f_range[
                    np.where(power[ind_f_range] * freq[ind_f_range] == np.max(power[ind_f_range] * freq[ind_f_range]))[
                        0]]

                ### Find phases for harmonics

                phase = np.angle(ft_of_section)

                harm = (np.arange(size / (ind_fundamental + 1)) + 1)
                max_harm_use = int(np.minimum(max_harm, len(harm)))

                ind_harm = (ind_fundamental + 1) * harm - 1

                phase_diff = (phase[ind_harm] - phase[ind_fundamental] * harm) % (2 * np.pi)

                ft_of_pulse = np.zeros(sec_length, dtype=np.complex)
                ft_of_pulse[harm[0:max_harm_use]] = (ft_of_section[ind_harm[0:max_harm_use]]) * np.exp(
                    -1j * phase[ind_fundamental[0:max_harm_use]] * harm[0:max_harm_use])

                # pulse = np.fft.ifft(ft_of_pulse)
                # pulses = pulses+[pulse]

                # if plot: p.plot(np.arange(len(pulse))*2*np.pi/len(pulse), (pulse), color='grey', alpha=0.1)
                phase_diff_all = np.concatenate([phase_diff_all, phase_diff])
                harm_all = np.concatenate([harm_all, harm])

        else:
            raise TypeError('Please supply a section length')

    # if plot: p.plot(np.arange(len(pulse))*2*np.pi/len(pulse), np.mean((np.array(pulses)), axis=0))

    harm_fin = np.arange(1, max(harm_all) + 1)
    phase_diff_fin = []

    for i in range(int(max(harm_all))):
        phase_diff_fin = phase_diff_fin + [phase_diff_all[np.where(harm_all == i + 1)[0]]]

    return harm_fin, phase_diff_fin
    #############################################


#######################################################################


def phase_dist_rel(timea, ratea, timeb, rateb, frequency=None, binsize=None, binfac=20, sec_length=None, min_length=10,
                   gap_threshold=1.5, verbose=False, max_harm=np.inf, plot=False):
    if plot: from matplotlib import pyplot as p

    #############################################

    ### Make sections as for PSD

    #############################################
    if binsize == None: binsize = min(time[1:] - time[0:-1])

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    # freqall  = np.array([])
    # powerall = np.array([])
    harm_all = []
    phase_diff_all = []
    pulses = []
    n_pgs = 0

    for j in range(n_sections):

        t_sec = time[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]
        r_sec = rate[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            ## Indices of frequencies in specified range for fundamental
            ind_f_range = np.intersect1d(np.where(freq >= freq_use[0])[0], np.where(freq <= freq_use[1])[0])

            ## Periodograms in this section
            pgs_sec = len(t_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(len(t_sec) / sec_length):
                ### Rates of each section from which to produce periodogram
                r_subsec = r_sec[(sec_length * k):(sec_length * (k + 1))]

                ### Find frequency with most power in given range
                ft_of_section = (np.fft.fft(r_subsec - np.mean(r_subsec)))[1:size + 1]
                power = abs(ft_of_section) ** 2 * 2. * binsize / np.mean(r_subsec) ** 2 / sec_length

                ind_fundamental = ind_f_range[
                    np.where(power[ind_f_range] * freq[ind_f_range] == np.max(power[ind_f_range] * freq[ind_f_range]))[
                        0]]

                ### Find phases for harmonics

                phase = np.angle(ft_of_section)

                harm = (np.arange(size / (ind_fundamental + 1)) + 1)
                max_harm_use = int(np.minimum(max_harm, len(harm)))

                ind_harm = (ind_fundamental + 1) * harm - 1

                phase_diff = (phase[ind_harm] - phase[ind_fundamental] * harm) % (2 * np.pi)

                ft_of_pulse = np.zeros(sec_length, dtype=np.complex)
                ft_of_pulse[harm[0:max_harm_use]] = (ft_of_section[ind_harm[0:max_harm_use]]) * np.exp(
                    -1j * phase[ind_fundamental[0:max_harm_use]] * harm[0:max_harm_use])

                # pulse = np.fft.ifft(ft_of_pulse)
                # pulses = pulses+[pulse]

                # if plot: p.plot(np.arange(len(pulse))*2*np.pi/len(pulse), (pulse), color='grey', alpha=0.1)
                phase_diff_all = np.concatenate([phase_diff_all, phase_diff])
                harm_all = np.concatenate([harm_all, harm])

        else:
            raise TypeError('Please supply a section length')

    # if plot: p.plot(np.arange(len(pulse))*2*np.pi/len(pulse), np.mean((np.array(pulses)), axis=0))

    harm_fin = np.arange(1, max(harm_all) + 1)
    phase_diff_fin = []

    for i in range(int(max(harm_all))):
        phase_diff_fin = phase_diff_fin + [phase_diff_all[np.where(harm_all == i + 1)[0]]]

    return freq_fin, phase_diff_fin
    #############################################


#############################################

## Earlier names

#############################################

phase_dist = phase_dist_harm


#######################################################################

### Higher-order (bispectra etc.)

#############################################

def bispectrum(time, rate, binsize=None, binfac=20, sec_length=None, min_length=10, gap_threshold=1.5, verbose=False):
    if binsize == None: binsize = min(time[1:] - time[0:-1])

    breaks = (np.where([time[1:] - time[0:-1] > binsize * gap_threshold]))[1]

    n_sections = len(breaks) + 1

    sec_ends = np.concatenate([[-1], breaks, [len(time) - 1]])

    size = (int(sec_length) + 1) / 2
    freqall = np.array([])
    powerall = np.array([])
    bisp = np.zeros([size + 1, size + 1]) + 0j
    n_pgs = 0

    for j in range(n_sections):

        t_sec = time[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]
        r_sec = rate[(sec_ends[j] + 1):(sec_ends[j + 1] + 1)]

        if sec_length is not None:

            size = (int(sec_length) + 1) // 2

            freq = (np.arange((size)) + 1.) / sec_length / binsize

            pgs_sec = len(t_sec) // sec_length
            n_pgs += pgs_sec

            for k in range(len(t_sec) / sec_length):

                r_subsec = r_sec[(sec_length * k):(sec_length * (k + 1))]

                f_comp = (np.fft.fft(r_subsec - np.mean(r_subsec)))[0:size + 1]

                for fi in range(size + 1):
                    fj = size + 1 - fi
                    bisp[fi, 0:fj] += f_comp[fi] * f_comp[0:fj] * np.conj(f_comp[fi:fi + fj])

                # power = abs( (np.fft.fft(r_subsec - np.mean(r_subsec)))[1:size+1] ) **2 * 2. * binsize / np.mean(r_subsec)**2 / sec_length

                # freqall  = np.concatenate([freqall, freq])
                # powerall = np.concatenate([powerall, power])


        else:
            raise TypeError('sec_length is required')

        # p.plot(freq,power,'k')

    ### Average /bin up periodograms
    if False:
        if n_pgs == 0:
            if verbose: print('No lightcurve sections long enough to create periodograms.')
            return np.array([]), np.array([]), np.array([])

        # round binfac UP to multiple of number of periodograms
        if verbose: print('Number of periodograms: ' + str(n_pgs))
        binfac_use = max(binfac, (int((binfac - 1) / n_pgs) + 1) * n_pgs)
        if verbose: print('Binning factor:         ' + str(binfac_use))

        length = (len(freqall) - 1) // binfac_use

        freqbin = np.zeros(length)
        powerbin = np.zeros(length)
        errorbin = np.zeros(length)

        for k in range(length):
            freqbin[k] = np.mean(freqall[(binfac_use * k):(binfac_use * (k + 1))])
            powerbin[k] = np.mean(powerall[(binfac_use * k):(binfac_use * (k + 1))])
            errorbin[k] = np.sqrt(np.var(powerall[(binfac_use * k):(binfac_use * (k + 1))]) / float(binfac_use))

    return freq, bisp[1:, 1:] / n_pgs  # , bisp_err[1:,1:]/n_pgs
#############################################






