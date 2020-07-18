"""
Script for generating simulated time series using the method of Timmer and Konig (1995).
"""

from numpy import log, zeros, sqrt, arange, exp, real, pi, interp, conj
from numpy.random import normal
from numpy.fft import fft


def ts_gen(n, dt=1., freq=[], pow=[], seed=None, time=0, spline=0, double=1, phase=[0], loginterp=True):
    """
    ; ----------------------------------------------------------
    ;+
    ; NAME:
    ;       TS_GEN
    ;
    ; PURPOSE:
    ;       Generate a random time series from a power spectrum model
;
; AUTHOR:
;       Simon Vaughan (U.Leicester)
;
; CALLING SEQUENCE:
;       x = TS_GEN(65536,dt=0.1)
;
; INPUTS:
;       n         - (scalar) length of time series (default = 65536)
;
; OPTIONAL INPUTS:
;       dt        - (scalar) Sampling period (default = 1.0)
;       freq      - (vector) frequencies at which spectrum is known
;       pow       - (vector) spectral density at frequencies FREQ
;       phase     - (vector) phase shift at frequencies FREQ (default=0)
;       seed      - (long integer) seed for random number generator
;       spline    - (logical) use cubic spline interpolation
;       log       - (logical) interpolate linearly in log-log space
;       double    - (logical) perform FFT in double prec.
;
; OUTPUTS:
;       x         - (vector) output time series
;
; OPTIONAL OUTPUTS:
;       time      - sampling times [0,n-1]*dt
;
; DETAILS:
;       Generate an evenly-sampled time series for a noise (random)
;       process with a power spectrum specified by POW and FREQ.
;
;       The method comes from:
;         Davies R. B., Harte D. S., 1987, Biometrika, v74, pp95-101
;       and was introduced to astronomy by
;         Timmer J., Konig M., 1995, A&A, v300, pp707-710
;
;       The time series is generated using the following algorithm:
;
;        1. The "true" power spectrum is specified using POW(FREQ)
;
;        2. Define the Fourier frequencies for the output time series
;        as f_j = j/(N*dT) with j=1,2,...,N/2. Use interpolation
;        to find the power spectrum at f_j from input POW(FREQ)
;
;        3. The "true" power spectrum is converted from power
;        (non-negative) to a "true" DFT for the process, using the
;        fact that POW = |DFT|^2, so we have a complex-valued
;        DFT = complex(sqrt(POW),sqrt(POW)) at each frequency f_j
;
;        4. Draw two sets of N/2 normal deviates (random numbers from
;        "normal" Gaussian distribution.
;
;        5. Multiply the real and imaginary parts of the DFT by the
;        deviates. This randomised the "true" DFT and gives it the
;        distribution expected for an observed or "sampled" DFT from a
;        single time series of a random process.
;         X(f_j) = DFT(f_j) * eps_j   where eps_j is a normal deviate
;
;        6. Use the inverse FT to convert from the frequency domain to
;        the time domain, i.e. from x(t_i) = FFT[X(f_j)]
;
;        7. Fill-in the array of times t_i = i*dT for i=0,...,N-1
;
;       The randomisation step (5) is equivalent to drawing the square
;       amplitude of the DFT from a chi-squared distribution (with two
;       degrees of freedom), and the phase of the DFT from a uniform
;       distribution over the range [0,2*pi]. These are the expected
;       sampling distributions from a random time series.
;
;       Note that in reality the DFT is also defined for negative
;       Fourier frequencies j=-N/2,...,-1. In order for the resulting
;       time series to be real we require that the X(f_j) = X'(-f_j),
;       so the negative frequencies carry the complex conjugate of the
;       positive frequencies. Each side of the DFT is normalised by
;       1/2 so that the sum over all (-ve and +ve) frequencies is
;       equal to the total variace (the integral of the power
;       spectrum).

;       Also, the DFT at the Nyquist frequency j=N/2 is always real
;       when N is even, so the imaginary part is set to zero.
;       The DFT at zero frequency (j=0) determines the mean (DC
;       component) of the resulting time series. Here we generate
;       zero-mean data, so this is set to zero, i.e. X(f_j = 0) = 0.
;
;       The spectrum is specified by the vectors FREQ and POW, which
;       are interpolated as needed to populate the periodogram needed
;       for the generation (step 2). Interpolation is linear unless SPLINE
;       keyword is set (in which case it is cubic spline). If FREQ and
;       POW are not specified, the spectrum is assumed to be flat
;       (i.e. POW = constant).
;
;       WARNING: The routine needs to know the power density at
;       frequencies f_j = j/(N*dT) with j=1,2,...,N/2. You need
;       to make sure your input spectrum spans this full range,
;       i.e. that MIN(FREQ) <= 1/(N*dT) and MAX(FREQ) >= 1/2dT.
;       This may involve simply adding more extra point to the
;       input power spectrum at a very low or high frequency.
;       If this is the case the program will return a WARNING
;       but carry on by extrapolating the data outside the
;       range of the input data.
;
;       If the input power spectrum is a power law it may be best
;       to use the LOG keyword. This forces the interpolation to
;       be done using the log of the power spectrum. I.e. it
;       interpolates log(pow) - log(freq) data, and then converts
;       the result back to linear-space. In effect, it interpolates
;       between points using a power law model.
;
;       As the routine uses the FFT function, it works fastest if N is
;       a power of 2 (default = 2^16)
;
;       There is an addition optional input PHASE. This allows a phase
;       shift to be added to the data. Since the phase is randomly and
;       uniformly distibuted over the range [0,2*pi] this is of no
;       value for a single time series. But it is possible to generate
;       two time series by calling the routine twice using the same
;       random number seed but different POW or PHASE values. The will
;       result in two time series that differ only in their power
;       spectrum (modulus square of DFT) or phase (argument of DFT).
;       If X = A*exp(i*theta) then applying a phase shift phi we get
;       X' = A*exp(i*[theta + phi]) = X * exp(i*phi).
;
; EXAMPLE USAGE:
;
;   Generate time series with 1/f spectrum
;
;     IDL> freq = (INDGEN(512)+1)/1024.0
;     IDL> pow = freq^(-1)
;     IDL> x = TS_GEN(1024, dt=1.0, freq=freq, pow=pow,time=time, $
;                     seed=seed)
;     IDL> plot,time,x
;
;   Generate time series with 1/f spectrum  making use of LOG keyword
;
;     IDL> freq = [1e-6,100]
;     IDL> pow = 0.01*freq^(-1)
;     IDL> x = TS_GEN(1024, dt=1.0, freq=freq, pow=pow,time=time, $
;                     seed=seed,/log)
;     IDL> plot,time,x
;
;   Because the spectrum is a power law, we only need define two
;   end points at let the interpolation (in log-log space) do the rest.
;   (NB: try this without the LOG keyword to see how it goes wrong!)
;
;   Generate two time series with constant phase delay of pi/2 using a
;   1/f^2 spectrum
;
;     IDL> freq = (INDGEN(512)+1)/1024.0
;     IDL> pow = freq^(-2)
;     IDL> s = 123L
;     IDL> x = TS_GEN(1024, dt=1.0, freq=freq, pow=pow,time=time, $
;                     seed=s,phase=0)
;     IDL> plot,time,x
;     IDL> phase = !pi/2
;     IDL> s = 123L
;     IDL> x = TS_GEN(1024, dt=1.0, freq=freq, pow=pow,time=time,$
;                     seed=s,phase=phase)
;     IDL> oplot,time,x,color=170
;
; NB: A constant time delay of tau can be produced using
; phase(j) = 2.0*!pi*tau*freq(j)
;
; HISTORY:
;       14/05/07  - v1.0 - first working version
;       15/05/07  - v1.1 - bug fix: INDGEN now uses L64 keyword
;                          this is needed for N > 2^15
;       20/12/07  - v1.2 - added PHASE keyword
;       15/01/09  - v1.3 - added LOG keyword
;       19/01/09  - v.14 - added check that range of FREQ
;                          spans [f_min = 1/NdT, f_max = 1/2dT]
;       22/09/10  - v1.5 - added clauses to allow for integer DT values
;
; NOTES:
;       + uses built in random number generator
;
;-
; ---------------------------------------------------------
"""
    # ; options for compilation (recommended by RSI)

    #  COMPILE_OPT idl2, HIDDEN

    # ; watch out for errors

    # on_error, 2

    # ; ----------------------------------------------------------
    # ; Check the arguments

    # ; if N not defined, set default

    # if len(n) == 0 : n = 65536

    # ; make sure N is even

    if (n % 2) != 0:
        print('** Please make N even in TS_GEN')
        return 0
    else:
        n = int(n)

    # ; check the shape of the input array

    nf = len(freq)
    np = len(pow)
    if (nf != np):
        print('** FREQ and POW of differing sizes in TS_GEN.')
        return 0

    if nf < 2:
        print('** FREQ too small in TS_GEN.')
        return 0

    # ; if FREQ is not defined, set-up default (flat) spectrum

    if nf == 0:
        freq = [0.0, 0.5 / dt]
        pow = [1.0, 1.0]

    # ; if PHASE is not defined, set-up default (zero) phase shift

    np = len(phase)
    if (np != nf and np != 1):
        print('** FREQ and PHASE of differing sizes in TS_GEN.')
        return (0)

    if (np == 0): phi = zeros(nf)
    if (np == 1): phi = zeros(nf) + phase[0]
    if (np == nf): phi = phase

    # ; check that PHI is within range [0,2*pi]

    phi = phi % (2. * pi)

    # ; ----------------------------------------------------------
    # ; check the range of input frequencies spans the range
    # ; needed to generate the simulation

    f_min = 1.0 / (n * dt)
    f_max = 1.0 / (2.0 * dt)

    if min(freq) > f_min:
        print("-- WARNING. MIN(FREQ) > f_min in TS_GEN.")
        print("-- MIN(FREQ) = ", min(freq), " f_min = ", f_min)
        print("-- Data will be extrapolated. You may prefer to EXPand the range of FREQ.")

    if max(freq) < f_max:
        print("** MAX(FREQ) < f_max in TS_GEN. EXPand range of FREQ")
        print("-- MAX(FREQ) = ", max(freq), " f_max = ", f_max)
        print("-- Data will be extrapolated. You may prefer to EXPand the range of FREQ.")

    # ; ----------------------------------------------------------
    # ; Main part of procedure

    # ; number of positive frequencies in Fourier Transform

    nf = n // 2

    # ; make array for Fourier Transform
    # ; (need room for positive and negative frequencies)

    x = zeros((2 * nf), dtype=complex)

    # ; make array for frequencies

    f = arange(nf + 1, dtype=float) / (n * dt)

    # ; interpolation of input power spectrum, to fill in any gaps
    # ; interpolate given spectrum POW(FREQ) onto required frequencies F

    if loginterp:

        # ; convert to log-log space, interpolate there, and convert back

        lpow = log(pow)
        lfreq = log(freq)
        lf = log(f[1:nf + 1])
        lspec = interp(lf, lfreq, lpow)
        spec = zeros(nf + 1)
        spec[1:nf + 1] = exp(lspec)
        # spec = [0.0, spec]
        lpow = 0
        lfreq = 0
        lf = 0

    else:

        # ; or just interpolate in lin-lin space as default

        spec = interp(f, freq, pow)

    # ; set DC value ( psd(f=0) ) to zero

    spec[0] = 0.0

    # ; interpolate phase shift spectrum (PHI)

    phi = interp(f, freq, phi)

    # ; normalise spectrum

    spec = spec * nf / (2.0 * dt)

    # ; ----------------------------------------------------------
    # ; now for the Davies-Harte algorithm

    # ; take square root of spectrum

    spec = sqrt(spec)

    # ; populate Fourier Transform of time series with SQRT(spectrum)
    # ; multiplied by normal deviate
    # ; (independent for real and complex components)

    # ; first positive frequencies

    x[1:nf] = spec[1:nf] * normal(size=nf - 1) + spec[1:nf] * normal(size=nf - 1) * 1j

    # ; apply phase shift X'(f) = X(f) * EXP(i*phi)

    x[1:nf] = x[1:nf] * exp(phi[1:nf] * 1j)

    # ; FT must be real at Nyquist frequency

    x[nf] = spec[nf] * normal()

    # ; make sure FT at negative frequencies is conjugate: X(-f) = X*(f)

    x[nf + 1:2 * nf] = conj(x[nf - 1:0:-1])

    # ; then inverse Fourier Transform into time domain: X(f) -> x(t)

    x = fft(x)

    # ; drop imaginary part (which is zero)

    x = real(x)

    # ; calculate TIME if needed

    time = arange(n) * dt

    # ; ----------------------------------------------------------
    # ; Return the data array to the user

    return x
