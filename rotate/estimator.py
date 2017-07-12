# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.ndimage import gaussian_filter

from astropy.stats import LombScargle

__all__ = ["lomb_scargle_estimator", "autocorr_estimator"]


def lomb_scargle_estimator(x, y, yerr=None,
                           min_period=None, max_period=None,
                           filter_period=None,
                           max_peaks=2,
                           **kwargs):
    """
    Estimate period of a time series using the periodogram

    Args:
        x (ndarray[N]): The times of the observations
        y (ndarray[N]): The observations at times ``x``
        yerr (Optional[ndarray[N]]): The uncertainties on ``y``
        min_period (Optional[float]): The minimum period to consider
        max_period (Optional[float]): The maximum period to consider
        filter_period (Optional[float]): If given, use a high-pass filter to
            down-weight period longer than this
        max_peaks (Optional[int]): The maximum number of peaks to return
            (default: 2)

    Returns:
        A dictionary with the computed ``periodogram`` and the parameters for
        up to ``max_peaks`` peaks in the periodogram.

    """
    if min_period is not None:
        kwargs["maximum_frequency"] = 1.0 / min_period
    if max_period is not None:
        kwargs["minimum_frequency"] = 1.0 / max_period

    # Estimate the power spectrum
    model = LombScargle(x, y, yerr)
    freq, power = model.autopower(method="fast", normalization="psd", **kwargs)
    power /= len(x)

    # Filter long periods
    if filter_period is not None:
        freq0 = 1.0 / filter_period
        filt = 1.0 / np.sqrt(1 + (freq0 / freq) ** (2*3))
        power *= filt

    # Find and fit peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for i in peak_inds[:max_peaks]:
        A = np.vander(freq[i-1:i+2], 3)
        w = np.linalg.solve(A, np.log(power[i-1:i+2]))
        sigma2 = -0.5 / w[0]
        freq0 = w[1] * sigma2
        peaks.append(dict(
            log_power=w[2] + 0.5*freq0**2 / sigma2,
            period=1.0 / freq0,
            period_uncert=np.sqrt(sigma2 / freq0**4),
        ))

    return dict(
        periodogram=(freq, power),
        peaks=peaks,
    )

def autocorr_estimator(x, y, yerr=None,
                       min_period=None, max_period=None,
                       oversample=2.0, smooth=2.0, max_peaks=10):
    """
    Estimate the period of a time series using the autocorrelation function

    .. note:: The signal is interpolated onto a uniform grid in time so that
        the autocorrelation function can be computed.

    Args:
        x (ndarray[N]): The times of the observations
        y (ndarray[N]): The observations at times ``x``
        yerr (Optional[ndarray[N]]): The uncertainties on ``y``
        min_period (Optional[float]): The minimum period to consider
        max_period (Optional[float]): The maximum period to consider
        oversample (Optional[float]): When interpolating, oversample the times
            by this factor (default: 2.0)
        smooth (Optional[float]): Smooth the autocorrelation function by this
            factor times the minimum period (default: 2.0)
        max_peaks (Optional[int]): The maximum number of peaks to identify in
            the autocorrelation function (default: 10)

    Returns:
        A dictionary with the computed autocorrelation function and the
        estimated period. For compatibility with the
        :func:`lomb_scargle_estimator`, the period is returned as a list with
        the key ``peaks``.

    """
    if min_period is None:
        min_period = np.min(np.diff(x))
    if max_period is None:
        max_period = x.max() - x.min()

    # Interpolate onto an evenly spaced grid
    dx = np.min(np.diff(x)) / float(oversample)
    xx = np.arange(x.min(), x.max(), dx)
    yy = np.interp(xx, x, y)

    # Estimate the autocorrelation function
    tau = xx - x[0]
    acor = autocorr_function(yy)
    smooth = smooth * min_period
    acor = gaussian_filter(acor, smooth / dx)

    # Find the peaks
    peak_inds = (acor[1:-1] > acor[:-2]) & (acor[1:-1] > acor[2:])
    peak_inds = np.arange(1, len(acor)-1)[peak_inds]
    peak_inds = peak_inds[tau[peak_inds] >= min_period]

    result = dict(
        autocorr=(tau, acor),
        peaks=[],
    )

    # No peaks were found
    if len(peak_inds) == 0 or tau[peak_inds[0]] > max_period:
        return result

    # Only one peak was found
    if len(peak_inds) == 1:
        result["peaks"] = [dict(period=tau[peak_inds[0]],
                                period_uncert=np.nan)]
        return result

    # Check to see if second peak is higher
    if acor[peak_inds[1]] > acor[peak_inds[0]]:
        peak_inds = peak_inds[1:]

    # The first peak is larger than the maximum period
    if tau[peak_inds[0]] > max_period:
        return result

    # This is the initial estimate of the period
    period = tau[peak_inds[0]]
    peaks = [0.0, period]

    # Find the string of peaks within 0.3 of the period estimate
    for i in peak_inds[1:]:
        next_peak = peaks[-1] + period
        if np.abs(next_peak - tau[i]) < 0.3 * period:
            peaks.append(tau[i])
            if len(peaks) >= max_peaks + 1:
                break

    if len(peaks) <= 2:
        result["peaks"] = [dict(period=peaks[1], period_uncert=np.nan)]
        return result

    # Estimate the period and the uncertainties
    diff = np.diff(peaks)
    period = np.median(diff)
    period_uncert = 1.483 * np.median(np.abs(diff - period))
    period_uncert /= np.sqrt(len(peaks) - 2)
    result["peaks"] = [dict(period=period, period_uncert=period_uncert)]
    return result

def autocorr_function(x):
    """Estimate the autocorrelation function of a 1D dataset"""
    x = np.atleast_1d(x)
    n = len(x)
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    return acf / acf[0]
