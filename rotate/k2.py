# -*- coding: utf-8 -*-

from __future__ import division, print_function

import everest
import exoarch
import numpy as np
from scipy.signal import savgol_filter

__all__ = ["get_light_curve"]


def sigma_clip(f, thresh=5, window=49):
    """Get a binary mask of 'good' points using sigma clipping

    Args:
        thresh (float): The sigma clipping threshold.
        window (int): The width of the smoothing window for sigma
            clipping.

    """
    f = f - savgol_filter(f, window, 2) + np.nanmedian(f)
    mu = np.median(f)
    std = np.sqrt(np.median((f - mu)**2))
    return np.abs(f - mu) < thresh*std


def get_light_curve(epicid, season=None, mask_transits=True, mask_width=3,
                    sigma_iter=10, sigma_thresh=5.0, sigma_window=49):
    """Get the light curve for a given EPIC ID

    Args:
        epicid (int): The ID of the target.
        mask_transits (bool): Should known candidates be masked?
        mask_width (float): The half width of the transit mask in units of the
            transit duration.
        sigma_iter (int): The maximum number of iterations of sigma clipping to
            run.
        sigma_thresh (float): The sigma clipping threshold.
        sigma_window (int): The width of the smoothing window for sigma
            clipping.

    Returns:
        t (ndarray): The array of timestamps.
        F (ndarray): The ``(ntime, npix)`` matrix of (normalized) pixel flux
            time series.
        yerr (ndarray): An estimate of the uncertainties of the SAP flux
            (``sum(F, axis=1)``).

    """
    star = everest.Everest(epicid, season=season, quiet=True)
    t = star.apply_mask(star.time)
    F = star.apply_mask(star.fpix)

    # Mask any known transits
    if mask_transits:
        k2cand = exoarch.ExoplanetArchiveCatalog("k2candidates").df
        epic = k2cand[k2cand.epic_name == "EPIC {0}".format(epicid)]
        cands = epic.groupby("epic_candname").mean()
        for _, cand in cands.iterrows():
            t0 = cand.pl_tranmid - 2454833.0
            per = cand.pl_orbper
            dur = cand.pl_trandur
            m = np.abs((t - t0 + 0.5*per) % per - 0.5*per) > mask_width * dur
            t = t[m]
            F = F[m]

    # Use 1st order PLD to do some sigma clipping
    fsap = np.sum(F, axis=1)
    A = F / fsap[:, None]
    m = np.ones_like(fsap, dtype=bool)
    for i in range(sigma_iter):
        w = np.linalg.solve(np.dot(A[m].T, A[m]), np.dot(A[m].T, fsap[m]))
        resid = fsap - np.dot(A, w)
        m_new = sigma_clip(resid, thresh=sigma_thresh, window=sigma_window)
        if m.sum() == m_new.sum():
            m = m_new
            break
        m = m_new
    t = t[m]
    fsap = fsap[m]
    F = F[m]

    # Normalize
    med = np.median(fsap)
    fsap /= med
    F /= med

    # Estimate flux uncertainty
    yerr = np.nanmedian(np.abs(np.diff(fsap)))

    return t, F, yerr
