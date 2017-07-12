# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from astropy.io import fits

from .pipeline import Pipeline

__all__ = ["get_light_curve"]

class EPIC(Pipeline):
    def process(self, epicid):
        pass


def _everest_url_and_fn(campaign, epicid):
    id_str = "{0:09d}".format(epicid)
    fn = "hlsp_everest_k2_llc_{0}-c{1:02d}_kepler_v2.0_lc.fits".format(
        id_str, campaign
    )
    url = "https://archive.stsci.edu/missions/hlsp/everest/v2/"
    url += "c{0:02d}/{1}00000/{2}/".format(campaign, id_str[:4], id_str[4:])
    return url + fn, fn

def get_light_curve(campaign, epicid, cache=False):
    url, fn = _everest_url_and_fn(campaign, epicid)
    with fits.open(url, cache=cache) as hdus:
        data = hdus[1].data
        hdr = hdus[1].header
        t = data["TIME"]
        q = data["QUALITY"]
        f = data["FLUX"]

        breaks = [0]
        for i in range(1, 100):
            k = "BRKPT{0:02d}".format(i)
            if k not in hdr:
                break
            breaks.append(hdr[k])
        breaks = np.array(breaks + [len(t) + 1], dtype=int)

    sections = np.zeros(len(t), dtype=int)
    for i in range(len(breaks) - 1):
        sections[breaks[i]:breaks[i+1]] = i

    m = np.isfinite(t) & np.isfinite(f) & (q == 0)
    sections = np.ascontiguousarray(sections[m], dtype=np.int)
    f = f[m]
    f = (f / np.median(f) - 1.0) * 100.0
    return (
        sections,
        np.ascontiguousarray(t[m], dtype=np.float64),
        np.ascontiguousarray(f, dtype=np.float64)
    )

class EverestLightCurve(Pipeline):
    pass
