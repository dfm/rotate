#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import json
# import pickle
import tqdm
import argparse
import numpy as np
from emcee.backends import HDFBackend

parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+", help="")
parser.add_argument("-o", "--output", default="periods.csv")
args = parser.parse_args()

with open(args.output, "w") as f:
    f.write("epicid,period,period_err_minus,period_err_plus\n")

names = None
for fn in tqdm.tqdm(args.filenames):
    dirname = os.path.split(os.path.abspath(fn))[0]

    # if names is None:
    #     with open(os.path.join(dirname, "model.pkl"), "rb") as f:
    #         model = pickle.load(f)
    #     names = model.gp.kernel.get_parameter_names()
    #     assert "log_P" in names[-1]

    with open(os.path.join(dirname, "summary.json"), "r") as f:
        summary = json.load(f)
    epicid = summary["epicid"]

    reader = HDFBackend(fn, read_only=True)
    tau = reader.get_autocorr_time(tol=0).min()
    chain = reader.get_chain(flat=True, discard=int(2.0*tau))
    period = np.percentile(np.exp(chain[:, -1]), [16, 50, 84])
    uncert = np.diff(period)

    with open(args.output, "a") as f:
        f.write("{0},{1},{2[0]},{2[1]}\n".format(epicid, period[1], uncert))
