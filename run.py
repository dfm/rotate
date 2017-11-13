#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import argparse
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from rotate.k2 import get_light_curve
from rotate.model import RotationModel

parser = argparse.ArgumentParser()
parser.add_argument("epicid", type=int, help="the target ID")
args = parser.parse_args()

# Load the data
epicid = args.epicid
t, F, yerr = get_light_curve(epicid)

# Set up the model. This takes a second because it builds the PLD basis
model = RotationModel(t, F, yerr)

def plot_estimators():  # NOQA
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    ax = axes[0]
    freq, power = model.lomb_scargle_result["periodogram"]
    period = 1.0 / freq
    ax.loglog(period, power, "k")
    for peak in model.lomb_scargle_result["peaks"]:
        ax.axvline(peak["period"], color="k", alpha=0.3, lw=2)
    ax.annotate("periodogram", xy=(0, 1), xycoords="axes fraction",
                ha="left", va="top", xytext=(5, -5),
                textcoords="offset points")
    ax.set_xlim(model.min_period, model.max_period)

    ax = axes[1]
    tau, acor = model.autocorr_result["autocorr"]
    ax.plot(tau, acor, "k")
    for peak in model.autocorr_result["peaks"]:
        period = peak["period"]
        t = period
        while t < model.t.max():
            ax.axvline(t, color="k", alpha=0.3, lw=2)
            t += period
    ax.set_xlim(0, model.t.max() - model.t.min())
    ax.annotate("autocorr function", xy=(1, 0), xycoords="axes fraction",
                ha="right", va="top", xytext=(-5, 5),
                textcoords="offset points")

    ax.set_xlabel("period [days]")
    return fig

fig = plot_estimators()
fig.savefig("{0}-est0.pdf".format(epicid), bbox_inches="tight")
plt.close(fig)

# Initial fit
model.optimize()

# Fit for a range of periods
def fit_for_period(period):  # NOQA
    model.set_default()
    model.period = period
    model.rotation_gp.freeze_parameter("kernel:terms[2]:log_P")
    soln = model.optimize_gp()
    model.rotation_gp.thaw_parameter("kernel:terms[2]:log_P")
    soln = model.optimize_gp()
    return (soln.fun, soln)

model.use_rotation_gp()  # NOQA
periods = [p["period"] for p in model.lomb_scargle_result["peaks"]]
periods += [p["period"] for p in model.autocorr_result["peaks"]]
with Pool() as pool:
    results = list(map(fit_for_period, periods))

# Find the best period
results = sorted(results, key=lambda o: o[0])
model.gp.set_parameter_vector(results[0][-1].x)

# Run one final fit
model.optimize()

def plot_data():  # NOQA
    # Plot the detrended data
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    xx = np.linspace(model.t.min(), model.t.max(), len(model.t))
    gp_pred = model.gp.predict(model.fdet, xx, return_cov=False)

    ax = axes[0]
    ax.plot(model.t, model.fsap, ".k", alpha=0.5, ms=4, mec="none")
    ax.plot(xx, gp_pred, lw=0.75)
    ax.plot(model.t, model.get_pld_model() + 1.0, alpha=0.3, lw=1)
    ax.annotate("raw", xy=(0, 1), xycoords="axes fraction",
                ha="left", va="top", xytext=(5, -5),
                textcoords="offset points")
    ax.set_title("EPIC {0}".format(epicid))

    ax = axes[1]
    ax.plot(model.t, model.fdet, ".k", alpha=0.5, ms=4, mec="none")
    ax.plot(xx, gp_pred, lw=1)
    ax.set_xlim(model.t.min(), model.t.max())
    ax.annotate("de-trended", xy=(0, 1), xycoords="axes fraction",
                ha="left", va="top", xytext=(5, -5),
                textcoords="offset points")

    ax.set_xlabel("time [days]")
    return fig

fig = plot_data()  # NOQA
fig.savefig("{0}-data.pdf".format(epicid), bbox_inches="tight")
plt.close(fig)

model.update_estimators()

fig = plot_estimators()
fig.savefig("{0}-est.pdf".format(epicid), bbox_inches="tight")
plt.close(fig)
