#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import json
import pickle
import argparse
from multiprocessing import Pool

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt

from rotate.k2 import get_light_curve
from rotate.model import RotationModel

parser = argparse.ArgumentParser()
parser.add_argument("epicid", type=int, help="the target ID")
parser.add_argument("-o", "--output", default="output",
                    help="the name of the output directory")
parser.add_argument("-p", "--progress", action="store_true",
                    help="show a progress bar?")
args = parser.parse_args()

def format_filename(fn):  # NOQA
    return os.path.join(args.output, "{0}".format(args.epicid), fn)

if os.path.exists(format_filename("corner.png")):
    print("skipping...")
    sys.exit(0)

os.makedirs(format_filename(""), exist_ok=True)  # NOQA

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
    p = len(model.t)**2*model.gp.kernel.get_psd(2*np.pi*freq)/(2*np.pi)
    ax.plot(period, p)
    ax.annotate("periodogram", xy=(0, 1), xycoords="axes fraction",
                ha="left", va="top", xytext=(5, -5),
                textcoords="offset points")
    ax.set_xlim(model.min_period, model.max_period)

    ax = axes[1]
    tau, acor = model.autocorr_result["autocorr"]
    ax.plot(tau, acor, "k")
    for peak in model.lomb_scargle_result["peaks"]:
        ax.axvline(peak["period"], color="k", alpha=0.3, lw=2)
    for peak in model.autocorr_result["peaks"]:
        period = peak["period"]
        t = period
        while t < model.t.max():
            ax.axvline(t, color="k", lw=0.75)
            t += period
    ax.plot(tau, len(model.t)**2*model.gp.kernel.get_value(tau))
    ax.set_xlim(0, model.t.max() - model.t.min())
    ax.annotate("autocorr function", xy=(1, 1), xycoords="axes fraction",
                ha="right", va="top", xytext=(-5, -5),
                textcoords="offset points")

    ax.set_xlabel("period [days]")
    return fig

fig = plot_estimators()  # NOQA
fig.savefig(format_filename("est0.pdf"), bbox_inches="tight")
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
fig.savefig(format_filename("data.pdf"), bbox_inches="tight")
plt.close(fig)

model.update_estimators()

fig = plot_estimators()
fig.savefig(format_filename("est.pdf"), bbox_inches="tight")
plt.close(fig)

# Save the maximum likelihood model
with open(format_filename("model.pkl"), "wb") as f:
    pickle.dump(model, f, -1)

# Save maximum likelihood results
with open(format_filename("summary.json"), "w") as f:
    json.dump(dict(
        epicid=epicid,
        lomb_scargle_period=model.lomb_scargle_period,
        autocorr_period=model.autocorr_period,
    ), f, sort_keys=True, indent=2)

# Run MCMC
def log_prob(params):  # NOQA
    model.gp.set_parameter_vector(params)
    lp = model.gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf, lp, model.period
    ll = model.gp.log_likelihood(model.fdet, quiet=True)
    if not np.isfinite(ll):
        return -np.inf, lp, model.period
    return ll + lp, lp, model.period

init0 = model.gp.get_parameter_vector()  # NOQA
init = init0 + 1e-5*np.random.randn(64, len(init0))
lp = np.array(list(map(log_prob, init)))[:, 0]
m = ~np.isfinite(lp)
while np.any(m):
    init[m] = init0 + 1e-5*np.random.randn(m.sum(), len(init0))
    lp[m] = np.array(list(map(log_prob, init[m])))[:, 0]
    m = ~np.isfinite(lp)
nwalkers, ndim = init.shape

# Backend
# Don't forget to clear it in case the file already exists
filename = format_filename("chain.h5")  # NOQA
backend = emcee.backends.HDFBackend(filename, retries=10)
backend.reset(nwalkers, ndim)

# Proposals
moves = [
    emcee.moves.StretchMove(randomize_split=True),
    emcee.moves.DEMove(1.0, randomize_split=True),
    emcee.moves.DESnookerMove(randomize_split=True),
]

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool,
                                    moves=moves, backend=backend)

    old_tau = np.inf
    autocorr = []
    converged = False
    for iteration in range(20):
        sampler.run_mcmc(init, 500, thin_by=10, progress=args.progress)
        init = None

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr.append(np.mean(tau))
        print(autocorr[-1])

        # Check convergence
        converged = np.all(tau * 1000 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.1)
        if converged:
            break
        old_tau = tau

    if converged:
        print("converged")
    else:
        print("not converged")

flatchain = sampler.get_chain(discard=2*int(np.max(tau)),
                              thin=int(np.min(tau)), flat=True)
fig = corner.corner(flatchain)
fig.savefig(format_filename("corner.png"))
plt.close(fig)
