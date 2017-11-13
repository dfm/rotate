# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RotationModel"]

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

import celerite
from celerite import modeling

from .pld import PLDModel
from .gp import get_simple_gp, get_rotation_gp
from .estimator import lomb_scargle_estimator, autocorr_estimator


class RotationModel(modeling.ModelSet):

    def __init__(self, t, F, yerr, min_period=0.1, max_period=40.0,
                 lomb_scargle_kwargs=None, autocorr_kwargs=None,
                 **pld_kwargs):
        self.t = np.array(t)
        self.F = np.array(F)
        self.fsap = np.sum(F, axis=1)
        self.yerr = yerr
        A = self.F / self.fsap[:, None]
        self.min_period = min_period
        self.max_period = max_period

        # Run 1st order PLD
        w = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, self.fsap-1.0))
        self.fdet = self.fsap - np.dot(A, w)

        self.update_estimators(lomb_scargle_kwargs, autocorr_kwargs)

        # Set up the PLD model
        pld = PLDModel(self.t, self.F / self.fsap[:, None], **pld_kwargs)

        # Set up the GP model:
        self.simple_gp = get_simple_gp(self.t, self.fsap, yerr)
        self.rotation_gp = get_rotation_gp(self.t, self.fsap, yerr,
                                           self.lomb_scargle_period,
                                           min_period, max_period)

        super(RotationModel, self).__init__([("gp", self.simple_gp),
                                             ("pld", pld)])

        # Save the default parameters
        self.default_pld_vector = \
            pld.get_parameter_vector(include_frozen=True)
        self.default_simple_vector = \
            self.simple_gp.get_parameter_vector(include_frozen=True)
        self.default_rotation_vector = \
            self.rotation_gp.get_parameter_vector(include_frozen=True)

        # Set up an optimization cache
        self.model_cache = []

    def update_estimators(self, lomb_scargle_kwargs=None,
                          autocorr_kwargs=None):
        # Esimate the periods
        if lomb_scargle_kwargs is None:
            lomb_scargle_kwargs = dict(filter_period=10.0)
        self.lomb_scargle_result = \
            lomb_scargle_estimator(self.t, self.fdet, self.yerr,
                                   self.min_period, self.max_period,
                                   **lomb_scargle_kwargs)
        peaks = self.lomb_scargle_result["peaks"]
        if len(peaks):
            self.lomb_scargle_period = peaks[0]["period"]
        else:
            self.lomb_scargle_period = self.max_period

        if autocorr_kwargs is None:
            autocorr_kwargs = {}
        self.autocorr_result = \
            autocorr_estimator(self.t, self.fdet, self.yerr,
                               self.min_period, self.max_period,
                               **autocorr_kwargs)
        peaks = self.autocorr_result["peaks"]
        if len(peaks):
            self.autocorr_period = peaks[0]["period"]
        else:
            self.autocorr_period = self.max_period

    def use_simple_gp(self):
        self.models["gp"] = self.simple_gp

    def use_rotation_gp(self):
        self.models["gp"] = self.rotation_gp

    def get_weights(self):
        log_lams = self.pld.get_parameter_vector()
        A = self.pld.A
        fsap = self.fsap
        gp = self.gp

        alpha = np.dot(A.T, gp.apply_inverse(fsap - gp.mean.value)[:, 0])
        ATKinvA = np.dot(A.T, gp.apply_inverse(A))
        S = np.array(ATKinvA)
        dids = np.diag_indices_from(S)
        for bid, (s, f) in enumerate(self.pld.block_inds):
            S[(dids[0][s:f], dids[1][s:f])] += np.exp(-log_lams[bid])
        factor = cho_factor(S, overwrite_a=True)
        alpha -= np.dot(ATKinvA, cho_solve(factor, alpha))
        for bid, (s, f) in enumerate(self.pld.block_inds):
            alpha[s:f] *= np.exp(log_lams[bid])
        return alpha

    def get_pld_model(self):
        return np.dot(self.pld.A, self.get_weights())

    def get_predictions(self):
        pld_pred = self.get_pld_model()
        gp_pred = self.gp.predict(self.fsap - pld_pred, return_cov=False)
        return pld_pred, gp_pred

    def log_likelihood(self):
        log_lams = self.pld.get_parameter_vector()
        A = self.pld.A
        fsap = self.fsap
        gp = self.gp

        r = fsap - gp.mean.value

        try:
            alpha = gp.apply_inverse(r)[:, 0]
        except celerite.solver.LinAlgError:
            return -np.inf

        value = np.dot(r, alpha)
        ATalpha = np.dot(A.T, alpha)

        try:
            KA = gp.apply_inverse(A)
        except celerite.solver.LinAlgError:
            return -np.inf

        S = np.dot(A.T, KA)

        dids = np.diag_indices_from(S)
        for bid, (s, f) in enumerate(self.pld.block_inds):
            S[(dids[0][s:f], dids[1][s:f])] += np.exp(-log_lams[bid])

        try:
            factor = cho_factor(S, overwrite_a=True)
            value -= np.dot(ATalpha, cho_solve(factor, ATalpha))
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf

        # Penalty terms
        log_det = 2*np.sum(np.log(np.diag(factor[0])))
        log_det += np.sum(log_lams * self.pld.nblocks)
        log_det += gp.solver.log_determinant()

        return -0.5 * (value + log_det)

    def nll(self, params):
        self.set_parameter_vector(params)
        ll = self.log_likelihood()
        if not np.isfinite(ll):
            ll = -1e10 + np.random.randn()
        return -ll

    @property
    def period(self):
        return np.exp(self.rotation_gp.kernel.get_parameter("terms[2]:log_P"))

    @period.setter
    def period(self, period):
        self.rotation_gp.kernel.set_parameter("terms[2]:log_P", np.log(period))

    def set_default(self):
        self.pld.set_parameter_vector(self.default_pld_vector,
                                      include_frozen=True)
        self.simple_gp.set_parameter_vector(self.default_simple_vector,
                                            include_frozen=True)
        self.rotation_gp.set_parameter_vector(self.default_rotation_vector,
                                              include_frozen=True)

    def optimize(self, **kwargs):
        init = self.get_parameter_vector()
        bounds = self.get_parameter_bounds()
        soln = minimize(self.nll, init, bounds=bounds, **kwargs)
        self.set_parameter_vector(soln.x)
        pld_pred = self.get_pld_model()
        self.fdet = self.fsap - pld_pred
        return soln

    def gp_grad_nll(self, params):
        self.gp.set_parameter_vector(params)
        gll = self.gp.grad_log_likelihood(self.fdet, quiet=True)
        if not np.isfinite(gll[0]):
            return (1e10 + np.random.randn(),
                    10000*np.random.randn(len(params)))
        return -gll[0], -gll[1]

    def optimize_gp(self, **kwargs):
        init = self.gp.get_parameter_vector()
        bounds = self.gp.get_parameter_bounds()
        soln = minimize(self.gp_grad_nll, init, bounds=bounds, jac=True,
                        **kwargs)
        self.gp.set_parameter_vector(soln.x)
        return soln
