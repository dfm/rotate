# -*- coding: utf-8 -*-

from __future__ import division, print_function

from celerite import terms
import autograd.numpy as np

__all__ = ["RotationTerm"]


class RotationTerm(terms.Term):
    parameter_names = ("log_a", "log_b", "log_f", "log_P")

    def get_real_coefficients(self, params):
        log_a, log_b, log_f, log_P = params
        b = np.exp(log_b)
        f = np.exp(log_f)
        mx = np.sqrt(1+f**2)
        c = 2*np.pi*f*np.exp(-log_P) / (1.0 + b + mx)
        return (
            np.exp(log_a) * (mx + b) / (2.0*mx + b), c,
        )

    def get_complex_coefficients(self, params):
        log_a, log_b, log_f, log_P = params
        b = np.exp(log_b)
        f = np.exp(log_f)
        mx = np.sqrt(1+f**2)
        c = 2*np.pi*f*np.exp(-log_P) / (1.0 + b + mx)
        factor = np.exp(log_a) / (2.0*mx + b)
        return (
            factor, factor * np.exp(log_f),
            c, 2*np.pi*np.exp(-log_P),
        )


class MixtureTerm(terms.Term):
    parameter_names = ("log_a1", "log_b1", "log_f1", "log_P",
                       "mix_par", "log_b2", "log_f2")

    def get_real_coefficients(self, params):
        log_a1, log_b1, log_f1, log_P, mix_par, log_b2, log_f2 = params

        mix = 1.0 / (1.0 + np.exp(-mix_par))
        a1 = np.exp(log_a1)
        b1 = np.exp(log_b1)
        f1 = np.exp(log_f1)
        mx1 = np.sqrt(1+f1**2)
        c1 = 2*np.pi*f1*np.exp(-log_P) / (1.0 + b1 + mx1)

        a2 = np.exp(log_a1) * mix
        b2 = np.exp(log_b2)
        f2 = np.exp(log_f2)
        mx2 = np.sqrt(1+f2**2)
        c2 = 4*np.pi*f2*np.exp(-log_P) / (1.0 + b2 + mx2)

        return (
            np.array([
                a1 * (mx1 + b1) / (2.0*mx1 + b1),
                a2 * (mx2 + b2) / (2.0*mx2 + b2),
            ]),
            np.array([c1, c2]),
        )

    def get_complex_coefficients(self, params):
        log_a1, log_b1, log_f1, log_P, mix_par, log_b2, log_f2 = params

        mix = 1.0 / (1.0 + np.exp(-mix_par))
        a1 = np.exp(log_a1)
        b1 = np.exp(log_b1)
        f1 = np.exp(log_f1)
        mx1 = np.sqrt(1+f1**2)
        c1 = 2*np.pi*f1*np.exp(-log_P) / (1.0 + b1 + mx1)
        factor1 = a1 / (2.0*mx1 + b1)

        a2 = np.exp(log_a1) * mix
        b2 = np.exp(log_b2)
        f2 = np.exp(log_f2)
        mx2 = np.sqrt(1+f2**2)
        c2 = 4*np.pi*f2*np.exp(-log_P) / (1.0 + b2 + mx2)
        factor2 = a2 / (2.0*mx2 + b2)

        return (
            np.array([factor1, factor2]),
            np.array([factor1*np.exp(log_f1), factor2*np.exp(log_f2)]),
            np.array([c1, c2]),
            np.array([2*np.pi*np.exp(-log_P), 4*np.pi*np.exp(-log_P)])
        )

    def log_prior(self):
        lp = super(MixtureTerm, self).log_prior()
        if not np.isfinite(lp):
            return -np.inf
        mix = 1.0 / (1.0 + np.exp(-self.mix_par))
        return lp + np.log(mix) + np.log(1.0 - mix)
