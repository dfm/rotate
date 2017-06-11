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
