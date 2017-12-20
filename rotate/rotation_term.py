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
            np.exp(log_a) * (mx + b) / (1.0 + mx + b), c,
        )

    def get_complex_coefficients(self, params):
        log_a, log_b, log_f, log_P = params
        b = np.exp(log_b)
        f = np.exp(log_f)
        mx = np.sqrt(1+f**2)
        c = 2*np.pi*f*np.exp(-log_P) / (1.0 + b + mx)
        factor = np.exp(log_a) / (1.0 + mx + b)
        return (
            factor, factor * np.exp(log_f),
            c, 2*np.pi*np.exp(-log_P),
        )


class MixtureOfSHOsTerm(terms.Term):
    parameter_names = ("log_S0", "log_Q1", "mix_par", "log_Q2", "log_P")

    def get_real_coefficients(self, params):
        log_S0, log_Q1, log_f, log_Q2, log_period = params

        Q = np.exp(log_Q1)
        if Q >= 0.5:
            a, c = np.empty(0), np.empty(0)
        else:
            S0 = np.exp(log_S0)
            w0 = 2.0 * np.pi * np.exp(-log_period)
            f = np.sqrt(1.0 - 4.0 * Q**2)
            a = 0.5*S0*w0*Q*np.array([1.0+1.0/f, 1.0-1.0/f])
            c = 0.5*w0/Q*np.array([1.0-f, 1.0+f])

        Q = np.exp(log_Q2)
        if Q >= 0.5:
            return a, c

        mix = 1.0 / (1.0 + np.exp(-self.mix_par))
        S0 = mix*np.exp(log_S0)
        w0 = np.pi * np.exp(-log_period)
        f = np.sqrt(1.0 - 4.0 * Q**2)
        # Dealing with autograd's lack of append
        a = list(a) + [v for v in 0.5*S0*w0*Q*np.array([1.0+1.0/f, 1.0-1.0/f])]
        c = list(c) + [v for v in 0.5*w0/Q*np.array([1.0-f, 1.0+f])]
        return np.array(a), np.array(c)

    def get_complex_coefficients(self, params):
        log_S0, log_Q1, log_f, log_Q2, log_period = params

        Q = np.exp(log_Q1)
        if Q < 0.5:
            a, b, c, d = [], [], [], []
        else:
            S0 = np.exp(log_S0)
            w0 = 2.0 * np.pi * np.exp(-log_period)
            f = np.sqrt(4.0 * Q**2-1)
            a = [S0 * w0 * Q]
            b = [S0 * w0 * Q / f]
            c = [0.5 * w0 / Q]
            d = [0.5 * w0 / Q * f]

        Q = np.exp(log_Q2)
        if Q < 0.5:
            return np.array(a), np.array(b), np.array(c), np.array(d)

        mix = 1.0 / (1.0 + np.exp(-self.mix_par))
        S0 = mix*np.exp(log_S0)
        w0 = np.pi * np.exp(-log_period)
        f = np.sqrt(4.0 * Q**2-1)
        a = a + [S0 * w0 * Q]
        b = b + [S0 * w0 * Q / f]
        c = c + [0.5 * w0 / Q]
        d = d + [0.5 * w0 / Q * f]
        return np.array(a), np.array(b), np.array(c), np.array(d)

    def log_prior(self):
        lp = super(MixtureOfSHOsTerm, self).log_prior()
        if not np.isfinite(lp):
            return -np.inf
        mix = 1.0 / (1.0 + np.exp(-self.mix_par))
        return lp + np.log(mix) + np.log(1.0 - mix)


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
                a1 * (mx1 + b1) / (1.0 + mx1 + b1),
                a2 * (mx2 + b2) / (1.0 + mx2 + b2),
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
        factor1 = a1 / (1.0 + mx1 + b1)

        a2 = np.exp(log_a1) * mix
        b2 = np.exp(log_b2)
        f2 = np.exp(log_f2)
        mx2 = np.sqrt(1+f2**2)
        c2 = 4*np.pi*f2*np.exp(-log_P) / (1.0 + b2 + mx2)
        factor2 = a2 / (1.0 + mx2 + b2)

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

    def grad_log_prior(self):
        g = np.zeros(self.full_size)
        theta = np.exp(-self.mix_par)
        g[4] += (theta - 1) / (1 + theta)
        return g[self.unfrozen_mask]
