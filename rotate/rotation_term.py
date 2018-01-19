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


class MixtureOfSHOsTerm(terms.SHOTerm):
    parameter_names = ("log_a", "log_Q1", "mix_par", "log_Q2", "log_P")

    def get_real_coefficients(self, params):
        return np.empty(0), np.empty(0)

    def get_complex_coefficients(self, params):
        log_a, log_Q1, mix_par, log_Q2, log_period = params

        Q = np.exp(log_Q2) + np.exp(log_Q1)
        log_Q1 = np.log(Q)
        P = np.exp(log_period)
        log_omega1 = np.log(4*np.pi*Q) - np.log(P) - 0.5*np.log(4.0*Q*Q-1.0)
        log_S1 = log_a - log_omega1 - log_Q1

        mix = -np.log(1.0 + np.exp(-mix_par))
        Q = np.exp(log_Q2)
        P = 0.5*np.exp(log_period)
        log_omega2 = np.log(4*np.pi*Q) - np.log(P) - 0.5*np.log(4.0*Q*Q-1.0)
        log_S2 = mix + log_a - log_omega2 - log_Q2

        c1 = super(MixtureOfSHOsTerm, self).get_complex_coefficients([
            log_S1, log_Q1, log_omega1,
        ])

        c2 = super(MixtureOfSHOsTerm, self).get_complex_coefficients([
            log_S2, log_Q2, log_omega2,
        ])

        return [np.array([a, b]) for a, b in zip(c1, c2)]

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
