# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from celerite import modeling

__all__ = ["PolynomialModel"]


class PolynomialModel(modeling.ModelSet):

    def __init__(self, gp, sections, t, y):
        self.t = t
        self.y = y
        self.A = A
        self.AT = A.T
        self.S = np.empty((A.shape[1], A.shape[1]))
        self.ATalpha = np.empty(A.shape[1])
        super(EverestModel, self).__init__([("gp", gp), ("lam", lam)])

    def get_weights(self, m=None):
        if m is None:
            m = np.ones(len(self.t), dtype=bool)

        gp = self.models["gp"]
        log_lam = self.models["lam"]

        gp.compute(self.t[m])
        alpha = gp.apply_inverse(self.y[m])[:, 0]
        KinvA = gp.apply_inverse(self.A[m])
        S = np.dot(self.AT[:, m], KinvA)
        S[np.diag_indices_from(S)] += np.exp(-log_lam.get_diagonal())

        factor = cho_factor(S, overwrite_a=True, check_finite=False)
        ATalpha = np.dot(self.AT[:, m], alpha)
        return cho_solve(factor, ATalpha, overwrite_b=True, check_finite=False)

    def alpha(self, m=None):
        if m is None:
            m = np.ones(len(self.t), dtype=bool)

        gp = self.models["gp"]
        log_lam = self.models["lam"]

        gp.compute(self.t[m])
        alpha = gp.apply_inverse(self.y[m])[:, 0]
        KinvA = gp.apply_inverse(self.A[m])
        self.S[:, :] = np.dot(self.AT[:, m], KinvA)
        self.S[np.diag_indices_from(self.S)] += np.exp(-log_lam.get_diagonal())

        factor = cho_factor(self.S, overwrite_a=True, check_finite=False)
        half_log_det = 0.5 * gp.solver.log_determinant()
        half_log_det += np.sum(np.log(np.diag(factor[0])))

        self.ATalpha[:] = np.dot(self.AT[:, m], alpha)
        term2 = np.dot(KinvA, cho_solve(factor, self.ATalpha,
                                        check_finite=False, overwrite_b=True))

        half_log_det = 0.5 * gp.solver.log_determinant()
        half_log_det += np.sum(np.log(np.diag(factor[0])))

        return alpha - term2, half_log_det

    def log_marginalized_likelihood(self, m=None):
        if m is None:
            m = np.ones(len(self.t), dtype=bool)
        try:
            alpha, half_log_det = self.alpha(m=m)
        except (np.linalg.LinAlgError, RuntimeError):
            return -np.inf
        return -0.5*np.dot(self.y[m], alpha) - half_log_det

    def predict(self, x=None, m=None):
        if m is None:
            m = np.ones(len(self.t), dtype=bool)
        if x is None:
            x = self.t
        gp = self.models["gp"]
        alpha, _ = self.alpha(m=m)
        K = gp.get_matrix(x, self.t[m])
        gp_pred = np.dot(K, alpha)
        log_lam = np.exp(self.models["lam"].get_diagonal())
        pld_pred = np.dot(np.dot(self.A, self.AT[:, m] * log_lam[:, None]),
                          alpha)
        return gp_pred, pld_pred
