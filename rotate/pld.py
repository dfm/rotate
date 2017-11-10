# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["PLDModel"]

from itertools import combinations_with_replacement

import numpy as np

from celerite import modeling


class PLDModel(modeling.Model):
    """A simple implementation of PLD based on everest

    Args:
        t (ndarray): The time stamp array.
        A (ndarray): The ``(ntimes, npix)`` matrix of pixel flux time series.
        order (int): The PLD order to use.
        maxn (int): The maximum number of PCA components to use for the higher
            order blocks.
        poly_order (int): The maximum order of the polynomial block.
        max_decay (int): The maximum lag to include in the initial decay model.

    """

    def __init__(self, t, A, order=3, maxn=32, poly_order=3, max_decay=10):
        self.order = order
        self.maxn = min(A.shape[1], maxn)

        # First order
        A = np.array(A[:, np.argsort(np.median(A, axis=0))[::-1]])
        A -= np.mean(A, axis=0)[None, :]

        # Higher order blocks
        blocks = [A]
        for order in range(2, self.order + 1):
            A2 = np.product(list(
                combinations_with_replacement(A[:, :maxn].T, order)), axis=1).T
            U, S, V = np.linalg.svd(A2-np.mean(A2, axis=0), full_matrices=True)
            block = U[:, :maxn] - np.mean(U[:, :maxn], axis=0)[None, :]
            blocks.append(block)

        # Polynimial block
        tt = 2*(t - t.min()) / (t.max() - t.min()) - 1
        blocks.append(np.vander(tt, poly_order + 1))

        # Initial decay block
        dt = t - t.min()
        decay = np.exp(-dt[:, None] / np.arange(1, max_decay+1, 1.0)[None, :])
        blocks.append(decay)

        # Combine the blocks.
        # block_inds tracks the beginning and end of each block in columns of A
        self.A = np.concatenate(blocks, axis=1)
        self.block_sizes = np.array([block.shape[1] for block in blocks])
        block_inds = np.append(0, np.cumsum(self.block_sizes))
        self.block_inds = list(zip(block_inds[:-1], block_inds[1:]))
        self.nblocks = len(self.block_sizes)

        params = dict(("log_lambda_{0}".format(i), -1.0)
                      for i in range(self.nblocks))
        self.parameter_names = tuple(sorted(params.keys()))
        params["bounds"] = [(-5.0, 5.0) for i in range(self.nblocks)]
        super(PLDModel, self).__init__(**params)
