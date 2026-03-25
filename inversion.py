#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:25:42 2026

@author: shubhamkukreja
"""

import numpy as np
from typing import List

class Inversion:
    def __init__(self, delays: List[int], dimensions: List[int]):
        self.delays = delays
        self.dimensions = dimensions
        self.D = len(delays)

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct multivariate time series.

        Output:
            X_rec: shape (T, D)
        """
        T_embed = Z.shape[0]
        # total_dim = sum(self.dimensions)

        # Recover total length
        max_lag = max((d_i - 1) * tau_i for d_i, tau_i in zip(self.dimensions, self.delays))
        T = T_embed + max_lag

        X_rec = np.zeros((T, self.D))

        col_start = 0

        for i in range(self.D):
            d = self.dimensions[i]
            tau = self.delays[i]

            block = Z[:, col_start : col_start + d]

            # First column fills main part
            X_rec[:T_embed, i] = block[:, 0]

            # Tail reconstruction
            last_row = block[-1]
            for j in range(1, d):
                idx = (T_embed - 1) + j * tau
                if idx < T:
                    X_rec[idx, i] = last_row[j]

            col_start += d

        return X_rec