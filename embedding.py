#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:23:42 2026

@author: shubhamkukreja
"""

import numpy as np
from typing import List


class Embedding:
    def __init__(self, delays: List[int], dimensions: List[int]):
        """
        delays: [\tau_1, \tau_2, ..., \tau_D]
        dimensions: [d_1, d_2, ..., d_D]
        """
        # The number of delays must equal number of dimensions
        assert len(delays) == len(dimensions), "Mismatch in dimensions"
        self.delays = delays
        self.dimensions = dimensions
        self.D = len(delays)

    def transform(self, X: np.ndarray) -> np.ndarray:# making sure ndarray is returned 
        """
        Multivariate Takens embedding.

        Input:
            X: shape (T, D)

        Output:
            Z: shape (T_embed, sum(d_i))
        """
        T, D = X.shape
        assert D == self.D, "Input dimension mismatch"

        # Compute max lag across all variables
        """
        t+(d_i-1)\tau_i \leq T-1 -> T_embed = T - max((d_i-1)\tau_i)
        """
        max_lag = max((d_i - 1) * tau_i for d_i, tau_i in zip(self.dimensions, self.delays))
        T_embed = T - max_lag

        if T_embed <= 0:
            raise ValueError("Time series too short for embedding")

        embedded_blocks = []

        # Build embedding for each variable
        for i in range(D):
            tau = self.delays[i]
            d = self.dimensions[i]

            Xi = X[:, i]
            block = np.zeros((T_embed, d))

            for j in range(d):
                block[:, j] = Xi[j * tau : j * tau + T_embed]

            embedded_blocks.append(block)

        # Concatenate all variables
        Z = np.hstack(embedded_blocks)

        return Z