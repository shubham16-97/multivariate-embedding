#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:21:46 2026

@author: shubhamkukreja
"""

import numpy as np

class Prediction:
    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth
        self.Z = None

    def fit(self, Z: np.ndarray):
        self.Z = Z

    def _kernel(self, distances):
        return np.exp(-(distances ** 2) / (2 * self.bandwidth ** 2))

    def predict_next(self, z_t: np.ndarray, k_neighbors: int = None):
        Z = self.Z
        
        # print(k_neighbors) # for debugging
        
        # Compute distances
        distances = np.linalg.norm(Z - z_t, axis=1)

        # Exclude last point (no forward mapping)
        valid_idx = np.arange(len(Z) - 1)

        distances = distances[valid_idx]
        Z_valid = Z[valid_idx]
        Z_next = Z[valid_idx + 1]

        # Select neighbors
        if k_neighbors:
            idx = np.argsort(distances)[:k_neighbors]
            distances = distances[idx]
            Z_valid = Z_valid[idx]
            Z_next = Z_next[idx]

        # Compute weights
        weights = self._kernel(distances)
        weights /= np.sum(weights) + 1e-8 #epsilon = 1e-8 added for stabilization

        # Kernel regression
        displacements = Z_next - Z_valid
        z_next =  z_t+np.sum(weights[:, None] * displacements, axis=0)

        return z_next