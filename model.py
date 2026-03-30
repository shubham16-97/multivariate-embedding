#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:27:27 2026

@author: shubhamkukreja
"""
from embedding import Embedding
from prediction import Prediction
from inversion import Inversion
import numpy as np



class TimeSeriesModel:
    def __init__(self, delays, dimensions, bandwidth=0.5):
        self.embedding = Embedding(delays, dimensions)
        self.predictor = Prediction(bandwidth)
        self.inversion = Inversion(delays, dimensions)

    def fit(self, X):
        self.Z = self.embedding.transform(X)
        self.predictor.fit(self.Z)

    def predict(self, steps=1):
        z_t = self.Z[-1]
        preds = []

        for _ in range(steps):
            z_t = self.predictor.predict_next(z_t, k_neighbors=10)
            preds.append(z_t)

        return np.array(preds)

    def reconstruct(self, Z_full):
        return self.inversion.reconstruct(Z_full)