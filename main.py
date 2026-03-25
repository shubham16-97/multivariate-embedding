#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:16:50 2026

@author: shubhamkukreja
"""

from model import TimeSeriesModel
import numpy  as np
import matplotlib.pyplot as plt

# Example multivariate time series (2D)
t = np.linspace(0, 20, 500)
X = np.vstack([
    np.sin(t),
    np.cos(t)
]).T  # shape (T, 2)

model = TimeSeriesModel(
    delays=[5, 5],
    dimensions=[3, 3],
    bandwidth=0.5
)

model.fit(X)

Z_pred = model.predict(steps=100)

Z_full = np.vstack([model.Z, Z_pred])

X_rec = model.reconstruct(Z_full)

print(X_rec.shape)
#%%
plt.figure()
plt.xlim(400,600)
plt.plot(X_rec[:,0])
plt.plot(X_rec[:,1])
# plt.plot(X_rec[500:,1])
plt.show()