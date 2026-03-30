#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:56:36 2026

@author: shubhamkukreja
"""
from model import TimeSeriesModel
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd
from nolitsa import dimension, delay
from scipy.signal import argrelextrema
plt.rcParams["figure.figsize"] = (7,7)
plt.rcParams.update({'font.size': 26})


def main():
    
    # Load data
    fx_data = pd.read_csv('fx_data.csv')
    fx_data_modif = fx_data.iloc[10:] # Skipping 2008 data
    min_window = 500
    taus = []
    dims = []
    preds = []
    dim_range = np.arange(1, 10)
    
    for t in range(min_window, min_window+10):
       #initializing
        data_list = []   
        tau_temp = []
        dim_temp = []
        for n in range(0,2):

            x = fx_data_modif.iloc[:t,n+1].values

            if len(x) < 50:
                tau_temp.append(1)
                dim_temp.append(1)
                continue

            # Delay
            mi = delay.dmi(x, maxtau=100)
            mins = argrelextrema(mi, np.less)[0]
            tau_opt = mins[0] if len(mins) > 0 else 1
            tau_temp.append(tau_opt)

            # FNN
            fnn_vals, _, _ = dimension.fnn(x, tau=tau_opt, dim=dim_range)

            idx = np.where(fnn_vals < 0.01)[0]
            dim_opt = dim_range[idx[0]] if len(idx) > 0 else dim_range[-1]
            dim_temp.append(dim_opt)

            data_list.append(x)

        taus.append(tau_temp)
        dims.append(dim_temp)

        # Align
        min_len = min(len(arr) for arr in data_list)
        data_list = [arr[-min_len:] for arr in data_list]

        X = np.vstack(data_list).T

        model = TimeSeriesModel(
            delays=tau_temp,
            dimensions=dim_temp,
            bandwidth=0.5
        )

        model.fit(X)

        Z_pred = model.predict(steps=1)
        Z_full = np.vstack([model.Z, Z_pred])
        X_rec = model.reconstruct(Z_full)
        preds.append(X_rec[-1])
    return preds, dims, taus


if __name__ == '__main__':
    Preds, Dims, Delays = main()
#%%
# Dims = np.array(Dims)
# Delays = np.array(Delays)
# #%%
# fig = plt.figure()
# plt.ylabel('d')
# plt.xlabel('t')
# plt.scatter(np.arange(1,11),Dims[:,0],label='audusd')
# plt.scatter(np.arange(1,11),Dims[:,1],label='nzusd')
# plt.legend()
# plt.show()
# #%%
# fig = plt.figure()
# plt.ylabel(r'$\tau$')
# plt.xlabel('t')
# plt.scatter(np.arange(1,11),Delays[:,0],label='audusd')
# plt.scatter(np.arange(1,11),Delays[:,1],label='nzusd')
# plt.legend()
# plt.show()    