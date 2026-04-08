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
import timeit
plt.rcParams["figure.figsize"] = (7,7)
plt.rcParams.update({'font.size': 26})


def main(fx_data_array, min_window,tstep, curr_idx, maxtau, bw,tol,d_upper):  
    # min_window = 500
    taus = []
    dims = []
    preds = []
    dim_range = np.arange(1, d_upper)

    
    for t in range(min_window, min_window+tstep):
       #initializing
        data_list = []   
        tau_temp = []
        dim_temp = []
        for n in range(0,curr_idx):
            # print('hello')
            # x = fx_data_modif.iloc[:t,n+1].values
            x =  fx_data_array[:t,n]

            if len(x) < 50:
                tau_temp.append(1)
                dim_temp.append(1)
                continue

            # Delay
            mi = delay.dmi(x, maxtau)
            mins = argrelextrema(mi, np.less)[0]
            tau_opt = mins[0] if len(mins) > 0 else 1
            tau_temp.append(tau_opt)
            # print('hello')
            # FNN
            fnn_vals, _, _ = dimension.fnn(x, tau=tau_opt, dim=dim_range)

            idx = np.where(fnn_vals < tol)[0]
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
            bandwidth=bw
        )

        model.fit(X)

        Z_pred = model.predict(steps=1)
        Z_full = np.vstack([model.Z, Z_pred])
        X_rec = model.reconstruct(Z_full)
        preds.append(X_rec[-1])
    return preds, dims, taus
    # return dims, taus


if __name__ == '__main__':
    # Load data
    # fx_data = pd.read_csv('fx_audnzd.csv')
    fx_data = pd.read_csv('fx_eurgbp.csv')
    idx = 1
    fx_data_np = fx_data.iloc[10:,1:idx+1].to_numpy()      # Skipping 2008 data 
    initial_window = 500
    T  = fx_data_np.shape[0]-initial_window
    # T = 1
    Max_delay = 100
    Bandwidth = 0.5
    nn_tol =0.01
    d_lim= 11
    tic = timeit.default_timer()
    Preds, Dims, Delays = main(fx_data_np,initial_window,T,idx,Max_delay,Bandwidth,nn_tol,d_lim)
    # Dims, Delays = main(fx_data_np,initial_window,T,idx,Max_delay,Bandwidth,nn_tol,d_lim)
    toc = timeit.default_timer()
    print(f"Time taken for calculation: {np.round(toc-tic,2)} seconds") 
    # np.save('Preds_audnzd.npy',Preds)
    # np.save('Dims_audnzd.npy',Dims)
    # np.save('Delays_audnzd.npy',Delays)

    



