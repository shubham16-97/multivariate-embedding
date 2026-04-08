#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:08:16 2026

@author: shubhamkukreja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#%%
fx_data = pd.read_csv('fx_audnzd.csv')
Preds = np.load('Preds_audnzd.npy')
Dims = np.load('Dims_audnzd.npy')
Delays = np.load('Delays_audnzd.npy')
#%%
df_pred = pd.DataFrame(Preds, columns=['audnzd_predicted'])

dates = fx_data.iloc[510:, 0].reset_index(drop=True)
df_pred['date'] = dates

df_pred['date'] = pd.to_datetime(df_pred['date'])
df_pred = df_pred.set_index('date')
#%%
df_dim = pd.DataFrame(Dims, columns=['dimension'])
df_dim['date'] = dates
df_dim['date'] = pd.to_datetime(df_dim['date'])
df_dim = df_dim.set_index('date')

df_delay = pd.DataFrame(Delays, columns=['delay'])
df_delay['date'] = dates
df_delay['date'] = pd.to_datetime(df_delay['date'])
df_delay = df_delay.set_index('date')
#%%
df_actual = fx_data.iloc[510:].reset_index(drop=True)
df_actual['date'] = pd.to_datetime(df_actual['date'])
df_actual = df_actual.set_index('date')

#%%
PIP = 0.0001
df_pred['returns_pip'] = df_pred['audnzd_predicted'].diff() / PIP
#%%
df_actual['returns_pip'] = df_actual['audnzd'].diff() / PIP
#%%
# df_pred['returns_pip'].plot()
df_dim['dimension'].plot(ylabel='d')
#%%
df_delay['delay'].plot(ylabel=r'$\tau$')
#%%
# fig = plt.figure()
# plt.xticks(rotation=45)
# plt.plot(df_actual.index, df_actual['audnzd'], label='Actual')
# plt.plot(df_pred.index, df_pred['audnzd_predicted'], label='Predicted')
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()
#%%
fig, ax = plt.subplots()
# Plot y=x by starting at (0,0) with a slope of 1
plt.xticks(rotation=45)
plt.plot(df_pred['audnzd_predicted'], df_actual['audnzd'])
ax.axline((0, 0), slope=1, color='red', linestyle='--',linewidth = 2.0,label = 'y = x')
plt.ylim(1,1.3)
plt.xlim(1,1.3)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend()
plt.show()
#%%
fig = plt.figure()
plt.xticks(rotation=45)
plt.plot(df_actual.index, df_actual['returns_pip'], label='Actual')
plt.plot(df_pred.index, df_pred['returns_pip'], label='Predicted')
plt.legend(fontsize = '12')
plt.xlabel('Date')
plt.ylabel('Pips')
plt.show()

#%%
l2 = np.linalg.norm(df_actual['audnzd'] - df_pred['audnzd_predicted'])
print(l2)
rmse = l2/np.sqrt(len(df_actual))
print(rmse)
#%%
fig = plt.figure()
plt.xticks(rotation=45)
plt.plot(df_actual.index, np.abs(df_actual['audnzd']-df_pred['audnzd_predicted']))
plt.xlabel('Date')
plt.ylabel('Absolute difference')
plt.show()
