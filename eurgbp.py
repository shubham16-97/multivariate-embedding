#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 04:57:01 2026

@author: shubhamkukreja
"""

import pandas as pd

df = pd.read_csv('fx_data.csv')  # important

new_df = df.assign(
    eurgbp = df['eurusd curncy'] / df['gbpusd curncy']
)[['date', 'eurgbp']]

new_df.to_csv('fx_eurgbp.csv',index = False)
#%%
# new_df_test = pd.read_csv('fx_eurgbp.csv')