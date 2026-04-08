#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 02:08:26 2026

@author: shubhamkukreja
"""

import pandas as pd

df = pd.read_csv('fx_data.csv')  # important

new_df = df.assign(
    audnzd = df['audusd curncy'] / df['nzdusd curncy']
)[['date', 'audnzd']]

new_df.to_csv('fx_audnzd.csv',index = False)
#%%
# new_df_test = pd.read_csv('fx_audnzd.csv')
