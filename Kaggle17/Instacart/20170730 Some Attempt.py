# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 01:44:52 2017

@author: Jiashen Liu

@Purpose: Quickily change threshold and submit two times
"""

from functions_in import *
import pandas as pd
import numpy as np

df = pd.read_csv('save_test_for_exp_3.csv')
submission = pd.read_csv('data/sample_submission.csv')
mean_prob = df.groupby(['order_id']).mean().reset_index()
mean_prob = mean_prob[['order_id','reordered']]
mean_prob.columns = ['order_id','prob_mean']

df = df.merge(mean_prob,on='order_id',how='left')

d = dict()
for row in df.itertuples():
    if row.reordered >= row.prob_mean:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)
for order in submission.order_id:
    if order not in d:
        d[order] = 'None'
sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']

sub.to_csv('submission/20170730_sub_63_features_full_data_depth6_eta0.1_500rounds_mean_order_threshold.csv',index=False)

sub = sub_file(df,0.21,submission)

sub.to_csv('submission/20170730_sub_63_features_full_data_depth6_eta0.1_500rounds_0.21_threshold.csv',index=False)