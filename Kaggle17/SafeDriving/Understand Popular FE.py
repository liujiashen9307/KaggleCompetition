# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:41:29 2017

@author: Jiashen Liu

@purpose: restart, try to understand the popular Feature Engineering
"""


import pandas as pd
import numpy as np
import xgboost as xgb

def get_feature_importance_xgb(model):
    Importance = model.get_fscore()
    Importance = list(Importance.items())
    Feature= []
    Score = []
    for each in Importance:
        Feature.append(each[0])
        Score.append(each[1])
    df = pd.DataFrame({'Feature':Feature,'Score':Score}).sort_values(by=['Score'],ascending=[0])
    return df    

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

for c in train.select_dtypes(include=['float64']).columns:
    train[c]=train[c].astype(np.float32)
    test[c]=test[c].astype(np.float32)
for c in train.select_dtypes(include=['int64']).columns[2:]:
    train[c]=train[c].astype(np.int8)
    test[c]=test[c].astype(np.int8)
    

train['ps_car_13_x_ps_reg_03'] = train['ps_car_13'] * train['ps_reg_03']
test['ps_car_13_x_ps_reg_03'] = test['ps_car_13'] * test['ps_reg_03']

# (595212, 60)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

# (595212, 40)
col = list(test.columns)
col.remove('id')
dtrain = xgb.DMatrix(train[col],train['target'])
dtest = xgb.DMatrix(test[col])

params = {'eta': 0.02, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'silent': True}

xgb_cvalid = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

"""
[0]     train-auc:0.590514      test-auc:0.586669
[50]    train-auc:0.622148      test-auc:0.616959
[100]   train-auc:0.626099      test-auc:0.619495
[150]   train-auc:0.631506      test-auc:0.623716
[200]   train-auc:0.638375      test-auc:0.628693
[250]   train-auc:0.644034      test-auc:0.632159
[300]   train-auc:0.648847      test-auc:0.634628
[350]   train-auc:0.652989      test-auc:0.636523
[400]   train-auc:0.656639      test-auc:0.637771
[450]   train-auc:0.659633      test-auc:0.638672
[500]   train-auc:0.662311      test-auc:0.639202
[550]   train-auc:0.664799      test-auc:0.639615
[600]   train-auc:0.66704       test-auc:0.639913
[650]   train-auc:0.669235      test-auc:0.640168
[700]   train-auc:0.671391      test-auc:0.640456
"""

"""
717 Rounds
"""


xgb_model = xgb.train(params,dtrain,num_boost_round=730)
FI = get_feature_importance_xgb(xgb_model)

"""
 Feature  Score
0               ps_car_13   1115
8   ps_car_13_x_ps_reg_03   1087
11              ps_ind_03    938
2           ps_ind_05_cat    656
19              ps_ind_15    570
9               ps_ind_01    532
3               ps_reg_03    522
20              ps_car_14    437
24              ps_reg_01    426
6           ps_car_01_cat    407
7               ps_reg_02    382
1           ps_ind_17_bin    374
12          ps_car_11_cat    314
5           ps_car_06_cat    288
29              ps_car_15    271
22          ps_car_09_cat    255
13          ps_car_07_cat    248
17          ps_car_03_cat    211
28              ps_car_12    210
14          ps_ind_02_cat    195
"""

pred = xgb_model.predict(dtest)

sub = pd.DataFrame({'id':test['id'],'target':pred})

sub.to_csv('submission/baseline_less_feats_6404.csv',index=False)


"""
Part 2 tonight: average this one and the previous two.

"""

sub2 = pd.read_csv('submission/baseline_xgb_auc_.6390.csv')

new_target = 0.8*pred + 0.2*sub2['target']

sub_avg = pd.DataFrame({'id':test['id'],'target':new_target})

sub_avg.to_csv('submission/avg_two_baseline_8to2.csv',index=False)

dtrain2 = xgb.DMatrix(train[col])
cc = xgb_model.predict(dtrain2)
dd = [1 if each >= 0.5 else 0 for each in cc]

"""
Part 3: Try 2000 Rounds
"""
xgb_model2 = xgb.train(params,dtrain,num_boost_round=2000)

pred = xgb_model2.predict(dtest)
sub3 = pd.DataFrame({'id':test['id'],'target':pred})
sub3.to_csv('submission/baseline_less_feats_2000_rounds.csv',index=False)