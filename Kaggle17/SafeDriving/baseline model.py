# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:29:52 2017

@author: LUI01
"""

import pandas as pd
import numpy as np
import functions
import matplotlib.pyplot as plt
import xgboost as xgb

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(train.shape)
print(test.shape)

"""
(595212, 59)
(892816, 58)
"""

## Check class balance

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

# 0.0364475178592

## Baseline Model xgb

"""
1. Baseline Model Fitting: No ID, no extra features.

"""
col = list(train.columns)[2:]

dtrain = xgb.DMatrix(train[col],train['target'])
dtest = xgb.DMatrix(test[col])

xgb_params = {'eta': 0.02, 
              'max_depth': 4, 
              'subsample': 0.9, 
              'colsample_bytree': 0.9, 
              'objective': 'binary:logistic', 
              'eval_metric': ['auc','logloss'], 
              'seed': 42, 
              'silent': True}

xgb_cvalid = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)#,eval_metric=gini_xgb)



"""
[0]     train-auc:0.59546       test-auc:0.590077
[50]    train-auc:0.622645      test-auc:0.616775
[100]   train-auc:0.625898      test-auc:0.618718
[150]   train-auc:0.632411      test-auc:0.623683
[200]   train-auc:0.6398        test-auc:0.628659
[250]   train-auc:0.645803      test-auc:0.632414
[300]   train-auc:0.650694      test-auc:0.634742
[350]   train-auc:0.655008      test-auc:0.636375
[400]   train-auc:0.658588      test-auc:0.63731
[450]   train-auc:0.661943      test-auc:0.637945
[500]   train-auc:0.66505       test-auc:0.638314
[550]   train-auc:0.66798       test-auc:0.638604
[600]   train-auc:0.670741      test-auc:0.638834
[650]   train-auc:0.673305      test-auc:0.639047
"""
len(xgb_cvalid)
"""
ROUNDS: 657
"""
#xgb_cvalid[['train-auc-mean', 'test-auc-mean']].plot()

model = xgb.train(xgb_params,dtrain,num_boost_round=len(xgb_cvalid))
FI = get_feature_importance(model)

"""
 Feature  Score
0        ps_car_13    969
19       ps_ind_03    778
3        ps_reg_03    703
2    ps_ind_05_cat    616
8        ps_ind_15    494
39       ps_ind_01    400
11       ps_reg_01    381
20       ps_reg_02    353
7    ps_car_01_cat    345
29       ps_car_14    337
1    ps_ind_17_bin    325
15   ps_car_11_cat    238
6    ps_car_07_cat    227
28   ps_car_09_cat    215
10       ps_car_15    214
22   ps_ind_02_cat    208
"""
pred = model.predict(dtest)

sub = pd.DataFrame({'id':test['id'],'target':pred})
sub.to_csv('submission/baseline_xgb_auc_.6390.csv',index=False)

"""
2. Baseline Model 2: Add ID as a feature.
"""

col.append('id')

dtrain = xgb.DMatrix(train[col],train['target'])
dtest = xgb.DMatrix(test[col])

xgb_params = {'eta': 0.02, 
              'max_depth': 4, 
              'subsample': 0.9, 
              'colsample_bytree': 0.9, 
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': 42, 
              'silent': True}

xgb_cvalid = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

"""
[500]   train-auc:0.66558       test-auc:0.638433
[550]   train-auc:0.668766      test-auc:0.638737
[600]   train-auc:0.671668      test-auc:0.638907
[650]   train-auc:0.674416      test-auc:0.639002
"""

model = xgb.train(xgb_params,dtrain,num_boost_round=len(xgb_cvalid))
FI = get_feature_importance(model)

"""
Feature  Score
0        ps_car_13    931
11       ps_ind_03    797
3        ps_reg_03    702
2    ps_ind_05_cat    629
8        ps_ind_15    458
19              id    441
12       ps_ind_01    373
1    ps_ind_17_bin    330
29       ps_reg_01    322
7    ps_car_01_cat    315
20       ps_reg_02    312
25       ps_car_14    299
6    ps_car_07_cat    223
30   ps_car_09_cat    205
"""

pred = model.predict(dtest)

sub = pd.DataFrame({'id':test['id'],'target':pred})
sub.to_csv('submission/baseline_xgb_auc_add_id_.6390.csv',index=False)


#