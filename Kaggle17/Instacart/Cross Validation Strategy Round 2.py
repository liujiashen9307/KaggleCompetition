# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:38:53 2017

@author: Jiashen Liu

Purpose: To further test the cross-validation strategy with two submissions

"""

## Import functions
from functions_in import *
import pandas as pd
import numpy as np

data = pd.read_csv('data_bakup.csv')
train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
train['reordered'] = train['reordered'].fillna(0)
test['reordered'] = test['reordered'].fillna(0)
del data

from sklearn.model_selection import GroupKFold

kf = GroupKFold(n_splits=10) 

train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train, groups=train['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)

test_index = train_indexes[0]
train_index = test_indexes[0]


## Define the columns

#training = train.iloc[0:round(0.15*len(train)),:]
#testing = train.iloc[1271200:len(train),:]
training = train.iloc[train_index,:]
testing2 = train.iloc[test_index,:]
col = list(testing.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')

## Start Building XGB Model

import xgboost as xgb
dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(testing[col])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(dtrain, "train")]
bst = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=90, evals=watchlist, verbose_eval=10)

pred_test = bst.predict(dtest)

## Do Validation

valid_file = CV_file(testing)

testing['reordered'] = pred_test

pred_file = sub_file(testing,0.20,valid_file)

valid_file['products_o'] = valid_file['products_o'].apply(lambda x: string_to_list(x))
pred_file['products'] = pred_file['products'].apply(lambda x: string_to_list(x))

valid_file = valid_file.merge(pred_file,how='left',on = 'order_id')


### Test F1-Score

f1_score(valid_file['products'],valid_file['products_o'])

## 0.38854751961815781
## 0.38854751961815781
## Real: 0.3787403


## Make a submission

dtest = xgb.DMatrix(test[col])
pred = bst.predict(dtest)
test['reordered'] = pred
submission = pd.read_csv('data/sample_submission.csv')
sub = sub_file(test,0.21,submission)
sub.to_csv('CV_Round2_xgb.csv',index=False)

### Rethink on CV: When I am calculating the validation set, did I miss the cases when 'None'
### is the ground truth?

validation2 = CV_file2(testing2)

vv = validation2[validation2['products_o']=='None']

validation2['products_o'] = validation2['products_o'].apply(lambda x: string_to_list(x))

pred_file = sub_file(testing,0.20,validation2)

pred_file['products'] = pred_file['products'].apply(lambda x: string_to_list(x))
valid_2 = validation2.merge(pred_file,on='order_id',how='left')

f1_score(valid_2['products'],valid_2['products_o'])

dd = valid_2[valid_2['products']]




