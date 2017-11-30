# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:38:14 2017

@author: Jiashen Liu

Purpose: Split huge training set by user_id, and do cross validation
"""

import pandas as pd
import numpy as np

submission = pd.read_csv('data/sample_submission.csv')


"""
F1-Score Function, Copied from Kernel

"""

def f1_score_single(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)
    
def f1_score(y_true, y_pred):
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])



def sub_file(test_set,threshold,submission):
    d = dict()
    for row in test_set.itertuples():
        if row.reordered > threshold:
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
    return sub

data = pd.read_csv('data_bakup.csv')

test_set = data[data['eval_set']=='test']
train_set = data[data['eval_set']=='train']
del data
train_set['reordered']=train_set['reordered'].fillna(0)
test_set['reordered'] = test_set['reordered'].fillna(0)

col = list(test_set.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')

"""
CV Attempt 1: Just Split By User ID

"""

train = train_set.iloc[0:round(0.15*len(train_set)),:]
test = train_set.iloc[1271200:len(train_set),:]

import xgboost as xgb
dtrain = xgb.DMatrix(train[col],train['reordered'])
dtest = xgb.DMatrix(test[col])
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

"""

Through test set, we create two files. The first one is the ground truth, the second is for validation

"""

d = dict()
for row in test.itertuples():
    if row.reordered==1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)
    #else:
        #d[row.order_id] = 'None'
T_test = pd.DataFrame.from_dict(d, orient='index')
T_test.reset_index(inplace=True)
T_test.columns = ['order_id', 'products_o']

def string_to_list(string):
    return string.split(' ')

T_test['products_o'] = T_test['products_o'].apply(lambda x:string_to_list(x))



"""
Create Predicted File

"""

test['reordered'] = pred_test
"""
d = dict()
for row in test.itertuples():
    if row.reordered>=0.21:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

valid_file = pd.DataFrame.from_dict(d, orient='index')
valid_file.reset_index(inplace=True)
valid_file.columns = ['order_id', 'products_pred']

valid_file = T_test.merge(valid_file,on='order_id',how='left')

valid_file['products_pred'] = valid_file['products_pred'].fillna('None')
"""

valid_file = sub_file(test,0.21,T_test)

valid_file['products'] = valid_file['products'].apply(lambda x:string_to_list(x))

valid = T_test.merge(valid_file,on='order_id',how='left')

f1_score(valid['products'],valid['products_o'])
f1_score(valid['products'],valid['products_o'])

## 0.054464277867130713
## 0.38542282369305658

## Yeah, I got proper CV technique, already!!!


### Although result is shitty, we try to make a submission ###

dtest = xgb.DMatrix(test_set[col])
pred = bst.predict(dtest)

test_set['reordered']=pred

sub = sub_file(test_set,0.21,submission)

sub.to_csv('submission/sub_test_cross_validation_strategy.csv',index=False)

"""
CV Strategy Round 2: GroupKFold

"""

from sklearn.model_selection import GroupKFold

kf = GroupKFold(n_splits=10) 

train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train_set, groups=train_set['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)

train_index = test_indexes[0]
test_index = train_indexes[0]

train,test = train_set.iloc[train_index,:], train_set.iloc[test_index,:]

dtrain = xgb.DMatrix(train[col],train['reordered'])
dtest = xgb.DMatrix(test[col])
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

d = dict()
for row in test.itertuples():
    if row.reordered==1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)
    else:
        d[row.order_id] = str(row.product_id)
T_test = pd.DataFrame.from_dict(d, orient='index')
T_test.reset_index(inplace=True)
T_test.columns = ['order_id', 'products_orginal']

T_test['products_orginal'] = T_test['products_orginal'].apply(lambda x:string_to_list(x))

test['reordered'] = pred_test
valid_file = sub_file(test_set,0.21,T_test)
valid_file['products'] = valid_file['products'].apply(lambda x: string_to_list(x))

#cc = valid_file[valid_file['products_orginal'].notnull()]

valid_file = T_test.merge(valid_file,on='order_id',how='left')

f1_score(valid_file['products'],valid_file['products_orginal'])