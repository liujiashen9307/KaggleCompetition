# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:37:23 2017

This file is used to create some submissions with sklearn and tensorflow

Feature Engineering: Currently without new features coming in.

"""

import pandas as pd


### Create Function for Submission File

submission = pd.read_csv('data/sample_submission.csv')

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
Model 1: Logistic Regression from sklearn

"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
reordered = train_set[train_set.reordered==1]
unordered = train_set[train_set.reordered==0]

unorder_train,unorder_test = train_test_split(unordered,test_size=0.9,random_state=42)

del unorder_test

unordered = unorder_train
train = pd.concat([reordered,unordered])

del reordered
del unorder_train,train_set


train = train.sample(frac=1)

training,testing = train_test_split(train,test_size=0.3,random_state = 42)

### Try Logistic Regression

lg = LogisticRegression(C=0.1,random_state=42)
lg.fit(training[col],training['reordered'])
pred1 = lg.predict(testing[col])
accu1 = accuracy_score(pred1,testing['reordered'])
ll1 = log_loss(pred1,testing['reordered'])
print(accu1)
print(ll1)

"""
Logistic Regression Performance:
    
Default: Accu: 0.732063101566 
         Log-loss: 9.25429273608

C = 10: 0.728207638545
        9.38744765882

C = 0.1: 0.733943764212
         9.18933507269

"""


### Try xgboost on subsampled data set
import xgboost as xgb
dtrain = xgb.DMatrix(training[col], list(training['reordered']))
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
model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=120, evals=watchlist, verbose_eval=10)
pred2 = model.predict(dtest)
pred2 = [1 if each>=0.5 else 0 for each in pred2]
accu2 = accuracy_score(pred2,testing['reordered'])
ll2 = log_loss(pred2,testing['reordered'])
print(accu2)
print(ll2)

### Make a submission with current xgb
dtrain = xgb.DMatrix(train[col],train['reordered'])
model_xgb = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=150, evals=watchlist, verbose_eval=10)
dtest = xgb.DMatrix(test_set[col])
pred_xgb = model_xgb.predict(dtest)
test_set['reordered'] = pred_xgb
test_set['product_id'] = test_set.product_id.astype(str)

sub = sub_file(test_set,0.5,submission)
sub.to_csv('submission/xgb_undersampled_threshold0.5.csv',index=False)


"""
Param Num 1:
    
[0]     train-logloss:0.664431
[10]    train-logloss:0.539278
[20]    train-logloss:0.512522
[30]    train-logloss:0.505347
[40]    train-logloss:0.502768
[50]    train-logloss:0.501322
[60]    train-logloss:0.500431
[70]    train-logloss:0.499852
[80]    train-logloss:0.499305

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
0.753932342168
8.49897190784

Param Num 2:

 xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :1
    ,"alpha"            :2e-05
    ,"lambda"           :10
}


[0]     train-logloss:0.663528
[10]    train-logloss:0.538879
[20]    train-logloss:0.512416
[30]    train-logloss:0.505271
[40]    train-logloss:0.502647
[50]    train-logloss:0.501237
[60]    train-logloss:0.500413
[70]    train-logloss:0.499869
[80]    train-logloss:0.499316
0.75359553829
8.51060502951

Para Num 3:
    
Same Para as first, Rounds 120.
    
[0]     train-logloss:0.664431
[10]    train-logloss:0.539278
[20]    train-logloss:0.512522
[30]    train-logloss:0.505347
[40]    train-logloss:0.502768
[50]    train-logloss:0.501322
[60]    train-logloss:0.500431
[70]    train-logloss:0.499852
[80]    train-logloss:0.499305
[90]    train-logloss:0.498866
[100]   train-logloss:0.49846
[110]   train-logloss:0.498086
0.754041123544
8.49521462849

"""

