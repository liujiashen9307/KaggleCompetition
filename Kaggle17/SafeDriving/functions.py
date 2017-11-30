# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:25:03 2017

@author: Jiashen Liu

"""

import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

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

def get_feature_importance_lgb(model):
    Importance = list(model.feature_importance())
    Feature= model.feature_name()
    df = pd.DataFrame({'Feature':Feature,'Score':Importance}).sort_values(by=['Score'],ascending=[0])
    return df  

def xgb_cross_val(params,rounds,train,col,label,num_split):
    i = 0
    kf = KFold(n_splits=num_split,shuffle=True)
    train = train.reset_index(drop=True)
    for train_index, test_index in kf.split(train):
        print("Training "+str(i+1)+' Fold')
        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train, y_test = label.iloc[train_index],label.iloc[test_index]
        dtrain = xgb.DMatrix(X_train[col],y_train)
        dtest = xgb.DMatrix(X_test[col])
        model = xgb.train(params,dtrain,num_boost_round=rounds)
        pred = model.predict(dtest)
        true = list(y_test)

def lgb_cross_val(params,rounds,train,col,label,num_split=5):
    i = 0
    kf = KFold(n_splits=num_split,shuffle=True)
    train = train.reset_index(drop=True)
    for train_index, test_index in kf.split(train):
        print("Training "+str(i+1)+' Fold')
        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train, y_test = label.iloc[train_index],label.iloc[test_index]
        train_lgb=lgb.Dataset(X_train[col],y_train)
        model = lgb.train(params,train_lgb,num_boost_round=rounds)
        pred = model.predict(X_test[col])
        true = list(y_test)

