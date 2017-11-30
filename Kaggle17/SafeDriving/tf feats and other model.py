# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:14:56 2017

@author: LUI01
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
import numpy as np


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


train = pd.read_csv('data/train.csv')

label = train['target']
feat = pd.read_csv('valid_nn_features.csv')

kf = KFold(n_splits=5,random_state=42)
clf = BernoulliNB()

for i,(train_index,test_index) in enumerate(kf.split(train)):
    train_feat,valid_feat = feat.iloc[train_index,:],feat.iloc[test_index,:]
    train_label,test_label = label.iloc[train_index],label.iloc[test_index]
    clf.fit(train_feat,train_label)
    prob = clf.predict_proba(valid_feat)[:,1]
    print(eval_gini(test_label,prob))