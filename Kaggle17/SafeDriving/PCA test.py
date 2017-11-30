# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 21:53:49 2017

@author: LUI01
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
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
test = pd.read_csv('data/test.csv')

target = train['target']
train = train.drop(['id','target'],axis=1)
test = test.drop(['id'],axis=1)

pca = PCA(n_components=10,random_state=42)
train = pca.fit_transform(train)
test = pca.transform(test)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

lr = LogisticRegression(C=100)
kf = KFold(n_splits=5,random_state=42)

for i,(train_index,test_index) in enumerate(kf.split(train)):
    X_train,X_valid = train.iloc[train_index,:],train.iloc[test_index,:]
    y_train,y_valid = target.iloc[train_index],target.iloc[test_index]
    lr.fit(X_train,y_train)
    pred = lr.predict_proba(X_valid)[:,1]
    print(eval_gini(pred,y_valid))