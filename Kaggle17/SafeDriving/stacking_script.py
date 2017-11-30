# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:46:33 2017

@author: LUI01
"""

import os
import pandas as pd
from Frame_Model import *
dirs = 'stacking/'

all_files = os.listdir(dirs)

for each in all_files:
    df = pd.read_csv(dirs+each)
    name = each[9:12]
    df.columns = ['id',name]
    df = df.sort_values(['id'],ascending=1)
    if each == all_files[0]:
        final = df
    else:
        final[name] = df[name]
final = final.reset_index(drop=True)

train = pd.read_csv('data/train.csv')

final['target'] = train['target']

del final['id']


from xgboost import XGBClassifier

from sklearn.model_selection import KFold
LR = XGBClassifier(n_estimators=400,
                   max_depth=4,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8)
kf = KFold(n_splits=5,random_state=42)
avg = 0
for i, (train_index, test_index) in enumerate(kf.split(final)):
    X_train, X_valid = final.iloc[train_index,:].copy(), final.iloc[test_index,:].copy()
    y_train = X_train['target']
    y_valid = X_valid['target']
    del X_train['target']
    del X_valid['target']
    LR.fit(X_train,y_train)
    pred = LR.predict_proba(X_valid)[:,1]
    gini=eval_gini(y_valid, pred)
    avg+=gini
    print( "  Gini = ", gini)

print(avg/5)

"""
dafault

  Gini =  0.285688272845
  Gini =  0.288037636784
  Gini =  0.282640543648
  Gini =  0.291321936006
  Gini =  0.278089689458
0.285155615748
"""


