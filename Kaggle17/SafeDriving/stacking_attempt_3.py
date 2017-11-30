# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:00:28 2017

@author: LUI01
"""

#from Frame_Model import *

import pandas as pd
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

y_test_pred = 0
training = pd.DataFrame()
testing = pd.DataFrame()

xgb_sub = pd.read_csv('stacking2/xgb_undersample.csv')
xgb_valid = pd.read_csv('stacking2/xgb_valid.csv')

training['xgb'] = xgb_valid['target']
testing['xgb'] = xgb_sub['target']

forza_sub= pd.read_csv('stacking2/Froza_and_Pascal.csv')
forza_valid = pd.read_csv('stacking2/forza_pascal_oof.csv')

training['forza'] = forza_valid['target']
testing['forza'] = forza_sub['target']

stacker_sub = pd.read_csv('stacking2/stacked_1.csv')
stacker_valid = pd.read_csv('stacking2/stacker_oof_preds_1.csv')

training['stacker'] = stacker_valid['target']
testing['stacker'] = stacker_sub['target']

rgf_sub = pd.read_csv('stacking2/rgf_submit.csv')
rgf_valid = pd.read_csv('stacking2/rgf_valid.csv')

training['rgf'] = rgf_valid['target']
testing['rgf'] = rgf_sub['target']

gp_sub = pd.read_csv('stacking2/gpari.csv')
gp_valid = pd.read_csv('stacking2/gp_pseu_val.csv')


training['gp'] = gp_valid['target']
testing['gp'] = gp_sub['target']

lgb_sub = pd.read_csv('stacking2/single_lgb.csv')
lgb_valid = pd.read_csv('stacking2/stacking_lgb.csv').sort_values(['id'],ascending=1)

training['lgb'] = lgb_valid['target']
testing['lgb'] = lgb_sub['target']


cat_sub = pd.read_csv('stacking2/single_catboost.csv')
cat_valid = pd.read_csv('stacking2/stacking_cat.csv').sort_values(['id'],ascending=1)

training['cat'] = cat_valid['target']
testing['cat'] = cat_sub['target']

nnet_sub = pd.read_csv('submission/NN_EntityEmbed_10fold-sub.csv')
nn_valid = pd.read_csv('stacking2/NN_EntityEmbed_10fold-val_preds.csv')

training['nn'] = nn_valid['0']
testing['nn'] = nnet_sub['target']

target = pd.read_csv('data/train.csv')['target']

training['target']=target


"""
  Level 2: stacker 1: xgb
"""
col = list(testing.columns)
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True,random_state=42)

y_valid_pred = 0*training['target']

xgb = XGBClassifier(n_estimators=150,
                   max_depth=2,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=1)


y_pred = 0
for i,(train_index,test_index) in enumerate(kf.split(training)):
    train_set,valid_set = training.iloc[train_index,:],training.iloc[test_index,:]
    xgb.fit(train_set[col],train_set['target'])
    prob = xgb.predict_proba(valid_set[col])[:,1]
    test_prob = xgb.predict_proba(testing)[:,1]
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob
print('Overall Gini: ',eval_gini(training['target'],y_valid_pred))

xgb_stack = y_valid_pred
xgb_stack_sub = y_pred/5
"""
Fold:  0  Gini:  0.297436336027
Fold:  1  Gini:  0.28625311775
Fold:  2  Gini:  0.307750510841
Fold:  3  Gini:  0.287137722177
Fold:  4  Gini:  0.284454248543
Overall Gini:  0.292434272624
"""

"""
   Level 2 stacker 2: LGB
"""
import lightgbm as lgb

lgb_params = {'metric': 'auc', 'learning_rate' : 0.07, 'max_depth':2, 'max_bin':10,  
              'objective': 'binary', 
              'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}

y_valid_pred = 0*training['target']
y_pred = 0

for i,(train_index,test_index) in enumerate(kf.split(training)):
    train_set,valid_set = training.iloc[train_index,:],training.iloc[test_index,:]
    dtrain = lgb.Dataset(train_set[col],train_set['target'])
    model = lgb.train(lgb_params,dtrain,num_boost_round=150)
    prob = model.predict(valid_set[col])
    test_prob = model.predict(testing)
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob

print('Overall Gini: ',eval_gini(training['target'],y_valid_pred))

"""
Fold:  0  Gini:  0.297179358422
Fold:  1  Gini:  0.286681875931
Fold:  2  Gini:  0.308426411167
Fold:  3  Gini:  0.287625583037
Fold:  4  Gini:  0.284516444521
Overall Gini:  0.292740506089
"""


lgb_stack = y_valid_pred
lgb_stack_sub = y_pred/5

"""
   Level 2 stacker 3: RandomForest
"""

from sklearn.ensemble import RandomForestClassifier


y_valid_pred = 0*training['target']

rf = RandomForestClassifier(n_estimators=200,
                   n_jobs=-1, min_samples_split=5, max_depth=4,
                          criterion='gini', random_state=0)


y_pred = 0
for i,(train_index,test_index) in enumerate(kf.split(training)):
    train_set,valid_set = training.iloc[train_index,:],training.iloc[test_index,:]
    rf.fit(train_set[col],train_set['target'])
    prob = rf.predict_proba(valid_set[col])[:,1]
    test_prob = rf.predict_proba(testing)[:,1]
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob
print('Overall Gini: ',eval_gini(training['target'],y_valid_pred))

"""
Fold:  0  Gini:  0.296573230831
Fold:  1  Gini:  0.285492755334
Fold:  2  Gini:  0.307766057327
Fold:  3  Gini:  0.287362463828
Fold:  4  Gini:  0.285004129726
Overall Gini:  0.292260464275
"""

rf_stack = y_valid_pred
rf_stack_sub = y_pred/5

"""
   level 2 Stacker 4: LogisticRegression
"""

from sklearn.linear_model import LogisticRegression

y_valid_pred = 0*training['target']

lr = LogisticRegression()


y_pred = 0
for i,(train_index,test_index) in enumerate(kf.split(training)):
    train_set,valid_set = training.iloc[train_index,:],training.iloc[test_index,:]
    lr.fit(train_set[col],train_set['target'])
    prob = lr.predict_proba(valid_set[col])[:,1]
    test_prob = lr.predict_proba(testing)[:,1]
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob
print('Overall Gini: ',eval_gini(training['target'],y_valid_pred))

"""
Fold:  0  Gini:  0.293722967672
Fold:  1  Gini:  0.288301218612
Fold:  2  Gini:  0.30332945363
Fold:  3  Gini:  0.278579398966
Fold:  4  Gini:  0.279156053941
Overall Gini:  0.288652056731
"""

lr_stack = y_valid_pred
lr_stack_sub = y_pred/5

"""
   Level 2 Stacker 5: ExtraTree
"""

from sklearn.ensemble import ExtraTreesClassifier

y_valid_pred = 0*training['target']

et = ExtraTreesClassifier(n_estimators=200,
                   n_jobs=-1, min_samples_split=5, max_depth=4,
                          criterion='gini', random_state=0)


y_pred = 0
for i,(train_index,test_index) in enumerate(kf.split(training)):
    train_set,valid_set = training.iloc[train_index,:],training.iloc[test_index,:]
    et.fit(train_set[col],train_set['target'])
    prob = et.predict_proba(valid_set[col])[:,1]
    test_prob = et.predict_proba(testing)[:,1]
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob
print('Overall Gini: ',eval_gini(training['target'],y_valid_pred))

"""
Fold:  0  Gini:  0.296224152446
Fold:  1  Gini:  0.287601735197
Fold:  2  Gini:  0.307834267488
Fold:  3  Gini:  0.285389263354
Fold:  4  Gini:  0.283859419583
Overall Gini:  0.292099335985
"""

et_stack = y_valid_pred
et_stack_sub = y_pred/5


"""
   Put everything together
"""

train = pd.DataFrame({'xgb':xgb_stack,'lgb':lgb_stack,'rf':rf_stack,'lr':lr_stack,'et':et_stack,'target':training['target']})
test = pd.DataFrame({'xgb':xgb_stack_sub,'lgb':lgb_stack_sub,'rf':rf_stack_sub,'lr':lr_stack_sub,'et':et_stack_sub})
col = list(test.columns)


train.to_csv('level2_stack_train.csv',index=False)
test.to_csv('level2_stack_test.csv',index=False)
"""
Level 3: stacker 1: xgb
"""

y_valid_pred = 0*training['target']

xgb = XGBClassifier(n_estimators=150,
                   max_depth=2,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=1)


y_pred = 0
for i,(train_index,test_index) in enumerate(kf.split(train)):
    train_set,valid_set = train.iloc[train_index,:],train.iloc[test_index,:]
    xgb.fit(train_set[col],train_set['target'])
    prob = xgb.predict_proba(valid_set[col])[:,1]
    test_prob = xgb.predict_proba(test)[:,1]
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob
print('Overall Gini: ',eval_gini(train['target'],y_valid_pred))

l2_xgb = y_valid_pred
l2_xgb_sub = y_pred/5

"""
Fold:  0  Gini:  0.296997654996
Fold:  1  Gini:  0.286436863767
Fold:  2  Gini:  0.30765358434
Fold:  3  Gini:  0.285756360744
Fold:  4  Gini:  0.282745696501
Overall Gini:  0.291270395194
"""


"""
Level3: Stacker 2: lgb
"""

import lightgbm as lgb

lgb_params = {'metric': 'auc', 'learning_rate' : 0.07, 'max_depth':2, 'max_bin':10,  
              'objective': 'binary', 
              'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}

y_valid_pred = 0*training['target']
y_pred = 0

for i,(train_index,test_index) in enumerate(kf.split(train)):
    train_set,valid_set = train.iloc[train_index,:],train.iloc[test_index,:]
    dtrain = lgb.Dataset(train_set[col],train_set['target'])
    model = lgb.train(lgb_params,dtrain,num_boost_round=150)
    prob = model.predict(valid_set[col])
    test_prob = model.predict(test)
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob

print('Overall Gini: ',eval_gini(training['target'],y_valid_pred))

"""
Fold:  0  Gini:  0.297538325916
Fold:  1  Gini:  0.287003228813
Fold:  2  Gini:  0.308173441381
Fold:  3  Gini:  0.284629686744
Fold:  4  Gini:  0.283846398724
Overall Gini:  0.291467609293
"""
l2_lgb = y_valid_pred
l2_lgb_sub = y_pred/5

"""
Level3: Stacker 3: lr
"""

y_valid_pred = 0*training['target']

lr = LogisticRegression()


y_pred = 0
for i,(train_index,test_index) in enumerate(kf.split(train)):
    train_set,valid_set = train.iloc[train_index,:],train.iloc[test_index,:]
    lr.fit(train_set[col],train_set['target'])
    prob = lr.predict_proba(valid_set[col])[:,1]
    test_prob = lr.predict_proba(test)[:,1]
    print('Fold: ',i,' Gini: ', eval_gini(valid_set['target'],prob))
    y_valid_pred.iloc[test_index] = prob   
    y_pred+=test_prob
print('Overall Gini: ',eval_gini(train['target'],y_valid_pred))

"""
Fold:  0  Gini:  0.297710876148
Fold:  1  Gini:  0.287559806278
Fold:  2  Gini:  0.308566077718
Fold:  3  Gini:  0.287226863396
Fold:  4  Gini:  0.284922773562
Overall Gini:  0.292943069693
"""

l2_lr = y_valid_pred
l2_lr_sub = y_pred/5

## 20171127 sub1: LR level3
tt = pd.read_csv('data/test.csv')
sub = pd.DataFrame()
sub['id'] = tt['id']
sub['target'] = l2_lr_sub
sub.to_csv('submission/level3_stacking_LR_only.csv',index=False)

## 20171127 sub2: log_average three level 3

average = 0.4*np.log(l2_lr_sub)+0.3*np.log(l2_xgb_sub)+0.3*np.log(l2_lgb_sub)
target = np.exp(average)
sub = pd.DataFrame()
sub['id'] = tt['id']
sub['target'] = target
sub.to_csv('submission/level3_stacking_LR_XGB_LGB_Log_Average.csv',index=False)

## 20171127 sub3: level 2 log_average

average = 0.25*np.log(test['xgb']) + 0.25*np.log(test['lgb']) + 0.15*np.log(test['rf']) + 0.2*np.log(test['et']) + 0.15*np.log(test['lr'])
target = np.exp(average)
sub = pd.DataFrame()
sub['id'] = tt['id']
sub['target'] = target
sub.to_csv('submission/Level2_stacking_log_average.csv',index=False)

## 20171127 Final Sub: Simple average log

df1 = pd.read_csv('submission/current_best_add_NN.csv')
df2 = pd.read_csv('submission/NN_EntityEmbed_10fold-sub.csv')

average = np.exp(0.9*np.log(df1['target']) + 0.1*np.log(df2['target']))

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/add_entity_embedding_NN_log_average.csv',index=False)


