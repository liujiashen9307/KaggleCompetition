# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:55:14 2017

@author: LUI01
"""

from Frame_Model import *
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

target = pd.read_csv('data/train.csv')['target']

training['target']=target

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
clf = LogisticRegression()
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=100,
                   max_depth=3,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=1)

kf = KFold(n_splits=5,random_state=42)
y_valid_pred = 0*training['target']
for i, (train_index, test_index) in enumerate(kf.split(training)):
    
    X_train,X_test = training.iloc[train_index,:].copy(), training.iloc[test_index,:].copy()
    y_train,y_test = X_train['target'],X_test['target']
    del X_train['target']
    del X_test['target']
    
    clf.fit(X_train,y_train)
    pred = clf.predict_proba(X_test)[:,1]
    print( "  Gini = ", eval_gini(y_test, pred) )
    y_valid_pred.iloc[test_index] = pred

print(eval_gini(training['target'],y_valid_pred))

"""
 XGBClassifier(n_estimators=100,
                   max_depth=2,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8)
 
   Gini =  0.293626285517
  Gini =  0.292429688665
  Gini =  0.2934502109
  Gini =  0.299198722089
  Gini =  0.287922659639
0.292973857114

    Gini =  0.288891210562
  Gini =  0.289801110653
  Gini =  0.285466467401
  Gini =  0.29484963749
  Gini =  0.282653667596
0.28827630269

"""
col = list(training.columns)
col.remove('target')
clf.fit(training[col],training['target'])

final_pred = clf.predict_proba(testing)[:,1]

sub = pd.DataFrame()
sub['target'] = final_pred
sub['id'] = xgb_sub['id']

sub.to_csv('submission/stack_seven_models_xgb_meta.csv',index=False)


### LightGBM Stacking

import lightgbm as lgb

lgb_params = {'metric': 'auc', 'learning_rate' : 0.07, 'max_depth':2, 'max_bin':10,  
              'objective': 'binary', 
              'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}


y_valid_pred = 0*training['target']
for i, (train_index, test_index) in enumerate(kf.split(training)):
    
    X_train,X_test = training.iloc[train_index,:].copy(), training.iloc[test_index,:].copy()
    y_train,y_test = X_train['target'],X_test['target']
    del X_train['target']
    del X_test['target']
    dtrain = lgb.Dataset(X_train,y_train)
    model = lgb.train(lgb_params,dtrain,num_boost_round=100)
    pred = model.predict(X_test)
    print( "  Gini = ", eval_gini(y_test, pred) )
    y_valid_pred.iloc[test_index] = pred
    y_test_pred += model.predict(testing)
print(eval_gini(training['target'],y_valid_pred))

y_test_pred /= 5
"""
lgb_params = {'metric': 'auc', 'learning_rate' : 0.07, 'max_depth':1, 'max_bin':10,  
              'objective': 'binary', 
              'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}

Gini =  0.293784047809
  Gini =  0.292355493014
  Gini =  0.294574845968
  Gini =  0.298755014638
  Gini =  0.287811971663
0.2930840898

lgb_params = {'metric': 'auc', 'learning_rate' : 0.07, 'max_depth':2, 'max_bin':10,  
              'objective': 'binary', 
              'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}

  Gini =  0.293425610097
  Gini =  0.29320981326
  Gini =  0.293939357098
  Gini =  0.298297385577
  Gini =  0.287838539982
0.293054967077
"""
"""
lgb_params = {'metric': 'auc', 'learning_rate' : 0.07, 'max_depth':2, 'max_bin':10,  
              'objective': 'binary', 
              'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}

dtrain = lgb.Dataset(training[col],training['target'])

model = lgb.train(lgb_params,dtrain,num_boost_round=100)

pred = model.predict(testing)
"""
sub = pd.DataFrame()
sub['target'] = y_test_pred
sub['id'] = xgb_sub['id']

sub.to_csv('submission/stack_seven_models_lgb_meta_undersampled_attepmt.csv',index=False)



"""
Validating simple blending
"""


kf = KFold(n_splits=5,random_state=42)
y_valid_pred = 0*training['target']
for i, (train_index, test_index) in enumerate(kf.split(training)):
    
    X_train,X_test = training.iloc[train_index,:].copy(), training.iloc[test_index,:].copy()
    y_train,y_test = X_train['target'],X_test['target']
    del X_train['target']
    del X_test['target']
    
    pred = (X_test['xgb']+X_test['forza']+X_test['stacker']+X_test['rgf']+X_test['gp'])/5
    print( "  Gini = ", eval_gini(y_test, pred) )
    y_valid_pred.iloc[test_index] = pred

print(eval_gini(training['target'],y_valid_pred))


"""
  Gini =  0.289908657082
  Gini =  0.291023793396
  Gini =  0.287398620331
  Gini =  0.296055986043
  Gini =  0.283981246284
0.289651513285
"""

"""
Validating five stacking
"""

from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=100,
                   max_depth=3,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=1)

y_valid_pred = 0*training['target']
for i, (train_index, test_index) in enumerate(kf.split(training)):
    
    X_train,X_test = training.iloc[train_index,:].copy(), training.iloc[test_index,:].copy()
    y_train,y_test = X_train['target'],X_test['target']
    del X_train['target']
    del X_test['target']
    
    clf.fit(X_train,y_train)
    pred = clf.predict_proba(X_test)[:,1]
    print( "  Gini = ", eval_gini(y_test, pred) )
    y_valid_pred.iloc[test_index] = pred

print(eval_gini(training['target'],y_valid_pred))

"""
   Gini =  0.29375613994
  Gini =  0.292447633395
  Gini =  0.293852763468
  Gini =  0.298107524512
  Gini =  0.287653507182
0.292853987475
"""

"""
   Local CV: Stacking is better. 
"""



"""
   Averaging log form values
"""

kf = KFold(n_splits=5,random_state=42)
y_valid_pred = 0*training['target']
for i, (train_index, test_index) in enumerate(kf.split(training)):
    
    X_train,X_test = training.iloc[train_index,:].copy(), training.iloc[test_index,:].copy()
    y_train,y_test = X_train['target'],X_test['target']
    del X_train['target']
    del X_test['target']
    
    avg = (np.log(X_test['xgb'])+np.log(X_test['forza'])+np.log(X_test['stacker'])+np.log(X_test['rgf'])+np.log(X_test['gp']))/5
    pred = np.exp(avg)
    print( "  Gini = ", eval_gini(y_test, pred) )
    y_valid_pred.iloc[test_index] = pred

print(eval_gini(training['target'],y_valid_pred))

"""
  Gini =  0.291589648235
  Gini =  0.292182995337
  Gini =  0.29091466798
  Gini =  0.297868604805
  Gini =  0.285487604282
0.291596631232
"""

