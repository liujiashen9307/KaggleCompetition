# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:07:06 2017

@author: Jiashen Liu
"""

from Frame_Model import *
import numpy as np


y_test_pred = 0
training = pd.DataFrame()
testing = pd.DataFrame()

xgb_sub = pd.read_csv('stacking2/xgb_undersample.csv')
xgb_valid = pd.read_csv('stacking2/xgb_valid.csv')

training['xgb'] = np.log(xgb_valid['target'])
testing['xgb'] = np.log(xgb_sub['target'])

forza_sub= pd.read_csv('stacking2/Froza_and_Pascal.csv')
forza_valid = pd.read_csv('stacking2/forza_pascal_oof.csv')

training['forza'] = np.log(forza_valid['target'])
testing['forza'] = np.log(forza_sub['target'])

stacker_sub = pd.read_csv('stacking2/stacked_1.csv')
stacker_valid = pd.read_csv('stacking2/stacker_oof_preds_1.csv')

training['stacker'] = np.log(stacker_valid['target'])
testing['stacker'] = np.log(stacker_sub['target'])

rgf_sub = pd.read_csv('stacking2/rgf_submit.csv')
rgf_valid = pd.read_csv('stacking2/rgf_valid.csv')

training['rgf'] = np.log(rgf_valid['target'])
testing['rgf'] = np.log(rgf_sub['target'])

gp_sub = pd.read_csv('stacking2/gpari.csv')
gp_valid = pd.read_csv('stacking2/gp_pseu_val.csv')

training['gp'] = np.log(gp_valid['target'])
testing['gp'] = np.log(gp_sub['target'])

lgb_sub = pd.read_csv('stacking2/single_lgb.csv')
lgb_valid = pd.read_csv('stacking2/stacking_lgb.csv').sort_values(['id'],ascending=1)

training['lgb'] = np.log(lgb_valid['target'])
testing['lgb'] = np.log(lgb_sub['target'])

cat_sub = pd.read_csv('stacking2/single_catboost.csv')
cat_valid = pd.read_csv('stacking2/stacking_cat.csv').sort_values(['id'],ascending=1)

training['cat'] = np.log(cat_valid['target'])
testing['cat'] = np.log(cat_sub['target'])

target = pd.read_csv('data/train.csv')['target']

training['target']=target

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
Gini =  0.293998573624
  Gini =  0.292360094956
  Gini =  0.293208753906
  Gini =  0.298733024225
  Gini =  0.287534987321
0.292833305746
"""

col = list(training.columns)
col.remove('target')
clf.fit(training[col],training['target'])

final_pred = clf.predict_proba(testing)[:,1]

sub = pd.DataFrame()
sub['target'] = final_pred
sub['id'] = xgb_sub['id']

sub.to_csv('submission/stack_seven_models_log_oob_xgb_meta.csv',index=False)
