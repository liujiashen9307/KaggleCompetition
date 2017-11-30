# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:20:52 2017

@author: LUI01
"""

### 0.283 on Public LB

from Frame_Model import *

import lightgbm as lgb

LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50  
OPTIMIZE = False
MAX_ROUND = 264


# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)


lgb_params = {'metric': 'auc', 'learning_rate' : LEARNING_RATE, 'max_depth':4, 'max_bin':10,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}
"""
dtrain_cv = lgb.Dataset(X,y)


lgbcv = lgb.cv(lgb_params,dtrain_cv,early_stopping_rounds=EARLY_STOPPING_ROUNDS,num_boost_round=1000,verbose_eval=50)

MAX_ROUND = len(lgbcv['auc-mean'])
"""

for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    
    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_valid[f],
                                                        tst_series=X_test[f],
                                                        target=y_train,
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )
    if i==0 and OPTIMIZE:
        dtrain_cv = lgb.Dataset(X,y)
        lgbcv = lgb.cv(lgb_params,dtrain_cv,early_stopping_rounds=EARLY_STOPPING_ROUNDS,num_boost_round=1000,verbose_eval=50)
        MAX_ROUND = len(lgbcv['auc-mean'])
        print(MAX_ROUND)
    # Run model for this fold
    dtrain = lgb.Dataset(X_train,y_train)     
    fit_model = lgb.train(lgb_params,dtrain,num_boost_round=MAX_ROUND)
    # Generate validation predictions for this fold
    pred = fit_model.predict(X_valid)
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    tmp = pd.DataFrame({'id':list(test_index),'target':pred})
    if i==0:
        stack = tmp
    else:
        stack = pd.concat([stack,tmp])
        
    # Accumulate test set predictions
    y_test_pred += fit_model.predict(X_test)
    
    #del X_test, X_train, X_valid, y_train
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)

stack.to_csv('stacking/stacking_lgb.csv',index=False)

sub = pd.DataFrame({'id':id_test,'target':y_test_pred})

sub.to_csv('submission/single_lgb.csv',index=False)




"""

Record 2 264

params:
    lgb_params = {'metric': 'auc', 'learning_rate' : LEARNING_RATE, 'max_depth':4, 'max_bin':10,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}
Fold  0
  Gini =  0.279453097993

Fold  1
  Gini =  0.278996617479

Fold  2
  Gini =  0.270319874449

Fold  3
  Gini =  0.298285535067

Fold  4
  Gini =  0.284980054139

Gini for full training set:
Out[28]: 0.28222025425119068


Record 1 141

params:
    
     {'metric': 'auc', 'learning_rate' : LEARNING_RATE, 'max_depth':10, 'max_bin':10,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10,  'min_data': 500}
    

MAX_ROUND: 

Fold  0
  Gini =  0.280627704052

Fold  1
  Gini =  0.276303021556

Fold  2
  Gini =  0.269763717232

Fold  3
  Gini =  0.29816914795

Fold  4
  Gini =  0.283157755414

0.281368485756612
"""