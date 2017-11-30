# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:20:52 2017

@author: LUI01
"""

### 0.283 on Public LB

from Frame_Model import *

import lightgbm as lgb

def get_feature_importance_lgb(model):
    Importance = list(model.feature_importance())
    Feature= model.feature_name()
    df = pd.DataFrame({'Feature':Feature,'Score':Importance}).sort_values(by=['Score'],ascending=[0])
    return df  

col = ['nn_feat_8','nn_feat_7','nn_feat_11','nn_feat_2','nn_feat_0','nn_feat_3']

train_nn_feats = pd.read_csv('valid_nn_features.csv')[col]
test_nn_feats = pd.read_csv('test_nn_features.csv')[col]
X = pd.concat([X,train_nn_feats],axis=1)
test = pd.concat([test_df,test_nn_feats],axis=1)
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50  
OPTIMIZE = True
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

lgb_params = {'metric': 'auc', 'learning_rate' : LEARNING_RATE, 'max_depth':4, 'max_bin':10,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10}
168
  Gini =  0.27391399186

Fold  1
  Gini =  0.274512944033

Fold  2
  Gini =  0.26703089805

Fold  3
  Gini =  0.292679118625

Fold  4
  Gini =  0.27182359389

"""

