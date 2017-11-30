# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:53:16 2017

@author: Jiashen Liu

@purpose: Re-run xgb model, for better understanding the situation and test functions
"""

from xgboost import XGBClassifier 
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

from functions import *
from preprocessing import *

train,test = feature_engineering()

col = list(test.columns)
col.remove('id')

"""
  Undersampling techniques: I don't quite get the point of this, but seems effective in this competition.
"""

kf = KFold(n_splits=5,random_state=42,shuffle=True)

learning_rate = 0.07
early_stopping_rounds = 50
optimization = True

f_cats = [f for f in train.columns if "_cat" in f]

y_valid_pred = 0*train['target']
y_test_pred = 0


model = XGBClassifier(    
                        n_estimators=1000,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=learning_rate, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                     )

Record = {}
for i, (train_index, test_index) in enumerate(kf.split(train)):
    
    # Create data for this fold
    X_train = train.iloc[train_index,:]
    X_valid = train.iloc[test_index,:]
    X_test = test
    print( "\nFold ", i)
    
    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_valid[f],
                                                        tst_series=X_test[f],
                                                        target=X_train['target'],
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )
    ## Update the list of features
    col = list(X_test.columns)
    col.remove('id')
    print('Number of features: '+str(len(col)))
    # Run model for this fold
    if optimization:
        eval_set=[(X_valid[col],list(X_valid['target']))]
        fit_model = model.fit( X_train[col], list(X_train['target']), 
                               eval_set=eval_set,
                               eval_metric=gini_xgb,
                               early_stopping_rounds=early_stopping_rounds,
                               verbose=False
                             )
        print( "  Best N trees = ", model.best_ntree_limit )
        print( "  Best gini = ", model.best_score )
    else:
        fit_model = model.fit( X_train[col], list(X_train['target']) )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid[col])[:,1]
    gini_cal = eval_gini(list(X_valid['target']), pred)
    print( "  Gini = ", gini_cal )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test[col])[:,1]
    Record.update({'Round'+str(i):model.best_ntree_limit,'Gini'+str(i):gini_cal})
    

    
y_test_pred /= 5  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(list(train['target']), y_valid_pred)

sub = pd.DataFrame()
sub['id'] = test['id']
sub['target'] = y_test_pred

sub.to_csv('submission/sub_xgb_284_other_seed.csv',index=False)




