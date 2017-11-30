# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:10:13 2017

@author: Jiashen Liu
"""

from Frame_Model import *

from catboost import CatBoostClassifier

MAX_ROUNDS = 650
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.05
#EARLY_STOPPING_ROUNDS = 50  



# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

model = CatBoostClassifier(
    learning_rate=LEARNING_RATE, 
    depth=6, 
    l2_leaf_reg = 14, 
    iterations = MAX_ROUNDS,
#    verbose = True,
    loss_function='Logloss'
)



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
    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        eval_set=[(X_valid,y_valid)]
        fit_model = model.fit( X_train, y_train, 
                               eval_set=[X_valid, y_valid],
                               use_best_model=True

                             )
        print( "  Best N trees = ", model.tree_count_)
    else:
        fit_model = model.fit( X_train, y_train )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test)[:,1]
    tmp = pd.DataFrame({'id':list(test_index),'target':pred})
    if i==0:
        stack = tmp
    else:
        stack = pd.concat([stack,tmp])
    
    del X_test, X_train, X_valid, y_train
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)


stack.to_csv('stacking/stacking_cat.csv',index=False)



"""
Fold  0
  Gini =  0.284919067069

Fold  1
  Gini =  0.278911577733

Fold  2
  Gini =  0.274454693854

Fold  3
  Gini =  0.297086559907

Fold  4
  Gini =  0.282473629913

Gini for full training set:

eval_gini(y, y_valid_pred)
Out[2]: 0.28339271670432165
"""