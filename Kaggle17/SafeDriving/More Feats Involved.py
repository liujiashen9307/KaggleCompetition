# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:51:37 2017

@author: Jiashen Liu
"""

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import time

"""
Use the most popular feature engineering techniques provided by popular kernel.
"""

def get_feature_importance_xgb(model):
    Importance = model.get_fscore()
    Importance = list(Importance.items())
    Feature= []
    Score = []
    for each in Importance:
        Feature.append(each[0])
        Score.append(each[1])
    df = pd.DataFrame({'Feature':Feature,'Score':Score}).sort_values(by=['Score'],ascending=[0])
    return df    

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)



trn_df = pd.read_csv("data/train.csv", index_col=0)
sub_df = pd.read_csv("data/test.csv", index_col=0)

target = trn_df["target"]
del trn_df["target"]

train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
	"ps_car_11_cat" # Very nice spot from Tilii : https://www.kaggle.com/tilii7
]
# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),  
    ('ps_reg_01', 'ps_car_04_cat'),
]
start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    trn_df[name1] = trn_df[f1].apply(lambda x: str(x)) + "_" + trn_df[f2].apply(lambda x: str(x))
    sub_df[name1] = sub_df[f1].apply(lambda x: str(x)) + "_" + sub_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(trn_df[name1].values) + list(sub_df[name1].values))
    trn_df[name1] = lbl.transform(list(trn_df[name1].values))
    sub_df[name1] = lbl.transform(list(sub_df[name1].values))

    train_features.append(name1)
    
trn_df = trn_df[train_features]
sub_df = sub_df[train_features]

f_cats = [f for f in trn_df.columns if "_cat" in f]

for f in f_cats:
    trn_df[f + "_avg"], sub_df[f + "_avg"] = target_encode(trn_series=trn_df[f],
                                         tst_series=sub_df[f],
                                         target=target,
                                         min_samples_leaf=200,
                                         smoothing=10,
                                         noise_level=0)
    
dtrain = xgb.DMatrix(trn_df,target)
dtest = xgb.DMatrix(sub_df)

params = {'eta': 0.02, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'silent': True}

xgb_cvalid = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

"""
[0]     train-auc:0.601491      test-auc:0.598552
[50]    train-auc:0.623257      test-auc:0.616687
[100]   train-auc:0.628371      test-auc:0.620698
[150]   train-auc:0.635952      test-auc:0.626529
[200]   train-auc:0.643582      test-auc:0.631684
[250]   train-auc:0.648931      test-auc:0.635115
[300]   train-auc:0.65329       test-auc:0.637317
[350]   train-auc:0.656986      test-auc:0.638768
[400]   train-auc:0.660317      test-auc:0.639833
[450]   train-auc:0.66322       test-auc:0.640439
[500]   train-auc:0.665844      test-auc:0.640949
[550]   train-auc:0.668338      test-auc:0.641333
[600]   train-auc:0.670512      test-auc:0.641531
[650]   train-auc:0.672563      test-auc:0.641652
[700]   train-auc:0.674546      test-auc:0.641821
[750]   train-auc:0.676525      test-auc:0.641943

"""

## 752 Rounds

model_xbg = xgb.train(params,dtrain,num_boost_round=800)

get_feature_importance_xgb(model_xbg)

"""
Feature  Score
0                          ps_car_13   1206
3                          ps_reg_03    924
19                         ps_ind_03    844
6                          ps_ind_15    621
7                          ps_ind_01    500
1                  ps_ind_05_cat_avg    450
32                         ps_car_14    448
10                         ps_reg_02    436
15                 ps_car_11_cat_avg    435
2                      ps_ind_17_bin    427
9   ps_reg_01_plus_ps_car_04_cat_avg    424
21                 ps_car_01_cat_avg    347
35                         ps_car_15    319
30  ps_reg_01_plus_ps_car_02_cat_avg    292
42                     ps_car_11_cat    266
25                 ps_car_09_cat_avg    241
8                      ps_car_07_cat    231
"""

pred = model_xbg.predict(dtest)

sub = pd.DataFrame({'id':pd.read_csv('data/test.csv')['id'],'target':pred})

sub.to_csv('submission/xgb_6419.csv',index=False)


"""
  Try other parameter set
"""

params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'silent': True,
          'gamma':1,
          "reg_alpha":0,
          "reg_lambda":1
          }
model_xbg2 = xgb.train(params,dtrain,num_boost_round=200)

pred = model_xbg2.predict(dtest)

test_id = pd.read_csv('data/test.csv')['id']

sub2 = pd.DataFrame({'id':test_id,'target':pred})

sub2.to_csv('submission/xgb_tuned_eta_0.1.csv',index=False)

"""
Andy's Parameter, try it!

"""

params = {'eta': 0.07, 
          'max_depth': 4, 
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
          'min_child_weight':6,
          'scale_pos_weight':1.6,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'silent': True,
          'gamma':10,
          "reg_alpha":8,
          "reg_lambda":1.3
          }

model_xbg3 = xgb.train(params,dtrain,num_boost_round=400)

pred = model_xbg2.predict(dtest)

sub3 = pd.DataFrame({'id':test_id,'target':pred})

sub3.to_csv('submission/xgb_tuned_eta_0.07_borrow_params.csv',index=False)
