# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:03:22 2017

@author: Jiashen Liu
"""

import pandas as pd
import numpy as np
import functions
import matplotlib.pyplot as plt
import xgboost as xgb

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

"""
Idea: find out feats that do not have enough variance and create PCA feats from them.

"""

col = list(test.columns)
col.remove('id')
stds = []

for each in col:
    stds.append(np.std(train[each]))

std_df = pd.DataFrame({'var':col,'std':stds}) 

std_df = std_df.sort_values(by=['std'],ascending=[1])


"""
std             var
0    0.019309       ps_ind_01
1    0.030768   ps_ind_02_cat
2    0.041097       ps_ind_03
3    0.058327   ps_ind_04_cat
4    0.091619   ps_ind_05_cat
5    0.096693   ps_ind_06_bin
6    0.127545   ps_ind_07_bin
7    0.224588   ps_ind_08_bin
8    0.286893   ps_ind_09_bin
9    0.287153   ps_ind_10_bin
10   0.287198   ps_ind_11_bin
11   0.287642   ps_ind_12_bin
12   0.326222   ps_ind_13_bin
13   0.327778       ps_ind_14
14   0.347106       ps_ind_15
15   0.357154   ps_ind_16_bin
16   0.360294   ps_ind_17_bin
17   0.360417   ps_ind_18_bin
18   0.370204       ps_reg_01
19   0.373795       ps_reg_02
20   0.375716       ps_reg_03
21   0.388544   ps_car_01_cat
22   0.404264   ps_car_02_cat
23   0.436998   ps_car_03_cat
24   0.452447   ps_car_04_cat
25   0.473430   ps_car_05_cat
26   0.476661   ps_car_06_cat
27   0.483381   ps_car_07_cat
28   0.488579   ps_car_08_cat
29   0.493311   ps_car_09_cat
30   0.497056   ps_car_10_cat
31   0.664594   ps_car_11_cat
32   0.731366       ps_car_11
33   0.788654       ps_car_12
34   0.793505       ps_car_13
35   0.832547       ps_car_14
36   0.844416       ps_car_15
37   0.978746      ps_calc_01
38   1.117218      ps_calc_02
39   1.134926      ps_calc_03
40   1.202962      ps_calc_04
41   1.246948      ps_calc_05
42   1.334311      ps_calc_06
43   1.350641      ps_calc_07
44   1.414563      ps_calc_08
45   1.459671      ps_calc_09
46   1.694885      ps_calc_10
47   1.983787      ps_calc_11
48   2.153461      ps_calc_12
49   2.332869      ps_calc_13
50   2.508268      ps_calc_14
51   2.699900  ps_calc_15_bin
52   2.746649  ps_calc_16_bin
53   2.904595  ps_calc_17_bin
54   3.546039  ps_calc_18_bin
55   5.501440  ps_calc_19_bin
56  33.012428  ps_calc_20_bin
"""

## Do PCA and TruncatedSVD to capture more variance


var_dr = list(std_df['var'])[:37]

from sklearn.decomposition import PCA,TruncatedSVD
n_comp = 5

pca = PCA(n_components= n_comp,random_state=42)

pca.fit(train[var_dr])

pca_train = pca.fit_transform(train[var_dr])
pca_test = pca.transform(test[var_dr])

TSVD = TruncatedSVD(n_components=n_comp,random_state=42)
tsvd_train = TSVD.fit_transform(train[var_dr])
tsvd_test = TSVD.transform(test[var_dr])

for i in range(1,n_comp+1):
    train['pca_'+str(i)] = pca_train[:,i-1]
    test['pca_'+str(i)] = pca_test[:,i-1]
    train['tsvd_'+str(i)] = tsvd_train[:,i-1]
    test['tsvd_'+str(i)] = tsvd_test[:,i-1]
    
col = list(test.columns)
col.remove('id')


dtrain = xgb.DMatrix(train[col],train['target'])
dtest = xgb.DMatrix(test[col])

xgb_params = {'eta': 0.02, 
              'max_depth': 4, 
              'subsample': 0.9, 
              'colsample_bytree': 0.9, 
              'objective': 'binary:logistic', 
              'eval_metric': ['auc','logloss'], 
              'seed': 42, 
              'silent': True}

xgb_cvalid = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
print(len(xgb_cvalid))

"""

[0]     train-auc:0.596492      train-logloss:0.676111  test-auc:0.59193        test-logloss:0.676112
[50]    train-auc:0.622526      train-logloss:0.27558   test-auc:0.615973       test-logloss:0.275674
[100]   train-auc:0.626451      train-logloss:0.181736  test-auc:0.61883        test-logloss:0.18195
[150]   train-auc:0.634233      train-logloss:0.158598  test-auc:0.624544       test-logloss:0.159001
[200]   train-auc:0.642092      train-logloss:0.153264  test-auc:0.629285       test-logloss:0.153918
[250]   train-auc:0.648259      train-logloss:0.151811  test-auc:0.632465       test-logloss:0.152755
[300]   train-auc:0.653096      train-logloss:0.151164  test-auc:0.634465       test-logloss:0.152395
[350]   train-auc:0.657565      train-logloss:0.150708  test-auc:0.635761       test-logloss:0.152233
[400]   train-auc:0.661803      train-logloss:0.150332  test-auc:0.636787       test-logloss:0.152139
[450]   train-auc:0.665338      train-logloss:0.15001   test-auc:0.637311       test-logloss:0.152086
[500]   train-auc:0.668762      train-logloss:0.14972   test-auc:0.637633       test-logloss:0.152058
[550]   train-auc:0.671784      train-logloss:0.149451  test-auc:0.6378 test-logloss:0.15204
[600]   train-auc:0.674711      train-logloss:0.149197  test-auc:0.637912       test-logloss:0.152029
[650]   train-auc:0.677466      train-logloss:0.148957  test-auc:0.638109       test-logloss:0.152015

662
"""