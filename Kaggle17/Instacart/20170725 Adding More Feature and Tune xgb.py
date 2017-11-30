# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:55:25 2017

@author: Jiashen Liu

@Purpose: today: focuse on more features, and xgb-parameter tunning.

Still: Use 10% of data
"""

from functions_in import *
import pandas as np
import numpy as np

data = pd.read_csv('data_bakup.csv')
data['reordered'] = data['reordered'].fillna(0)

train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
del data

###步骤1： 重拾xgb_cv,看原有的参数所有的cv结果

from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits=10) 
train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train, groups=train['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)
test_index = train_indexes[0]
train_index = test_indexes[0]
training = train.iloc[train_index,:]

import xgboost as xgb

col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')

dtrain = xgb.DMatrix(training[col],training['reordered'])

xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

xgb_cv = xgb.cv(xgb_params, dtrain, num_boost_round=300, early_stopping_rounds=20,
    verbose_eval=5, show_stdv=False)

"""
记录cv结果
[280]   train-logloss:0.234514  test-logloss:0.244016
[285]   train-logloss:0.234338  test-logloss:0.243999
[290]   train-logloss:0.234149  test-logloss:0.243967
[295]   train-logloss:0.233995  test-logloss:0.243948
尝试多轮数xgb并cv
"""
watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=300,verbose_eval=10, evals=watchlist)
##做CV##
testing = train.iloc[test_index,:]
dtest = xgb.DMatrix(testing[col])
pred = model.predict(dtest)
f1_score = Cross_Validation(testing,pred,0.20)
#0.37668066478839202
del training,testing,model
priors,train_o,orders,products,aisles,departments,sub = load_data(DIR)
"""
创造新特征，今日实验：加入department和aisle的特征
"""
products = products.merge(departments,on='department_id',how='left')
products = products.merge(aisles,on='aisle_id',how='left')
del products['department']
del products['aisle']

priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')
priors_orders_detail = priors_orders_detail.merge(products,on='product_id',how='left')
priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

"""
关于部门和aisle，创造：总订单量，回购订单数，回购率。
"""
agg_dict = {'user_id':{'dept_prod_tot_cnts':'count'}, 
            'reordered':{'dept_prod_reorder_tot_cnts':'sum'}}
dept = ka_add_groupby_features_1_vs_n(priors_orders_detail,['department_id'],agg_dict)
dept['dept_reorder_rate'] = dept['dept_prod_reorder_tot_cnts']/dept['dept_prod_tot_cnts']
products = products.merge(dept,on='department_id',how='left')

agg_dict2 = {'user_id':{'ais_prod_tot_cnts':'count'}, 
            'reordered':{'ais_prod_reorder_tot_cnts':'sum'}}
ais = ka_add_groupby_features_1_vs_n(priors_orders_detail,['aisle_id'],agg_dict2)
ais['ais_reorder_rate'] = ais['ais_prod_reorder_tot_cnts']/ais['ais_prod_tot_cnts']
products = products.merge(ais,on='aisle_id',how='left')
train = train.merge(products,on='product_id',how='left')
test = test.merge(products,on='product_id',how='left')
training = train.iloc[train_index,:]
testing = train.iloc[test_index,:]
##保留特征
col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')
col.remove('department_id')
col.remove('aisle_id')

dtrain = xgb.DMatrix(training[col],training['reordered'])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}
xgb_cv = xgb.cv(xgb_params, dtrain, num_boost_round=300, early_stopping_rounds=20,
    verbose_eval=5, show_stdv=False)

"""
[285]   train-logloss:0.233335  test-logloss:0.243794
[290]   train-logloss:0.233121  test-logloss:0.243773
[295]   train-logloss:0.232946  test-logloss:0.243754
"""
watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=300,verbose_eval=10, evals=watchlist)
dtest = xgb.DMatrix(testing[col])
pred = model.predict(dtest)
f1_score = Cross_Validation(testing,pred,0.20)
FI = get_feature_importance(model)
#0.37718366693543837

"""
Feature  Score
7                 _user_reorder_ratio   1243
14               _user_average_basket   1040
10  _user_mean_days_since_prior_order    976
5               time_since_last_order    771
16   _user_sum_days_since_prior_order    768
19            _user_distinct_products    712
3                 _prod_reorder_ratio    664
4                  _prod_reorder_prob    656
11               _user_total_products    646
1          _up_order_since_last_order    644
12          _up_average_cart_position    616
0                      _up_order_rate    604
6    _up_order_rate_since_first_order    534
18     _prod_buy_first_time_total_cnt    414
8               _up_last_order_number    411
9                    ais_reorder_rate    408
13                     _prod_tot_cnts    354
15             _prod_reorder_tot_cnts    322
23             _up_first_order_number    321
2                     _up_order_count    312
21                  ais_prod_tot_cnts    302
17                 _user_total_orders    234
20    _prod_buy_second_time_total_cnt    228
22                 dept_prod_tot_cnts    188
26          ais_prod_reorder_tot_cnts    183
25                  dept_reorder_rate    140
27         dept_prod_reorder_tot_cnts     68
24                _prod_reorder_times     40
"""

"""
加入Word2Vector特征
"""

W2C = pd.read_csv('data/WORD2VEC_Feat.csv')

train = train.merge(W2C,on='product_id',how='left')
test = test.merge(W2C,on='product_id',how='left')
training = train.iloc[train_index,:]
testing = train.iloc[test_index,:]
##保留特征
col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')
col.remove('department_id')
col.remove('aisle_id')

dtrain = xgb.DMatrix(training[col],training['reordered'])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}
xgb_cv = xgb.cv(xgb_params, dtrain, num_boost_round=300, early_stopping_rounds=20,
    verbose_eval=5, show_stdv=False)

"""
[280]   train-logloss:0.233391  test-logloss:0.244132
[285]   train-logloss:0.233191  test-logloss:0.244107
[290]   train-logloss:0.233033  test-logloss:0.244097
[295]   train-logloss:0.232813  test-logloss:0.244064
"""
watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=300,verbose_eval=10, evals=watchlist)
dtest = xgb.DMatrix(testing[col])
pred = model.predict(dtest)
f1_score = Cross_Validation(testing,pred,0.20)
##0.377429152132
FI = get_feature_importance(model)
"""
                  Feature  Score
7                 _user_reorder_ratio   1134
21               _user_average_basket    865
10  _user_mean_days_since_prior_order    856
19   _user_sum_days_since_prior_order    746
5               time_since_last_order    720
17            _user_distinct_products    703
1          _up_order_since_last_order    628
4                  _prod_reorder_prob    620
0                      _up_order_rate    609
16               _user_total_products    603
13                              W2V_1    557
3                 _prod_reorder_ratio    553
14          _up_average_cart_position    549
6    _up_order_rate_since_first_order    527
25                              W2V_2    479
8               _up_last_order_number    380
23     _prod_buy_first_time_total_cnt    374
11             _prod_reorder_tot_cnts    328
9                    ais_reorder_rate    325
12             _up_first_order_number    306
18                     _prod_tot_cnts    301
2                     _up_order_count    280
27                  ais_prod_tot_cnts    243
22    _prod_buy_second_time_total_cnt    218
20                 _user_total_orders    206
24          ais_prod_reorder_tot_cnts    188
26                 dept_prod_tot_cnts    169
29                  dept_reorder_rate    128
28         dept_prod_reorder_tot_cnts     51
15                _prod_reorder_times     36
"""

"""
今日不提交，得出结论：
1. 保留2个word2vec的特征
2. 尝试找几个强特征，而不是盲目的加特征。
   感觉用户级别的会好加一些。关于用户级别：已经有了用户的平均购物栏大小，为何不加入：购物篮大小的标准差？
   以及：用户最近购物的平均和加和时间已有，为何不加入：时间的方差？
   first order number与last order number的差值可以作为一个特征输入。
"""