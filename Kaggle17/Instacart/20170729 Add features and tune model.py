# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:15:58 2017

@author: Jiashen Liu

@Purpose: Make the code neat and faster, especially on us part.

现在主要的任务就是：搞更好的特征出来，同时去调模型。

主要用10%的数据去训练和CV XGB（90-100轮次，0.1 eta，缩短训练时间），百分之90去测试F-Score(用不同的threshold)，
如果有一个相对较大的提高，则提交一次，用百分百的数据。

调xgb，尝试lgb，加特征。主要就是这三个任务。
"""

from functions_in import *
import pandas as pd
import numpy as np

DIR = 'data/'
W2C = pd.read_csv('data/WORD2VEC_Feat.csv')
ostreak = pd.read_csv(DIR+'order_streaks.csv')
priors, train, orders, products, aisles, departments, sample_submission = load_data(DIR)


## detail表里记录这以前所有订单的信息
priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

#新特征
# _user_buy_product_times: 用户是第几次购买该商品
priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

"""
特征组1：产品组。主要看：在之前的订单记录中，产品的一些特性
_prod_tot_cnts： 产品被买的次数
_prod_reorder_tot_cnts：产品被回购的总次数
_prod_buy_first_time_total_cnt：产品被首次购买的次数
_prod_buy_second_time_total_cnt：产品被二次购买的次数
_prod_mean_cart_order：产品被放入购物篮顺序的均值
_prod_std_cart_order：产品被放入购物篮顺序的标准差
_prod_median_cart_order:产品被放入顺序的中位数
_prod_reorder_prob：不好理解，看特征重要性再说
_prod_reorder_ratio：回购率
_prod_reorder_times：产品被回购的次数？？不好理解，看重要性
_prod_dow_*,_prod_hod_*,'_prod_days_since'，：三个大类指标分别去衡量产品被订购的时间和日期，以及产品
被上次购买的信息
"""


agg_dict = {'user_id':{'_prod_tot_cnts':'count'}, 
            'reordered':{'_prod_reorder_tot_cnts':'sum'}, 
            '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)},
            'add_to_cart_order':{'_prod_mean_cart_order':'mean',
                                '_prod_std_cart_order':'std',
                                '_prod_median_cart_order':'median'},
             'order_dow':{'_prod_mean_dow':'mean',
                          '_prod_std_dow':'std',
                          '_prod_median_dow':'median'},
              'order_hour_of_day':{'_prod_mean_hod':'mean',
                                   '_prod_std_hod':'std',
                                    '_prod_median_hod':'median'},
               'days_since_prior_order':{
                                        '_prod_sum_days_since_prior_order':'sum', 
                                        '_prod_mean_days_since_prior_order': 'mean',
                                        '_prod_std_days_since_prior_order':'std',
                                        '_prod_median_days_since_prior_order':'median'
                                         }}
prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt


"""
特征组2： 用户组，统计一些用户的信息
_user_total_orders: 用户的总订单数
_user_sum_days_since_prior_order: 距离上次购买时间(和),这个只能在orders表里面计算，priors_orders_detail不是在order level上面unique
_user_mean_days_since_prior_order: 距离上次购买时间(均值)
_user_std_days_since_prior_order：距离上次买的时间的标准差
_user_median_days_since_prior_order:距离上次买的时间的中位数
_dow,_hod：购买时间特征
# _user_reorder_ratio: reorder的总次数 / 第一单后买后的总次数
# _user_total_products: 用户购买的总商品数
# _user_distinct_products: 用户购买的unique商品数
_user_average_basket: 购物蓝的大小

"""
agg_dict_2 = {'order_number':{'_user_total_orders':'max'},
              'days_since_prior_order':{'_user_sum_days_since_prior_order':'sum', 
                                        '_user_mean_days_since_prior_order': 'mean',
                                        '_user_std_days_since_prior_order':'std',
                                        '_user_median_days_since_prior_order':'median'},
               'order_dow':{'_user_mean_dow':'mean',
                          '_user_std_dow':'std',
                          '_user_median_dow':'median'},
              'order_hour_of_day':{'_user_mean_hod':'mean',
                                   '_user_std_hod':'std',
                                    '_user_median_hod':'median'}
               }
users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'], agg_dict_2)

#用户相关的特征重新写成以下格式，时间缩短为不到20秒

us = pd.concat([
    priors_orders_detail.groupby('user_id')['product_id'].count().rename('_user_total_products'),
    priors_orders_detail.groupby('user_id')['product_id'].nunique().rename('_user_distinct_products'),
    (priors_orders_detail.groupby('user_id')['reordered'].sum() /
        priors_orders_detail[priors_orders_detail['order_number'] > 1].groupby('user_id')['order_number'].count()).rename('_user_reorder_ratio')
], axis=1).reset_index()


users = users.merge(us, how='inner')
users['_user_average_basket'] = users._user_total_products / users._user_total_orders

us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)
users = users.merge(us, how='inner')

"""
特征组3：用户和产品交互特征
"""

agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                              '_up_first_order_number': 'min', 
                              '_up_last_order_number':'max'}, 
              'add_to_cart_order':{'_up_average_cart_position': 'mean',
                                   '_up_cart_position_std':'std',
                                   '_up_cart_position_median':'median'},
              'order_dow':{'_user_prd_order_mean_day':'mean',
                           '_user_prd_order_std_day':'std',
                           '_user_prd_order_median_day':'median'},
              'order_hour_of_day':{'_order_hod_mean':'mean',
                                   '_order_hod_std':'std',
                                   '_order_hod_median':'median'},
               'reordered':{'_total_time_of_reorder':'sum'}}

data = ka_add_groupby_features_1_vs_n(df=priors_orders_detail, 
                                                      group_columns_list=['user_id', 'product_id'], 
                                                      agg_dict=agg_dict_4)

data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')
# 该商品购买次数 / 总的订单数
# 最近一次购买商品 - 最后一次购买该商品
# 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
data['_up_order_rate'] = data._up_order_count / data._user_total_orders
data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders - data._up_first_order_number + 1)
data['_usr_prd_reorder_rate'] = data._total_time_of_reorder/data._up_order_count
data['_usr_prd_buy_rate'] = data._up_order_count/data._user_total_products
# add user_id to train set
train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')
data['reordered'] = data['reordered'].fillna(0)
data = data.merge(W2C,on='product_id',how='left')
data = data.merge(ostreak,on=['user_id','product_id'],how='left')
# release Memory
del train, prd, users,priors_orders_detail, orders
del products,us,aisles,departments,priors,ostreak,W2C
del agg_dict,agg_dict_2,agg_dict_4


train_set = data[data['eval_set']=='train']
del data

from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits=10) 
train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train_set, groups=train_set['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)
test_index = train_indexes[0]
train_index = test_indexes[0]
training = train_set.iloc[train_index,:]
testing = train_set.iloc[test_index,:]

##调整算法

import xgboost as xgb

col = list(train_set.columns)
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
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
}
xgb_cv = xgb.cv(xgb_params, dtrain, num_boost_round=90, early_stopping_rounds=20,
    verbose_eval=5, show_stdv=False)

##VALID一次
FI = get_feature_importance(model)
dtest = xgb.DMatrix(testing[col])
model = xgb.train(xgb_params,dtrain,num_boost_round=90)
pred = model.predict(dtest)
FScore = Cross_Validation(testing,pred,0.21)
"""
depth5: 0.37690822593628992
depth6: 0.37782129417038796
depth7: 0.37739409175293698, F-Score
"""

'''
['_up_order_count',
 '_up_first_order_number',
 '_up_last_order_number',
 '_up_average_cart_position',
 '_up_cart_position_std',
 '_up_cart_position_median',
 '_user_prd_order_mean_day',
 '_user_prd_order_std_day',
 '_user_prd_order_median_day',
 '_order_hod_mean',
 '_order_hod_std',
 '_order_hod_median',
 '_total_time_of_reorder',
 '_prod_tot_cnts',
 '_prod_reorder_tot_cnts',
 '_prod_buy_first_time_total_cnt',
 '_prod_buy_second_time_total_cnt',
 '_prod_mean_cart_order',
 '_prod_std_cart_order',
 '_prod_median_cart_order',
 '_prod_mean_dow',
 '_prod_std_dow',
 '_prod_median_dow',
 '_prod_mean_hod',
 '_prod_std_hod',
 '_prod_median_hod',
 '_prod_sum_days_since_prior_order',
 '_prod_mean_days_since_prior_order',
 '_prod_std_days_since_prior_order',
 '_prod_median_days_since_prior_order',
 '_prod_reorder_prob',
 '_prod_reorder_ratio',
 '_prod_reorder_times',
 '_user_total_orders',
 '_user_sum_days_since_prior_order',
 '_user_mean_days_since_prior_order',
 '_user_std_days_since_prior_order',
 '_user_median_days_since_prior_order',
 '_user_mean_dow',
 '_user_std_dow',
 '_user_median_dow',
 '_user_mean_hod',
 '_user_std_hod',
 '_user_median_hod',
 '_user_total_products',
 '_user_distinct_products',
 '_user_reorder_ratio',
 '_user_average_basket',
 'time_since_last_order',
 '_up_order_rate',
 '_up_order_since_last_order',
 '_up_order_rate_since_first_order',
 '_usr_prd_reorder_rate',
 '_usr_prd_buy_rate',
 'W2V_1',
 'W2V_2',
 'order_streak']

                               Feature  Score
11                  _user_reorder_ratio    624
7                 time_since_last_order    483
1            _up_order_since_last_order    470
4                    _prod_reorder_prob    435
33                 _user_average_basket    357
0                        _up_order_rate    334
6      _up_order_rate_since_first_order    327
41     _user_std_days_since_prior_order    297
21     _user_sum_days_since_prior_order    280
18                        _user_std_hod    278
3                   _prod_reorder_ratio    276
45                        _user_std_dow    254
22                       _user_mean_hod    244
24    _user_mean_days_since_prior_order    239
20    _prod_mean_days_since_prior_order    222
9                  _user_total_products    205
28                       _user_mean_dow    204
17                _up_last_order_number    195
42                        _prod_std_hod    190
38              _user_distinct_products    189
8                        _order_hod_std    181
10                _prod_mean_cart_order    181
26                        _prod_std_dow    181
5                 _up_cart_position_std    180
16                       _prod_mean_dow    177
12                                W2V_1    175
19       _prod_buy_first_time_total_cnt    173
23     _prod_std_days_since_prior_order    172
30            _up_average_cart_position    172
37                       _prod_mean_hod    166
35                 _prod_std_cart_order    165
2                       _up_order_count    161
47                    _usr_prd_buy_rate    153
52                                W2V_2    143
14              _user_prd_order_std_day    133
44                      _order_hod_mean    128
13               _prod_reorder_tot_cnts    117
43               _up_first_order_number    115
15      _prod_buy_second_time_total_cnt    107
25                       _prod_tot_cnts    103
31                         order_streak     98
27             _up_cart_position_median     95
50                     _user_median_hod     92
39  _user_median_days_since_prior_order     89
29             _user_prd_order_mean_day     88
36                   _user_total_orders     85
46     _prod_sum_days_since_prior_order     78
40                    _order_hod_median     53
54                     _user_median_dow     53
32                  _prod_reorder_times     34
34              _prod_median_cart_order     27
51  _prod_median_days_since_prior_order     27
48           _user_prd_order_median_day     22
49                     _prod_median_hod     10
53                     _prod_median_dow      5
55               _total_time_of_reorder      2
'''

"""
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 5
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
}

[0]     train-logloss:0.625488  test-logloss:0.625619
[5]     train-logloss:0.424971  test-logloss:0.425725
[10]    train-logloss:0.334086  test-logloss:0.335424
[15]    train-logloss:0.289109  test-logloss:0.290988
[20]    train-logloss:0.266201  test-logloss:0.268642
[25]    train-logloss:0.25433   test-logloss:0.257377
[30]    train-logloss:0.247994  test-logloss:0.251712
[35]    train-logloss:0.244482  test-logloss:0.248839
[40]    train-logloss:0.242249  test-logloss:0.247285
[45]    train-logloss:0.240709  test-logloss:0.246439
[50]    train-logloss:0.239566  test-logloss:0.245943
[55]    train-logloss:0.238609  test-logloss:0.245641
[60]    train-logloss:0.237764  test-logloss:0.245429
[65]    train-logloss:0.237028  test-logloss:0.245264
[70]    train-logloss:0.236282  test-logloss:0.245108
[75]    train-logloss:0.235651  test-logloss:0.24499
[80]    train-logloss:0.235046  test-logloss:0.244877
[85]    train-logloss:0.23446   test-logloss:0.244784


[0]     train-logloss:0.625922  test-logloss:0.625957
[5]     train-logloss:0.426835  test-logloss:0.427032
[10]    train-logloss:0.336836  test-logloss:0.337179
[15]    train-logloss:0.292365  test-logloss:0.292803
[20]    train-logloss:0.269896  test-logloss:0.270453
[25]    train-logloss:0.25846   test-logloss:0.25918
[30]    train-logloss:0.252517  test-logloss:0.253411
[35]    train-logloss:0.249325  test-logloss:0.250378
[40]    train-logloss:0.24754   test-logloss:0.248773
[45]    train-logloss:0.24645   test-logloss:0.247875
[50]    train-logloss:0.245703  test-logloss:0.247311
[55]    train-logloss:0.245117  test-logloss:0.246921
[60]    train-logloss:0.244665  test-logloss:0.246681
[65]    train-logloss:0.244225  test-logloss:0.246444
[70]    train-logloss:0.24386   test-logloss:0.246307
[75]    train-logloss:0.243522  test-logloss:0.246172
[80]    train-logloss:0.243236  test-logloss:0.246078
[85]    train-logloss:0.242962  test-logloss:0.245981

[0]     train-logloss:0.625922  test-logloss:0.625957
[5]     train-logloss:0.426835  test-logloss:0.427032
[10]    train-logloss:0.336836  test-logloss:0.337179
[15]    train-logloss:0.292365  test-logloss:0.292803
[20]    train-logloss:0.269896  test-logloss:0.270453
[25]    train-logloss:0.25846   test-logloss:0.25918
[30]    train-logloss:0.252517  test-logloss:0.253411
[35]    train-logloss:0.249325  test-logloss:0.250378
[40]    train-logloss:0.24754   test-logloss:0.248773
[45]    train-logloss:0.24645   test-logloss:0.247875
[50]    train-logloss:0.245703  test-logloss:0.247311
[55]    train-logloss:0.245117  test-logloss:0.246921
[60]    train-logloss:0.244665  test-logloss:0.246681
[65]    train-logloss:0.244225  test-logloss:0.246444
[70]    train-logloss:0.24386   test-logloss:0.246307
[75]    train-logloss:0.243522  test-logloss:0.246172
[80]    train-logloss:0.243236  test-logloss:0.246078
[85]    train-logloss:0.242962  test-logloss:0.245981

xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
}

[0]     train-logloss:0.625687  test-logloss:0.625749
[5]     train-logloss:0.425837  test-logloss:0.426201
[10]    train-logloss:0.335432  test-logloss:0.336079
[15]    train-logloss:0.290801  test-logloss:0.291716
[20]    train-logloss:0.268225  test-logloss:0.269415
[25]    train-logloss:0.25663   test-logloss:0.258128
[30]    train-logloss:0.250586  test-logloss:0.252436
[35]    train-logloss:0.247291  test-logloss:0.249471
[40]    train-logloss:0.245357  test-logloss:0.247895
[45]    train-logloss:0.244167  test-logloss:0.247085
[50]    train-logloss:0.243294  test-logloss:0.246604
[55]    train-logloss:0.242594  test-logloss:0.246292
[60]    train-logloss:0.241992  test-logloss:0.246061
[65]    train-logloss:0.241426  test-logloss:0.245843
[70]    train-logloss:0.240884  test-logloss:0.245698
[75]    train-logloss:0.240404  test-logloss:0.245561
[80]    train-logloss:0.240043  test-logloss:0.245474
[85]    train-logloss:0.239629  test-logloss:0.245381

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
[0]     train-logloss:0.625824  test-logloss:0.62589
[5]     train-logloss:0.426256  test-logloss:0.426599
[10]    train-logloss:0.335923  test-logloss:0.336525
[15]    train-logloss:0.291284  test-logloss:0.292142
[20]    train-logloss:0.268694  test-logloss:0.269794
[25]    train-logloss:0.257094  test-logloss:0.258469
[30]    train-logloss:0.251033  test-logloss:0.25271
[35]    train-logloss:0.247723  test-logloss:0.24969
[40]    train-logloss:0.24586   test-logloss:0.248102
[45]    train-logloss:0.24471   test-logloss:0.247219
[50]    train-logloss:0.243922  test-logloss:0.246695
[55]    train-logloss:0.243364  test-logloss:0.246394
[60]    train-logloss:0.242854  test-logloss:0.246178
[65]    train-logloss:0.242399  test-logloss:0.245967
[70]    train-logloss:0.242007  test-logloss:0.245832
[75]    train-logloss:0.24159   test-logloss:0.245682
[80]    train-logloss:0.241223  test-logloss:0.245547
[85]    train-logloss:0.240925  test-logloss:0.245461
"""