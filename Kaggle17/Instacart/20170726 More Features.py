# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:21:34 2017

@author: Jiashen Liu

@Purpose: Putting more features in data set for train.
          Use same train/test split, of course
"""

from functions_in import *
import pandas as pd
import numpy as np

DIR = 'data/'

priors, train, orders, products, aisles, departments, sample_submission = load_data(DIR)

priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

# create new variables
# _user_buy_product_times: 用户是第几次购买该商品
priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

#Jiashen: 加入：产品被放进购物篮的顺序的均值和方差
agg_dict = {'user_id':{'_prod_tot_cnts':'count'}, 
            'reordered':{'_prod_reorder_tot_cnts':'sum'}, 
            '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)},
            'add_to_cart_order':{'_prod_mean_cart_order':'mean',
                                '_prod_std_cart_order':'std'}}
prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

# _prod_reorder_prob: 这个指标不好理解
# _prod_reorder_ratio: 商品复购率
prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt


# _user_total_orders: 用户的总订单数
# 可以考虑加入其它统计指标++++++++++++++++++++++++++
# _user_sum_days_since_prior_order: 距离上次购买时间(和),这个只能在orders表里面计算，priors_orders_detail不是在order level上面unique
# _user_mean_days_since_prior_order: 距离上次购买时间(均值)
# Jiashen:尝试在这里加入：since_last_order的标准差。 _user_std_days_since_prior_order
agg_dict_2 = {'order_number':{'_user_total_orders':'max'},
              'days_since_prior_order':{'_user_sum_days_since_prior_order':'sum', 
                                        '_user_mean_days_since_prior_order': 'mean',
                                        '_user_std_days_since_prior_order':'std'}}
users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'], agg_dict_2)

# _user_reorder_ratio: reorder的总次数 / 第一单后买后的总次数
# _user_total_products: 用户购买的总商品数
# _user_distinct_products: 用户购买的unique商品数
agg_dict_3 = {'reordered':
              {'_user_reorder_ratio': 
               lambda x: sum(priors_orders_detail.ix[x.index,'reordered']==1)/
                         sum(priors_orders_detail.ix[x.index,'order_number'] > 1)},
              'product_id':{'_user_total_products':'count', 
                            '_user_distinct_products': lambda x: x.nunique()}}
us = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['user_id'], agg_dict_3)

#us = pd.read_csv('user_transform.csv')
#us.to_csv('user_transform.csv',index=False)
users = users.merge(us, how='inner')

# 平均每单的商品数
# 每单中最多的商品数，最少的商品数++++++++++++++
users['_user_average_basket'] = users._user_total_products / users._user_total_orders

us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

users = users.merge(us, how='inner')

## Database Part

agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                              '_up_first_order_number': 'min', 
                              '_up_last_order_number':'max'}, 
              'add_to_cart_order':{'_up_average_cart_position': 'mean',
                                   '_up_cart_position_std':'std'},
              'order_dow':{'_order_mean_day':'mean',
                           '_order_std_day':'std'},
              'order_hour_of_day':{'_order_hod_mean':'mean',
                                   '_order_hod_std':'std'},
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
# release Memory
del train, prd, users
# gc.collect()
# release Memory
del priors_orders_detail, orders

#data.to_csv('data_bakup.csv',index=False)


## Try ML Algorithm

train_set = data[data['eval_set']=='train']
test_set = data[data['eval_set']=='test'] 
W2C = pd.read_csv('data/WORD2VEC_Feat.csv')
train_set = train_set.merge(W2C,on='product_id',how='left')
test_set = test_set.merge(W2C,on='product_id',how='left')

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
先暂时不去做这个CV了
[285]   train-logloss:0.232307  test-logloss:0.244314
[290]   train-logloss:0.232104  test-logloss:0.2443
[295]   train-logloss:0.231894  test-logloss:0.244296

                   _ Feature  Score
13                _user_reorder_ratio    935
11   _user_std_days_since_prior_order    857
14               _user_average_basket    738
9   _user_mean_days_since_prior_order    672
5               time_since_last_order    608
16   _user_sum_days_since_prior_order    569
1          _up_order_since_last_order    537
6    _up_order_rate_since_first_order    495
4                  _prod_reorder_prob    483
17            _user_distinct_products    470
29               _user_total_products    464
23                    _order_hod_mean    452
7               _prod_mean_cart_order    448
3                 _prod_reorder_ratio    446
8                      _order_hod_std    423
19          _up_average_cart_position    419
31                              W2V_1    410
0                      _up_order_rate    406
18                  _usr_prd_buy_rate    403
30               _prod_std_cart_order    388
32                              W2V_2    386
12              _up_cart_position_std    376
21                     _order_std_day    348
20                    _order_mean_day    341
26     _prod_buy_first_time_total_cnt    301
10              _up_last_order_number    277
27             _prod_reorder_tot_cnts    232
15                     _prod_tot_cnts    215
28             _up_first_order_number    209
22                 _user_total_orders    166
25    _prod_buy_second_time_total_cnt    157
2                     _up_order_count    137
24             _total_time_of_reorder     18
33                _prod_reorder_times     17
"""

watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=300,verbose_eval=10, evals=watchlist)
dtest = xgb.DMatrix(testing[col])
pred = model.predict(dtest)
FI = get_feature_importance(model)
df = pd.DataFrame({'reordered':testing['reordered'],'prob':pred})
pb = np.mean(df[df['reordered']==1]['prob'])
F1_Score = Cross_Validation(testing,pred,0.20)
### F1 Score: 0.376452

## 尝试加入Faron的特征，然后再次学习验证

del training
del testing


ostreak = pd.read_csv('data/order_streaks.csv')
train_set = train_set.merge(ostreak,on=['user_id','product_id'],how='left')
test_set = test_set.merge(ostreak,on=['user_id','product_id'],how='left')
training = train_set.iloc[train_index,:]
testing = train_set.iloc[test_index,:]
col.append('order_streak')
dtrain = xgb.DMatrix(training[col],training['reordered'])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 8
    ,"min_child_weight" :5
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :5
}

watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=90,verbose_eval=10, evals=watchlist,early_stopping_rounds=20)
dtest = xgb.DMatrix(testing[col])
pred = model.predict(dtest)
FI = get_feature_importance(model)
FScore = Cross_Validation(testing,pred,0.20)
"""

depth =5, 90 rounds: 0.37750296911560499
depth =7, 90 rounds: 0.37787870009033381
depth = 8, 90 rounds,0.37745308075830131
Feature  Score
16                _user_reorder_ratio   1051
6               time_since_last_order    779
19               _user_average_basket    685
25   _user_std_days_since_prior_order    655
17  _user_mean_days_since_prior_order    570
5                  _prod_reorder_prob    564
10   _user_sum_days_since_prior_order    553
4                 _prod_reorder_ratio    486
11   _up_order_rate_since_first_order    481
14              _prod_mean_cart_order    449
24          _up_average_cart_position    447
1          _up_order_since_last_order    445
13            _user_distinct_products    433
18                              W2V_1    424
20                              W2V_2    395
0                      _up_order_rate    393
8                _user_total_products    375
26                    _order_hod_mean    372
7                      _order_hod_std    355
28               _prod_std_cart_order    353
29                  _usr_prd_buy_rate    320
21     _prod_buy_first_time_total_cnt    313
9               _up_cart_position_std    305
27                     _order_std_day    281
15              _up_last_order_number    278
23                    _order_mean_day    264
22             _prod_reorder_tot_cnts    227
12             _up_first_order_number    213
30                     _prod_tot_cnts    213
3                     _up_order_count    208
33    _prod_buy_second_time_total_cnt    173
2                        order_streak    173
32                 _user_total_orders    165
31                _prod_reorder_times     30
34             _total_time_of_reorder      5
depth =8,300 rounds
0.37528947442167687
Feature  Score
16                _user_reorder_ratio   2215
25   _user_std_days_since_prior_order   1891
19               _user_average_basket   1842
17  _user_mean_days_since_prior_order   1526
10   _user_sum_days_since_prior_order   1381
20                              W2V_2   1359
6               time_since_last_order   1352
18                              W2V_1   1331
14              _prod_mean_cart_order   1274
5                  _prod_reorder_prob   1237
24          _up_average_cart_position   1234
26                    _order_hod_mean   1229
13            _user_distinct_products   1174
4                 _prod_reorder_ratio   1168
28               _prod_std_cart_order   1124
7                      _order_hod_std   1060
29                  _usr_prd_buy_rate   1046
8                _user_total_products   1030
23                    _order_mean_day   1022
11   _up_order_rate_since_first_order    984
9               _up_cart_position_std    978
27                     _order_std_day    951
21     _prod_buy_first_time_total_cnt    845
0                      _up_order_rate    819
1          _up_order_since_last_order    700
15              _up_last_order_number    671
22             _prod_reorder_tot_cnts    606
12             _up_first_order_number    582
30                     _prod_tot_cnts    546
33    _prod_buy_second_time_total_cnt    481
32                 _user_total_orders    425
2                        order_streak    421
3                     _up_order_count    320
31                _prod_reorder_times     61
34             _total_time_of_reorder      8


depth = 6
0.37670321428725861
Feature  Score
7                 _user_reorder_ratio    892
15   _user_std_days_since_prior_order    745
19               _user_average_basket    733
9               time_since_last_order    607
23   _user_sum_days_since_prior_order    587
20  _user_mean_days_since_prior_order    578
5                  _prod_reorder_prob    499
8             _user_distinct_products    489
1          _up_order_since_last_order    466
28               _prod_std_cart_order    441
30                    _order_hod_mean    439
10                     _order_hod_std    438
33               _user_total_products    435
11              _prod_mean_cart_order    434
24                              W2V_1    425
6    _up_order_rate_since_first_order    423
4                 _prod_reorder_ratio    422
27                              W2V_2    409
0                      _up_order_rate    402
21          _up_average_cart_position    383
18                  _usr_prd_buy_rate    359
13              _up_cart_position_std    359
16                     _order_std_day    358
12     _prod_buy_first_time_total_cnt    357
32                    _order_mean_day    309
14              _up_last_order_number    299
26             _prod_reorder_tot_cnts    254
2                        order_streak    225
22             _up_first_order_number    222
17                     _prod_tot_cnts    214
29    _prod_buy_second_time_total_cnt    189
31                 _user_total_orders    176
3                     _up_order_count    140
25                _prod_reorder_times     20
34             _total_time_of_reorder      4

"""