# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 21:28:53 2017

@author: Jiashen Liu

Purpose: Further Understand Current Feature Creation, and Try to create more useful features

"""

from functions_in import *
import pandas as pd
import numpy as np

DIR = 'data/'

priors, train, orders, products, aisles, departments, sample_submission = load_data(DIR)

"""

研究现有的特征提取手段，标记每一次特征提取的目的

"""

"""
orders这个表里有的是订单的信息，而priors这个表内有的信息是每一个订单里对应的每一件产品的信息。
将prior作为左表将订单的信息加入到产品表内去。
"""
priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

### 某用户买某个产品是第几次。 cumcount返回的是：某值是第几次出现的。
priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

###加产品信息特征，分别为：产品被买了几次，产品被reorder了几次，后面sum的两个特征不好理解。
agg_dict = {'user_id':{'_prod_tot_cnts':'count'}, 
            'reordered':{'_prod_reorder_tot_cnts':'sum'}, 
            '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)}}
prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

# _prod_reorder_ratio: 商品复购率
prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt


"""
留白，想办法在产品区创造新特征
"""

### 用户信息特征,考虑自己添加：用户最喜欢在哪天进行购物，以及用户最喜欢在几点进行购物。

agg_dict_2 = {'order_number':{'_user_total_orders':'max'}, ## 用户的总订单
              'days_since_prior_order':{'_user_sum_days_since_prior_order':'sum', ## 用户下单周期的总和 
                                        '_user_mean_days_since_prior_order': 'mean',
                                        '_user_std_days_since_prior_order':'std'}}## 用户下单周期的平均值
 
users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'], agg_dict_2)

## 分别创建：用户哪天购物，以及用户在几点购物的订单数量和，并且选出最大的对应的天和时间

usr_day_count = orders[orders.eval_set == 'prior'].groupby(['user_id','order_dow']).size()
usr_day_count = usr_day_count.reset_index()
usr_day_count.columns=['user_id','order_dow','day_count']
usr_day_count_max = usr_day_count.groupby(['user_id']).aggregate({'day_count':{'popular_day_count':'max'}})
usr_day_count_max = pd.DataFrame({'user_id':list(usr_day_count_max.index),'maximum_count':list(usr_day_count_max.iloc[:,0])})
usr_day_count = usr_day_count.merge(usr_day_count_max,on='user_id',how='left')
usr_day_count = usr_day_count.query('day_count==maximum_count')
usr_pupular_day = usr_day_count[['user_id','order_dow']].drop_duplicates('user_id')
usr_pupular_day.columns = ['user_id','popular_dow']

### 同过程，查询最喜爱的时间并返回。

usr_time_count = orders[orders.eval_set == 'prior'].groupby(['user_id','order_hour_of_day']).size()
usr_time_count = usr_time_count.reset_index()
usr_time_count.columns=['user_id','order_hour_of_day','time_count']
usr_time_count_max = usr_time_count.groupby(['user_id']).aggregate({'time_count':{'popular_time_count':'max'}})
usr_time_count_max = pd.DataFrame({'user_id':list(usr_time_count_max.index),'maximum_count':list(usr_time_count_max.iloc[:,0])})
usr_time_count = usr_time_count.merge(usr_time_count_max,on='user_id',how='left')
usr_time_count = usr_time_count.query('time_count==maximum_count')
usr_pupular_time = usr_time_count[['user_id','order_hour_of_day']].drop_duplicates('user_id')
usr_pupular_time.columns = ['user_id','popular_hour_of_day']

users = users.merge(usr_pupular_day,on='user_id',how='left')
users = users.merge(usr_pupular_time,on='user_id',how='left')

### 根据用户，继续建立新的指标

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
## 这部分时间最久。
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
                                   '_up_std_cart_position':'std'}}

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

# add user_id to train set
train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

## Release Memory

del train, prd, users
# gc.collect()
# release Memory
del priors_orders_detail, orders
data['reordered'] = data['reordered'].fillna(0)
train_set = data[data['eval_set']=='train']
test_set = data[data['eval_set']=='test']

col = list(train_set.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
#col.remove('product_id')
col.remove('order_id')

from sklearn.model_selection import GroupKFold

kf = GroupKFold(n_splits=2) 

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
dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(testing[col])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.03
    ,"max_depth"        : 7 ## Originally 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(dtrain, "train")]
model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=300, evals=watchlist, verbose_eval=10)

"""
[0]     train-logloss:0.626058
[10]    train-logloss:0.336104
[20]    train-logloss:0.26914
[30]    train-logloss:0.251691
[40]    train-logloss:0.246789
[50]    train-logloss:0.245116
[60]    train-logloss:0.244258
[70]    train-logloss:0.243642
[80]    train-logloss:0.243099

With depth at 7

[0]     train-logloss:0.625872
[10]    train-logloss:0.335234
[20]    train-logloss:0.26786
[30]    train-logloss:0.250149
[40]    train-logloss:0.245126
[50]    train-logloss:0.243218
[60]    train-logloss:0.242171
[70]    train-logloss:0.241407
[80]    train-logloss:0.24067


[0]     train-logloss:0.672122
[10]    train-logloss:0.516119
[20]    train-logloss:0.421921
[30]    train-logloss:0.362094
[40]    train-logloss:0.323005
[50]    train-logloss:0.297121
[60]    train-logloss:0.279828
[70]    train-logloss:0.268309
[80]    train-logloss:0.260586
[90]    train-logloss:0.255419
[100]   train-logloss:0.251958
[110]   train-logloss:0.249645
[120]   train-logloss:0.248081
[130]   train-logloss:0.247025
[140]   train-logloss:0.246293
[150]   train-logloss:0.245774
[160]   train-logloss:0.245392
[170]   train-logloss:0.245105
[180]   train-logloss:0.244885
[190]   train-logloss:0.244694
[200]   train-logloss:0.244534
[210]   train-logloss:0.244401
[220]   train-logloss:0.244276
[230]   train-logloss:0.244171
[240]   train-logloss:0.244057
[250]   train-logloss:0.243974
[260]   train-logloss:0.243889
[270]   train-logloss:0.2438
[280]   train-logloss:0.243706
[290]   train-logloss:0.243622
Out[131]: 0.37668327397870921

"""

pred_test = model.predict(dtest)
from sklearn.metrics import log_loss
pred = [1 if each>0.5 else 0 for each in pred_test]
log_loss(pred,testing['reordered'])
### 3.1210184136350745
### 3.1178302782915011
### 3.1014726794258927
## Validation

validation_file = CV_file2(testing)

testing2 = testing.copy()
testing2['reordered'] = pred_test

test_file = sub_file(testing2,0.19,validation_file)

valid = validation_file.merge(test_file,on='order_id',how='left')

valid['products_o'] = valid['products_o'].apply(lambda x:string_to_list(x))
valid['products'] = valid['products'].apply(lambda x:string_to_list(x))

f1_score(valid['products_o'],valid['products'])
#0.3770565275264372

## 0.37746711218090295/0.3796188

dtest = xgb.DMatrix(test_set[col])
pred = model.predict(dtest)
test_set['reordered'] = pred
sub = sub_file(test_set,0.19,sample_submission)

sub.head()
sub.to_csv('submission/add_more_features_xgb.csv',index=False)

sub.to_csv('submission/xgb_300rounds_eta0.03_50%sample.csv',index=False)
###0.3809643
### 0.3770565275264372

f_score = get_feature_importance(model)

f_score.to_csv('Feature_Importance.csv',index=False)