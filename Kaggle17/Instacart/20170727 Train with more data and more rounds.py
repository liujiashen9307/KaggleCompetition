# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:50:30 2017

@author: Jiashen Liu

Purpose: Train with more data!
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
us = pd.read_csv('user_transform.csv')

users = users.merge(us, how='inner')

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
del train, prd, users,us
# gc.collect()
# release Memory
del priors_orders_detail, orders
del products,aisles,departments,priors

#data.to_csv('data_bakup.csv',index=False)


## Try ML Algorithm

train_set = data[data['eval_set']=='train']
test_set = data[data['eval_set']=='test'] 
W2C = pd.read_csv('data/WORD2VEC_Feat.csv')
train_set = train_set.merge(W2C,on='product_id',how='left')
test_set = test_set.merge(W2C,on='product_id',how='left')
ostreak = pd.read_csv('data/order_streaks.csv')
train_set = train_set.merge(ostreak,on=['user_id','product_id'],how='left')
test_set = test_set.merge(ostreak,on=['user_id','product_id'],how='left')

from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits=4) 
train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train_set, groups=train_set['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)
test_index = test_indexes[0]
train_index = train_indexes[0]
training = train_set.iloc[train_index,:]
testing = train_set.iloc[test_index,:]
del data,ostreak,W2C

import xgboost as xgb

col = list(train_set.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')

dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(test_set[col])
dtesting = xgb.DMatrix(testing[col])

xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.03
    ,"max_depth"        : 7
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=500,verbose_eval=10, evals=watchlist,early_stopping_rounds=20)
##CV一下
pred = model.predict(dtesting)

FScore = Cross_Validation(testing,pred,0.21)
## 0.20: 0.38197155062044263,[490]   train-logloss:0.242276
del pred
pred = model.predict(dtest)
test_set['reordered'] = pred
sub = sub_file(test_set,0.20,sample_submission)
sub.to_csv('submission/20170727_sub_more_feature_75%_data_0.382.csv',index=False)
##LB: 0.3823370

##尝试调一下参数，减少训练轮次，增加eta，看看cv的效果有没有提高。
del model
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 8
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=150,verbose_eval=10, evals=watchlist,early_stopping_rounds=20)
pred = model.predict(dtesting)

FScore = Cross_Validation(testing,pred,0.20)
## Rounds 100, max_depth 6: eta 0.1  F_Score: 0.38145736190049112,[90]    train-logloss:0.244252
## Rounds 100, max_depth 8: eta 0.1  0.38183763453949993          [90]    train-logloss:0.242404
## Rounds 150, max_depth 8: eta 0.1  0.38218127482892861          [140]   train-logloss:0.241016

del pred
pred = model.predict(dtest)
test_set['reordered'] = pred
sub = sub_file(test_set,0.20,sample_submission)
sub.to_csv('submission/20170727_sub_more_feature_75%_data_depth8_0.3822.csv',index=False)
## 0.3817864, Overfit了

##尝试用所有的数据进行提交，max_depth设置回是6，eta设为0.1，300轮次，今日最后一次。
del model
del dtrain
dtrain = xgb.DMatrix(train_set[col],train_set['reordered'])
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

watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=150,verbose_eval=10, evals=watchlist,early_stopping_rounds=20)
'''
[130]   train-logloss:0.243801
[140]   train-logloss:0.24369
'''
pred = model.predict(dtest)
test_set['reordered'] = pred
sub = sub_file(test_set,0.20,sample_submission)
sub.to_csv('submission/20170727_sub_more_feature_full_data_depth6_eta0.1_150rounds_0.20thres.csv',index=False)

