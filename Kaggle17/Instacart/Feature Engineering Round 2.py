# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 15:24:29 2017

@author: Jiashen Liu

Purpose: Create more features for the model.

"""

from functions_in import *
import pandas as pd
import numpy as np

DIR = 'data/'

priors, train, orders, products, aisles, departments, sample_submission = load_data(DIR)

product_pca = pd.read_csv('data/product_pca_features.csv')
product_pca = product_pca.drop('product_name',axis=1)
products = pd.read_csv('data/products.csv')
products = products.merge(product_pca,how='left',on='product_id')

del product_pca

"""
orders这个表里有的是订单的信息，而priors这个表内有的信息是每一个订单里对应的每一件产品的信息。
将prior作为左表将订单的信息加入到产品表内去。
"""
priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')
priors_orders_detail['days_since_prior_order'] = priors_orders_detail['days_since_prior_order'].fillna(0)

### 某用户买某个产品是第几次。 cumcount返回的是：某值是第几次出现的。
priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

###加产品信息特征，分别为：产品被买了几次，产品被reorder了几次，后面sum的两个特征不好理解。
###加入特征：关于产品被放入购物篮顺序的均值以及标准差，和产品
agg_dict = {'user_id':{'_prod_tot_cnts':'count'}, 
            'reordered':{'_prod_reorder_tot_cnts':'sum'}, 
            '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)},
                                         'add_to_cart_order':{'_prod_mean_order_add':'mean',
                                                              '_prod_std_order_add':'std',
                                                              '_prod_first_put_count':lambda x: sum(x==1)}}
prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

# _prod_reorder_ratio: 商品复购率
prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt
prd['_prod_first_put_ratio'] = prd._prod_first_put_count/prd._prod_tot_cnts


###加入新特征： 关于department和aisle

agg_dict_dpt = {'user_id':{'_dept_tot_cnts':'count'}, 
            'reordered':{'_dept_reorder_tot_cnts':'sum'}, 
            '_user_buy_product_times': {'_dept_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_dept_buy_second_time_total_cnt':lambda x: sum(x==2)},
            'add_to_cart_order':{'_dept_mean_order_add':'mean',
                                 '_dept_std_order_add':'std',
                                 '_dept_first_put_count':lambda x: sum(x==1)}}

priors_orders_detail = priors_orders_detail.merge(products[['product_id','aisle_id','department_id']],on='product_id',how = 'left')

dept = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['department_id'], agg_dict_dpt)

dept['_dept_reorder_prob'] = dept._dept_buy_second_time_total_cnt / dept._dept_buy_first_time_total_cnt
dept['_dept_reorder_ratio'] = dept._dept_reorder_tot_cnts / dept._dept_tot_cnts
dept['_dept_reorder_times'] = 1 + dept._dept_reorder_tot_cnts / dept._dept_buy_first_time_total_cnt
dept['_dept_first_put_ratio'] = dept._dept_first_put_count/dept._dept_tot_cnts

agg_dict_ais = {'user_id':{'_aisle_tot_cnts':'count'}, 
            'reordered':{'_aisle_reorder_tot_cnts':'sum'}, 
            '_user_buy_product_times': {'_aisle_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_aisle_buy_second_time_total_cnt':lambda x: sum(x==2)},
            'add_to_cart_order':{'_aisle_mean_order_add':'mean',
                                 '_aisle_std_order_add':'std',
                                 '_aisle_first_put_count':lambda x: sum(x==1)}}
ais = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['aisle_id'], agg_dict_ais)

ais['_ais_reorder_prob'] = ais._aisle_buy_second_time_total_cnt / ais._aisle_buy_first_time_total_cnt
ais['_ais_reorder_ratio'] = ais._aisle_reorder_tot_cnts / ais._aisle_tot_cnts
ais['_ais_reorder_times'] = 1 + ais._aisle_reorder_tot_cnts / ais._aisle_buy_first_time_total_cnt
ais['_ais_first_put_ratio'] = ais._aisle_first_put_count/ais._aisle_tot_cnts



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

## 用户与产品交互部分特征

agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                              '_up_first_order_number': 'min', 
                              '_up_last_order_number':'max'}, 
              'add_to_cart_order':{'_up_average_cart_position': 'mean',
                                   '_up_std_cart_position':'std'},
              'days_since_prior_order':{'_mean_day_between_orders':'mean',
                                        '_std_day_between_orders':'std',
                      }
              }

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

###Release Memory###
products = products.merge(dept,on='department_id',how='left')
col =[each for each in list(products.columns) if 'pca' not in each]
products = products[col]
products = products.drop('product_name',axis=1)
del train
del prd
del orders
del priors_orders_detail
del priors
del usr_day_count,usr_day_count_max,usr_pupular_day,usr_pupular_time,usr_time_count,usr_time_count_max,departments,aisles,dept
del users
del us
##尝试，希望顺利！！！##
train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
del data
del train['eval_set']
del test['eval_set']
train = train.merge(products,on='product_id',how='left')
train = train.merge(ais,on='aisle_id',how='left')
test = test.merge(products,on='product_id',how='left')
test = test.merge(ais,on='aisle_id',how='left')
##现阶段可以添加进来60个特征,训练xgboost并进行CV

import xgboost as xgb
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
testing = train.iloc[test_index,:]
del train
training['reordered']=training['reordered'].fillna(0)
testing['reordered']=testing['reordered'].fillna(0)
test['reordered'] = test['reordered'].fillna(0)
col = list(training.columns)
col.remove('reordered')
col.remove('user_id')
#col.remove('product_id')
col.remove('order_id')

dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(testing[col])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6 ## Originally 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(dtrain, "train")]
model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=90, evals=watchlist, verbose_eval=10)

F_score = get_feature_importance(model)
pred_test = model.predict(dtest)
from sklearn.metrics import log_loss
pred = [1 if each>0.5 else 0 for each in pred_test]
log_loss(pred,testing['reordered'])

"""
[0]     train-logloss:0.625817
[10]    train-logloss:0.336062
[20]    train-logloss:0.269095
[30]    train-logloss:0.251503
[40]    train-logloss:0.246628
[50]    train-logloss:0.244803
[60]    train-logloss:0.243866
[70]    train-logloss:0.243194
[80]    train-logloss:0.24258
3.1190393506331153
"""

validation_file = CV_file2(testing)
testing['reordered'] = pred_test
test_file = sub_file(testing,0.20,validation_file)
valid = validation_file.merge(test_file,on='order_id',how='left')
valid['products_o'] = valid['products_o'].apply(lambda x:string_to_list(x))
valid['products'] = valid['products'].apply(lambda x:string_to_list(x))
f1_score(valid['products_o'],valid['products'])
##0.37788365002339136/0.19
##0.3777117489448959

del dtest
dtest = xgb.DMatrix(test[col])
pred = model.predict(dtest)
test['reordered'] = pred
sub = sub_file(test,0.21,sample_submission)
sub.to_csv('submission/more_features_20170723_10%training_0.20threshold.csv',index=False)
sub.to_csv('submission/more_features_20170723_10%training_0.19threshold.csv',index=False)



###按50%的数据集尝试一下看看是否会有提升

train = pd.concat([training,testing])

del training,testing
train['reordered'] = train['reordered'].fillna(0)
del dtrain
del dtest
kf = GroupKFold(n_splits=2) 
train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train, groups=train['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)
test_index = train_indexes[0]
train_index = test_indexes[0]
training = train.iloc[train_index,:]
testing = train.iloc[test_index,:]
del train

dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(testing[col])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.03
    ,"max_depth"        : 6 ## Originally 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(dtrain, "train")]
model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=300, evals=watchlist, verbose_eval=10)
pred_test = model.predict(dtest)

pred = [1 if each>0.5 else 0 for each in pred_test]
log_loss(pred,testing['reordered'])

validation_file = CV_file2(testing)
testing['reordered'] = pred_test
test_file = sub_file(testing,0.19,validation_file)
valid = validation_file.merge(test_file,on='order_id',how='left')
valid['products_o'] = valid['products_o'].apply(lambda x:string_to_list(x))
valid['products'] = valid['products'].apply(lambda x:string_to_list(x))
f1_score(valid['products_o'],valid['products'])

del dtest
del dtrain

dtest = xgb.DMatrix(test[col])
pred = model.predict(dtest)
test['reordered'] = pred
sub = sub_file(test,0.21,sample_submission)
sub.to_csv('submission/more_features_20170723_50%training_0.19threshold.csv',index=False)
sub.to_csv('submission/more_features_20170723_50%training_0.20threshold.csv',index=False)
sub.to_csv('submission/more_features_20170723_50%training_0.21threshold.csv',index=False)


"""
[0]     train-logloss:0.672167
[10]    train-logloss:0.516479
[20]    train-logloss:0.4226
[30]    train-logloss:0.362978
[40]    train-logloss:0.324047
[50]    train-logloss:0.298335
[60]    train-logloss:0.281184

[70]    train-logloss:0.269827
[80]    train-logloss:0.262265
[90]    train-logloss:0.257229
[100]   train-logloss:0.253917
[110]   train-logloss:0.251738
[120]   train-logloss:0.250295
[130]   train-logloss:0.249341
[140]   train-logloss:0.24871
[150]   train-logloss:0.248273
[160]   train-logloss:0.247978
[170]   train-logloss:0.247763
[180]   train-logloss:0.247601
[190]   train-logloss:0.247482
[200]   train-logloss:0.247392
[210]   train-logloss:0.247311
[220]   train-logloss:0.247245
[230]   train-logloss:0.247189
[240]   train-logloss:0.247135
[250]   train-logloss:0.247095
[260]   train-logloss:0.247055
[270]   train-logloss:0.24702
[280]   train-logloss:0.246989
[290]   train-logloss:0.24696

"""



###部门的数据实在是进不来，考虑计划B
#
#train = data[data['eval_set']=='train']
#test = data[data['eval_set']=='test']
#del data
#
#
#
#
### 部门与用户交互部分特征
#
#agg_dict_5 = {'order_number':{'_dept_up_order_count': 'count', 
#                              '_dept_up_first_order_number': 'min', 
#                              '_dept_up_last_order_number':'max'}, 
#              'add_to_cart_order':{'_dept_up_average_cart_position': 'mean',
#                                   '_dept_up_std_cart_position':'std'},
#              'days_since_prior_order':{'_dept_mean_day_between_orders':'mean',
#                                        '_dept_std_day_between_orders':'std',
#                      }
#              }
#
#data2 = ka_add_groupby_features_1_vs_n(df=priors_orders_detail, 
#                                                      group_columns_list=['user_id', 'department_id'], 
#                                                      agg_dict=agg_dict_5)
#
#data2 = data2.merge(dept, how='inner', on='department_id').merge(users, how='inner', on='user_id')
## 该商品购买次数 / 总的订单数
## 最近一次购买商品 - 最后一次购买该商品
## 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
#data2['_dept_up_order_rate'] = data2._dept_up_order_count / data2._user_total_orders
#data2['_dept_up_order_since_last_order'] = data2._user_total_orders - data2._dept_up_last_order_number
#data2['_dept_up_order_rate_since_first_order'] = data2._dept_up_order_count / (data2._user_total_orders - data2._dept_up_first_order_number + 1)
#
### Aisle 与用户交互特征
#
#agg_dict_6 = {'order_number':{'_ais_up_order_count': 'count', 
#                              '_ais_up_first_order_number': 'min', 
#                              '_ais_up_last_order_number':'max'}, 
#              'add_to_cart_order':{'_ais_up_average_cart_position': 'mean',
#                                   '_ais_up_std_cart_position':'std'},
#              'days_since_prior_order':{'_ais_mean_day_between_orders':'mean',
#                                        '_ais_std_day_between_orders':'std',
#                      }
#              }
#
#data3 = ka_add_groupby_features_1_vs_n(df=priors_orders_detail, 
#                                                      group_columns_list=['user_id', 'aisle_id'], 
#                                                      agg_dict=agg_dict_6)
#
#data3 = data3.merge(ais, how='inner', on='aisle_id').merge(users, how='inner', on='user_id')
## 该商品购买次数 / 总的订单数
## 最近一次购买商品 - 最后一次购买该商品
## 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
#data3['_ais_up_order_rate'] = data3._ais_up_order_count / data3._user_total_orders
#data3['_ais_up_order_since_last_order'] = data3._user_total_orders - data3._ais_up_last_order_number
#data3['_ais_up_order_rate_since_first_order'] = data3._ais_up_order_count / (data3._user_total_orders - data3._ais_up_first_order_number + 1)
#
#col_data2 = list(data2.columns)
#col_data2 = [each for each in col_data2 if each not in list(test.columns)]
#col_data2.append('user_id')
#data2 = data2[col_data2]
#col_data3 = list(data3.columns)
#col_data3 = [each for each in col_data3 if each not in list(test.columns)]
#col_data3.append('user_id')
#data3 = data3[col_data3]
#data2.to_csv('dept_user_interaction.csv',index=False)
#data3.to_csv('ais_user_interaction.csv',index=False)
#
##del train
#del prd
#del orders
#del priors_orders_detail
#del priors
#del usr_day_count,usr_day_count_max,usr_pupular_day,usr_pupular_time,usr_time_count,usr_time_count_max,departments,aisles
#del users
#del us
#del products
#col_data2 = list(data2.columns)
#col_data2 = [each for each in col_data2 if each not in list(data.columns)]
#col_data2.append('user_id')
#data2 = data2[col_data2]
#del data3
#
