# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:39:39 2017

@author: jiashen
"""

from functions_in import *
import pandas as pd
import numpy as np
import time

DIR = 'data/'
W2C = pd.read_csv('data/WORD2VEC_Feat.csv')
ostreak = pd.read_csv(DIR+'order_streaks.csv')
priors, train, orders, products, aisles, departments, sample_submission = load_data(DIR)
product_name = pd.read_csv(DIR+'product_name_pca_2comp.csv')

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
                                '_prod_std_cart_order':'std'},
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
prd = prd.merge(products,on='product_id',how='left')
prd = prd.merge(departments,on='department_id',how='left')
prd = prd.merge(aisles,on='aisle_id',how='left')
del prd['department']
del prd['aisle']

print('product done')
print(time.ctime())
"""
创建dept和ais的单独特征
"""
priors_orders_detail = priors_orders_detail.merge(prd,on='product_id',how='left')
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
print('User Done')
print(time.ctime())
"""
0730： 加四个关于department和aisle的特征
从数据上看，department和aisle是类似于商品类别的数据，可能会非常有用。因此想知道：
这个department/aisle是否是用户最喜欢的？
这个department/aisle是否是用户最常订购的？
如果好用，可以再加最近订购的参数，类似一样的。
在最后的数据部分，由user_id和department_id/aisle_id一起join，也可以把这些作为一个特征输入进去。看看内存。
0731加入部门和aisle的recency特征
0801加入部门和aisle的dow/hour of day特征
"""
agg_dict_dept = {'user_id':{'_user_dept_total_orders':'count'},
              'reordered':{'_user_dept_total_reorders':'sum'},
              'days_since_prior_order':{'_user_dept_sum_days_since_prior_order':'sum', 
                                        '_user_dept_mean_days_since_prior_order': 'mean',
                                        '_user_dept_std_days_since_prior_order':'std',
                                        '_user_dept_median_days_since_prior_order':'median'},
               'order_dow':{'_user_dpet_mean_dow':'mean',
                          '_user_dept_std_dow':'std',
                          '_user_dept_median_dow':'median'},
              'order_hour_of_day':{'_user_dept_mean_hod':'mean',
                                   '_user_dept_std_hod':'std',
                                    '_user_dept_median_hod':'median'}
        }
agg_dict_ais = {'user_id':{'_user_ais_total_orders':'count'},
              'reordered':{'_user_ais_total_reorders':'sum'},
              'days_since_prior_order':{'_user_ais_sum_days_since_prior_order':'sum', 
                                        '_user_ais_mean_days_since_prior_order': 'mean',
                                        '_use_aisr_std_days_since_prior_order':'std',
                                        '_user_ais_median_days_since_prior_order':'median'},
               'order_dow':{'_user_ais_mean_dow':'mean',
                          '_user_ais_std_dow':'std',
                          '_user_ais_median_dow':'median'},
              'order_hour_of_day':{'_user_ais_mean_hod':'mean',
                                   '_user_ais_std_hod':'std',
                                    '_user_ais_median_hod':'median'}
        }

user_dept_data =  ka_add_groupby_features_1_vs_n(priors_orders_detail, 
                                                      group_columns_list=['user_id', 'department_id'], 
                                                      agg_dict=agg_dict_dept)
user_ais_data = ka_add_groupby_features_1_vs_n(priors_orders_detail, 
                                                      group_columns_list=['user_id', 'aisle_id'], 
                                                      agg_dict=agg_dict_ais)
user_dept_data['_user_dept_reorder_rate'] = user_dept_data['_user_dept_total_reorders']/user_dept_data['_user_dept_total_orders']
user_ais_data['_user_ais_reorder_rate'] = user_ais_data['_user_ais_total_reorders']/user_ais_data['_user_ais_total_orders']
print('User Ais Dept Done')
print(time.ctime())

"""
prepare more feature: days between the purchase of two products

Use orders and prior_order_detail
"""
orders = orders.sort_values(['user_id', 'order_number'])
user_cumday = orders.groupby('user_id')['days_since_prior_order'].apply(lambda x: x.cumsum())
user_cumday = pd.DataFrame({'user_id':list(user_cumday.index),'cum_days':list(user_cumday)})
orders['cum_days'] = user_cumday['cum_days']
del user_cumday
priors_orders_detail = priors_orders_detail.sort_values(['user_id', 'product_id', 'order_number'])
priors_orders_detail['last_order_id'] = priors_orders_detail.groupby(['user_id', 'product_id'])['order_id'].shift(1)
tmp = orders[['order_id','cum_days']] 
tmp.columns = ['order_id','order_cum_current']
priors_orders_detail = priors_orders_detail.merge(tmp,on='order_id',how='left')
tmp.columns = ['last_order_id','order_cum_previous']
priors_orders_detail = priors_orders_detail.merge(tmp,on='last_order_id',how='left')
priors_orders_detail['UP_days_since_last_order'] = priors_orders_detail['order_cum_current']-priors_orders_detail['order_cum_previous']

del priors_orders_detail['order_cum_previous']
del priors_orders_detail['last_order_id']
del tmp

"""
特征组3：用户和产品交互特征
0731： 加入：用户与产品的订购recency特征
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
               'reordered':{'_total_time_of_reorder':'sum'},
               'days_since_prior_order':{'_user_prd_sum_days_since_prior_order':'sum', 
                                        '_user_prd_mean_days_since_prior_order': 'mean',
                                        '_use_prd_std_days_since_prior_order':'std',
                                        '_user_prd_median_days_since_prior_order':'median'},
                'UP_days_since_last_order':{'_order_interval_max':'max',
                                            '_order_interval_mean':'mean',
                                            '_order_interval_std':'std'},
                'order_cum_current':{'_cum_interval_max':'max',
                                     '_cum_interval_mean':'mean',
                                     '_cum_interval_std':'std'}}

data = ka_add_groupby_features_1_vs_n(df=priors_orders_detail, 
                                                      group_columns_list=['user_id', 'product_id'], 
                                                      agg_dict=agg_dict_4)

data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')
print('Data Done 1')
print(time.ctime())
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
print('Data Done 2')
print(time.ctime())

del train, prd, users,priors_orders_detail, orders
del products,us,aisles,departments,priors,ostreak,W2C
del agg_dict,agg_dict_2,agg_dict_4,agg_dict_ais,agg_dict_dept

data = data.merge(user_dept_data,on=['user_id','department_id'],how='left')
del user_dept_data
data = data.merge(user_ais_data,on=['user_id','aisle_id'],how='left')
del user_ais_data
data = data.merge(product_name,on='product_id',how='left')
print('Data Done 3')
print(time.ctime())
#data = data.merge(dept,on='department_id',how='left').merge(ais,on='aisle_id',how='left')
#del dept,ais

train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
del data

print('Data Done')
print(time.ctime())

print('Ends')
print(time.ctime())

col = list(train.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')
col.remove('department_id')
col.remove('aisle_id')

import xgboost as xgb

dtrain = xgb.DMatrix(train[col],train['reordered'])
dtest = xgb.DMatrix(test[col])
del train

xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.05
    ,'max_depth':8
    ,"colsample_bytree" :0.6
    ,"min_child_weight ": 20
}

watchlist= [(dtrain, "train")]
print('start training')
print(time.ctime())
model = xgb.train(xgb_params,dtrain,num_boost_round=520,verbose_eval=10, evals=watchlist)
print(time.ctime())
print('Start predicting')
pred = model.predict(dtest)
print('Prediction Done')
print(time.ctime())
test['reordered'] = pred
sub = sub_file(test,0.20,sample_submission)
test_set2 = test[['user_id','product_id','order_id','reordered']]
test_set2.to_csv('save_test_for_exp_20170809_Azure_86_feats_gap_xgb_520_rounds_0.05_depth8.csv',index=False)
sub.to_csv('20170809_Azure_86_feats_gap_xgb_520_rounds_0.05_depth8.csv',index=False)
FI = get_feature_importance(model)
FI.to_csv('Feature_Importance_86_feats_gap_xgb_520_rounds_0.05_0809.csv',index=False)

"""
start training
Wed Aug  9 00:29:40 2017
[0]     train-logloss:0.658379
[10]    train-logloss:0.441205
[20]    train-logloss:0.342638
[30]    train-logloss:0.293971
[40]    train-logloss:0.269341
[50]    train-logloss:0.256709
[60]    train-logloss:0.250266
[70]    train-logloss:0.246948
[80]    train-logloss:0.245226
[90]    train-logloss:0.2442
[100]   train-logloss:0.243573
[110]   train-logloss:0.24315
[120]   train-logloss:0.242792
[130]   train-logloss:0.242472
[140]   train-logloss:0.242205
[150]   train-logloss:0.241926
[160]   train-logloss:0.241689
[170]   train-logloss:0.241444
[180]   train-logloss:0.241217
[190]   train-logloss:0.241008
[200]   train-logloss:0.240798
[210]   train-logloss:0.240604
[220]   train-logloss:0.240414
[230]   train-logloss:0.240248
[240]   train-logloss:0.2401
[250]   train-logloss:0.239941
[260]   train-logloss:0.239769
[270]   train-logloss:0.23961
[280]   train-logloss:0.239456
[290]   train-logloss:0.239289
[300]   train-logloss:0.239133
[310]   train-logloss:0.238979
[320]   train-logloss:0.238845
[330]   train-logloss:0.238713
[340]   train-logloss:0.23857
[350]   train-logloss:0.238419
[360]   train-logloss:0.238278
[370]   train-logloss:0.238128
[380]   train-logloss:0.238004
[390]   train-logloss:0.237863
[400]   train-logloss:0.237724
[410]   train-logloss:0.237595
[420]   train-logloss:0.23744
[430]   train-logloss:0.237301
[440]   train-logloss:0.237161
[450]   train-logloss:0.237042
[460]   train-logloss:0.236935
[470]   train-logloss:0.236814
[480]   train-logloss:0.236689
[490]   train-logloss:0.236579
[500]   train-logloss:0.236459
[510]   train-logloss:0.236349
Wed Aug  9 03:48:24 2017
Start predicting
Prediction Done
Wed Aug  9 03:50:34 2017

"""