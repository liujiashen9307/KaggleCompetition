# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:19:52 2017

@author: jiashen
"""

from functions_in import *
import pandas as pd
import numpy as np
import time

DIR = 'data/'
W2C = pd.read_csv('data/WORD2VEC_Feat.csv')
ostreak = pd.read_csv(DIR+'order_streak_updated.csv')
ostreak = ostreak[['user_id','product_id','streak']]
priors, train, orders, products, aisles, departments, sample_submission = load_data(DIR)
product_name = pd.read_csv(DIR+'product_name_pca_2comp.csv')
print('Begin')
print(time.ctime())

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
                          '_prod_std_dow':'std'},
              'order_hour_of_day':{'_prod_mean_hod':'mean',
                                   '_prod_std_hod':'std'},
               'days_since_prior_order':{
                                        '_prod_sum_days_since_prior_order':'sum', 
                                        '_prod_mean_days_since_prior_order': 'mean',
                                        '_prod_std_days_since_prior_order':'std'
                                         }}
prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
#prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt
prd = prd.merge(products,on='product_id',how='left')
prd = prd.merge(departments,on='department_id',how='left')
prd = prd.merge(aisles,on='aisle_id',how='left')
del prd['department']
del prd['aisle']

print('product done')
print(time.ctime())
"""
创建dept和ais的单独特征
Don't keep too many of features in these categories.

"""
priors_orders_detail = priors_orders_detail.merge(prd,on='product_id',how='left')

agg_dict_dpt = {'user_id':{'_dept_tot_cnts':'count'}, 
            'reordered':{'_dept_reorder_tot_cnts':'sum'},
            '_user_buy_product_times': {'_dept_buy_first_time_total_cnt':lambda x: sum(x==1),
                                         '_dept_buy_second_time_total_cnt':lambda x: sum(x==2)},
               'days_since_prior_order':{
                                        '_dept_mean_days_since_prior_order': 'mean',
                                        '_dept_std_days_since_prior_order':'std'
                                         }}
            
dept = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['department_id'], agg_dict_dpt)
dept['_dept_reorder_prob'] = dept._dept_buy_second_time_total_cnt / dept._dept_buy_first_time_total_cnt
dept['_dept_reorder_ratio'] = dept._dept_reorder_tot_cnts / dept._dept_tot_cnts
del dept['_dept_tot_cnts']
del dept['_dept_reorder_tot_cnts']

agg_dict_ais = {'user_id':{'_aisle_tot_cnts':'count'}, 
            'reordered':{'_aisle_reorder_tot_cnts':'sum'}, 
            '_user_buy_product_times': {'_aisle_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_aisle_buy_second_time_total_cnt':lambda x: sum(x==2)},
               'days_since_prior_order':{
                                        '_aisle_mean_days_since_prior_order': 'mean',
                                        '_aisle_std_days_since_prior_order':'std'
                                         }
            }
ais = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['aisle_id'], agg_dict_ais)
ais['_ais_reorder_prob'] = ais._aisle_buy_second_time_total_cnt / ais._aisle_buy_first_time_total_cnt
ais['_ais_reorder_ratio'] = ais._aisle_reorder_tot_cnts / ais._aisle_tot_cnts
del ais['_aisle_tot_cnts']
del ais['_aisle_reorder_tot_cnts']
print('DEPT AIS DONE')
print(time.ctime())

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
                                        '_user_std_days_since_prior_order':'std'},
               'order_dow':{'_user_mean_dow':'mean',
                          '_user_std_dow':'std'},
              'order_hour_of_day':{'_user_mean_hod':'mean',
                                   '_user_std_hod':'std'}
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
                                        '_user_dept_std_days_since_prior_order':'std'},
               'order_dow':{'_user_dpet_mean_dow':'mean',
                          '_user_dept_std_dow':'std'},
              'order_hour_of_day':{'_user_dept_mean_hod':'mean',
                                   '_user_dept_std_hod':'std'}
        }
agg_dict_ais = {'user_id':{'_user_ais_total_orders':'count'},
              'reordered':{'_user_ais_total_reorders':'sum'},
              'days_since_prior_order':{'_user_ais_sum_days_since_prior_order':'sum', 
                                        '_user_ais_mean_days_since_prior_order': 'mean',
                                        '_use_aisr_std_days_since_prior_order':'std'},
               'order_dow':{'_user_ais_mean_dow':'mean',
                          '_user_ais_std_dow':'std'},
              'order_hour_of_day':{'_user_ais_mean_hod':'mean',
                                   '_user_ais_std_hod':'std'}
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
0805 On Azure: Add features about user-product interval ordering time 
"""

agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                              '_up_first_order_number': 'min', 
                              '_up_last_order_number':'max'}, 
              'add_to_cart_order':{'_up_average_cart_position': 'mean',
                                   '_up_cart_position_std':'std'},
              'order_dow':{'_user_prd_order_mean_day':'mean',
                           '_user_prd_order_std_day':'std'},
              'order_hour_of_day':{'_order_hod_mean':'mean',
                                   '_order_hod_std':'std'},
               'reordered':{'_total_time_of_reorder':'sum'},
               'days_since_prior_order':{'_user_prd_sum_days_since_prior_order':'sum', 
                                        '_user_prd_mean_days_since_prior_order': 'mean',
                                        '_use_prd_std_days_since_prior_order':'std'},
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
data = data.merge(orders[['order_id','order_number']],on='order_id',how='left')


# add user_id to train set
train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')
data['reordered'] = data['reordered'].fillna(0)
data = data.merge(W2C,on='product_id',how='left')
data = data.merge(ostreak,on=['user_id','product_id'],how='left')
print('Data Done 2')
print(time.ctime())

del train, prd, users
del products,us,aisles,departments,priors,ostreak,W2C
del agg_dict,agg_dict_2,agg_dict_4,agg_dict_ais,agg_dict_dept,priors_orders_detail

data = data.merge(user_dept_data,on=['user_id','department_id'],how='left')
del user_dept_data
data = data.merge(user_ais_data,on=['user_id','aisle_id'],how='left')
del user_ais_data
print('Data Done 3')
print(time.ctime())
data = data.merge(dept,on='department_id',how='left').merge(ais,on='aisle_id',how='left')
del dept,ais
data = data.merge(product_name,on='product_id',how='left')
del product_name
print('Data Done 4')
print(time.ctime())
orders = orders[['order_id','order_dow','order_hour_of_day','days_since_prior_order']]
data = data.merge(orders,on='order_id',how='left')
del orders
train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
del data,test
print('Ends')
print(time.ctime())
col = list(train.columns)
col.remove('_usr_prd_reorder_rate')
col.remove('_total_time_of_reorder')
col.remove('department_id')
col.remove('aisle_id')
col.remove('reordered')
col.remove('eval_set')
col.remove('product_id')
col.remove('_prod_tot_cnts')
col.remove('order_number')


from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits=5) 
train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train, groups=train['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)
train_index = train_indexes[0]
test_index = test_indexes[0]

training = train.iloc[train_index,:]
testing = train.iloc[test_index,:]
del train
del train_index,train_indexes,test_index,test_indexes
del i
del agg_dict_dpt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

import lightgbm as lgb
dtrain = lgb.Dataset(training[col],training['reordered'])
validdata = lgb.Dataset(testing[col],testing['reordered'])
del training

lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss','auc'}, ## Add a new metrics
    'num_leaves': 240,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
     }
print('Start training')
model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
print('Start predicting')
pred = model.predict(testing[col],num_iteration=model.best_iteration)
print('Prediction Done')
actual_pred = [1 if each>=0.5 else 0 for each in pred]
print(roc_auc_score(actual_pred,testing['reordered']))
print(log_loss(actual_pred,testing['reordered']))
FScore=Cross_Validation(testing,pred,0.2) 
print(FScore)

"""
Early stopping, best iteration is:
[118]   training's auc: 0.853743        training's binary_logloss: 0.232387     valid_1's auc: 0.838276 valid_1's binary_logloss: 0.241046
Start predicting
Prediction Done
0.778609445404
3.06079976894
0.381573006152

"""

FI = get_feature_importance_lgb(model)
FI.to_csv('FI_92Feats_WithProdOrderInterval.csv',index=False)

evals = []
for leave in [80,120,300,500,1000]:
    lgb_params = {'learning_rate': 0.1,
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': {'binary_logloss','auc'},
                      'num_leaves': leave,
                      'feature_fraction': 0.95,
                      'bagging_fraction': 0.76,
                      'bagging_freq': 5,
                      'max_bin':500}
    print('Start training')
    model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
    print('Start predicting')
    pred = model.predict(testing[col],num_iteration=model.best_iteration)
    print('Prediction Done')
    actual_pred = [1 if each>=0.5 else 0 for each in pred]
    AUC=roc_auc_score(actual_pred,testing['reordered'])
    LL=log_loss(actual_pred,testing['reordered'])
    FScore=Cross_Validation(testing,pred,0.2) 
    report = {'Leave':leave,'Rounds':model.best_iteration,'AUC':AUC,'logloss':LL,'FScore':FScore}
    evals.append(report)
    print(str(report))
    
"""
{'Leave': 80, 'Rounds': 239, 'AUC': 0.7796388353787862, 'logloss': 3.057550581116482, 'FScore': 0.38134887395132655}
{'Leave': 120, 'Rounds': 154, 'AUC': 0.77846336461615018, 'logloss': 3.0597830624205962, 'FScore': 0.38157543541348593}
{'Leave': 300, 'Rounds': 114, 'AUC': 0.77951574195365014, 'logloss': 3.0570721214615224, 'FScore': 0.38155794259999443}
{'Leave': 500, 'Rounds': 90, 'AUC': 0.77989160975205063, 'logloss': 3.0562947326813763, 'FScore': 0.38180392593438822}
{'Leave': 1000, 'Rounds': 65, 'AUC': 0.77938274372798411, 'logloss': 3.0584077318623586, 'FScore': 0.38056527972228726}

"""

FI2 = get_feature_importance_lgb(model)

"""
Try another params set
"""

lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss','auc'}, ## Add a new metrics
    'num_leaves': 400,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
     }
print('Start training')
model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
print('Start predicting')
pred = model.predict(testing[col],num_iteration=model.best_iteration)
print('Prediction Done')
actual_pred = [1 if each>=0.5 else 0 for each in pred]
print(roc_auc_score(actual_pred,testing['reordered']))
print(log_loss(actual_pred,testing['reordered']))
FScore=Cross_Validation(testing,pred,0.2) 
print(FScore)

"""
Start predicting
Prediction Done
0.780016052608
3.05521827402
0.381686404106
"""

lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss','auc'}, ## Add a new metrics
    'num_leaves': 400,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 1,
    'max_bin':500
     }
print('Start training')
model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
print('Start predicting')
pred = model.predict(testing[col],num_iteration=model.best_iteration)
print('Prediction Done')
actual_pred = [1 if each>=0.5 else 0 for each in pred]
print(roc_auc_score(actual_pred,testing['reordered']))
print(log_loss(actual_pred,testing['reordered']))
FScore=Cross_Validation(testing,pred,0.2) 
print(FScore)


"""
Start predicting
Prediction Done
0.779235749607
3.05782959317
0.381491018179
"""

'''
Start training
Train until valid scores didn't improve in 25 rounds.
[30]    training's auc: 0.841763        training's binary_logloss: 0.244285     valid_1's auc: 0.835758 valid_1's binary_logloss: 0.24782
[60]    training's auc: 0.850653        training's binary_logloss: 0.233773     valid_1's auc: 0.837791 valid_1's binary_logloss: 0.241336
[90]    training's auc: 0.859205        training's binary_logloss: 0.229334     valid_1's auc: 0.838078 valid_1's binary_logloss: 0.241111
Early stopping, best iteration is:
[90]    training's auc: 0.859205        training's binary_logloss: 0.229334     valid_1's auc: 0.838078 valid_1's binary_logloss: 0.241111
'''



"""
Try 500
"""

lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss','auc'}, ## Add a new metrics
    'num_leaves': 500,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
     }
print('Start training')
model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
print('Start predicting')
pred = model.predict(testing[col],num_iteration=model.best_iteration)
print('Prediction Done')
actual_pred = [1 if each>=0.5 else 0 for each in pred]
print(roc_auc_score(actual_pred,testing['reordered']))
print(log_loss(actual_pred,testing['reordered']))
FScore=Cross_Validation(testing,pred,0.2) 
print(FScore)

"""
[99]    training's auc: 0.861511        training's binary_logloss: 0.22803      valid_1's auc: 0.838017 valid_1's binary_logloss: 0.241154
Start predicting
Prediction Done
0.77966597462
3.05583618871
0.381210125581
"""