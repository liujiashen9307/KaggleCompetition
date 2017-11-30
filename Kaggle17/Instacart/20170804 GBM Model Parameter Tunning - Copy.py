# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:15:39 2017

@author: jiashen

@Purpose: Tune lgb model, find a better parameter combination
"""

from functions_in import *
import pandas as pd
import numpy as np
import time

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
                                        '_user_prd_median_days_since_prior_order':'median'}}

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
print('Data Done 3')
print(time.ctime())
#data = data.merge(dept,on='department_id',how='left').merge(ais,on='aisle_id',how='left')
#del dept,ais

train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
del data
print('Data Done')
print(time.ctime())
#train.to_csv('train_all.csv',index=False)
#test.to_csv('test_all.csv',index=False)
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
del test
### Train a lgb model here

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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

import lightgbm as lgb
dtrain = lgb.Dataset(training[col],training['reordered'])
del training

validdata = lgb.Dataset(testing[col],testing['reordered'])

"""
Tune parameters of leaves, and min_data_leaf
"""
evals = []
for leave in [80,100,120]:
    for min_data in [0,20,50]:
        lgb_params = {'learning_rate': 0.1,
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': {'binary_logloss'},
                      'num_leaves': leave,
                      'feature_fraction': 0.95,
                      'bagging_fraction': 0.76,
                      'bagging_freq': 5,
                      'max_bin':500,
                      'min_data_in_leaf':min_data}
        print('Start training')
        model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
        print('Start predicting')
        pred = model.predict(testing[col],num_iteration=model.best_iteration)
        print('Prediction Done')
        actual_pred = [1 if each>=0.5 else 0 for each in pred]
        AUC=roc_auc_score(actual_pred,testing['reordered'])
        LL=log_loss(actual_pred,testing['reordered'])
        FScore=Cross_Validation(testing,pred,0.2) 
        report = {'Leave':leave,'min_data':min_data,'Rounds':model.best_iteration,'AUC':AUC,'logloss':LL,'FScore':FScore}
        evals.append(report)
        print(str(report))


## Search generalization for leave = 1000 and leave =120

evals = []
for leave in [1000,120]:
    for l1 in [0.01,0.5,1,5,10]:
        lgb_params = {'learning_rate': 0.1,
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': {'binary_logloss'},
                      'num_leaves': leave,
                      'feature_fraction': 0.95,
                      'bagging_fraction': 0.76,
                      'bagging_freq': 5,
                      'max_bin':500,
                      'lambda_l1':l1}
        print('Start training')
        model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
        print('Start predicting')
        pred = model.predict(testing[col],num_iteration=model.best_iteration)
        print('Prediction Done')
        actual_pred = [1 if each>=0.5 else 0 for each in pred]
        AUC=roc_auc_score(actual_pred,testing['reordered'])
        LL=log_loss(actual_pred,testing['reordered'])
        FScore=Cross_Validation(testing,pred,0.2) 
        report = {'Leave':leave,'lambda':l1,'Rounds':model.best_iteration,'AUC':AUC,'logloss':LL,'FScore':FScore}
        evals.append(report)
        print(str(report))


"""
leave =1000:
{'Leave': 1000, 'min_data': 20, 'Rounds': 66, 'AUC': 0.77920782893411689, 'logloss': 3.0889502804191014, 'FScore': 0.38278609982883094}
{'Leave': 1000, 'min_data': 100, 'Rounds': 62, 'AUC': 0.77877732975079772, 'logloss': 3.089765355400099, 'FScore': 0.38193358819613948}
{'Leave': 1000, 'min_data': 300, 'Rounds': 77, 'AUC': 0.77876477042335202, 'logloss': 3.0890113351015049, 'FScore': 0.3811734670097352}
{'Leave': 1000, 'min_data': 500, 'Rounds': 81, 'AUC': 0.77842371364821727, 'logloss': 3.0897245316658171, 'FScore': 0.38136293371419905}

Different Leaves:
{'Leave': 1000, 'min_data': 0, 'Rounds': 46, 'AUC': 0.77732295326557943, 'logloss': 3.099363534105203, 'FScore': 0.37902319481017227}
{'Leave': 1000, 'min_data': 20, 'Rounds': 66, 'AUC': 0.77920782893411689, 'logloss': 3.0889502804191014, 'FScore': 0.38278609982883094}
{'Leave': 1000, 'min_data': 50, 'Rounds': 70, 'AUC': 0.77793335440549272, 'logloss': 3.0926793760793796, 'FScore': 0.38146920440119481}
{'Leave': 1200, 'min_data': 0, 'Rounds': 48, 'AUC': 0.77544171661781547, 'logloss': 3.1025422832709304, 'FScore': 0.3786426141549033}
{'Leave': 1200, 'min_data': 20, 'Rounds': 64, 'AUC': 0.77767450106644609, 'logloss': 3.0938001655657037, 'FScore': 0.38071475737701083}
{'Leave': 1200, 'min_data': 50, 'Rounds': 64, 'AUC': 0.77769516276818385, 'logloss': 3.0938613061079812, 'FScore': 0.38052560066678032}
{'Leave': 1500, 'min_data': 0, 'Rounds': 44, 'AUC': 0.77567723821612766, 'logloss': 3.1030721808653747, 'FScore': 0.37803956750206635}
{'Leave': 1500, 'min_data': 20, 'Rounds': 55, 'AUC': 0.7775082834574375, 'logloss': 3.0949821092752714, 'FScore': 0.37976249172981241}
{'Leave': 1500, 'min_data': 50, 'Rounds': 56, 'AUC': 0.77756107715268208, 'logloss': 3.0943096161470676, 'FScore': 0.38119426883757801}

{'Leave': 80, 'min_data': 0, 'Rounds': 74, 'AUC': 0.77828560985255546, 'logloss': 3.09602157633398, 'FScore': 0.38093361188515007}
{'Leave': 80, 'min_data': 20, 'Rounds': 149, 'AUC': 0.77852305517307907, 'logloss': 3.0915994060815173, 'FScore': 0.38176364339441893}
{'Leave': 80, 'min_data': 50, 'Rounds': 173, 'AUC': 0.77997206503595984, 'logloss': 3.0870144286892329, 'FScore': 0.38174354493411594}
{'Leave': 100, 'min_data': 0, 'Rounds': 75, 'AUC': 0.77794868597640054, 'logloss': 3.0953082538104031, 'FScore': 0.38154285093342993}
{'Leave': 100, 'min_data': 20, 'Rounds': 138, 'AUC': 0.77895627118218491, 'logloss': 3.0899895559442463, 'FScore': 0.38236576573373188}
{'Leave': 100, 'min_data': 50, 'Rounds': 128, 'AUC': 0.77914453701564268, 'logloss': 3.0890521630816044, 'FScore': 0.38193490404609037}
{'Leave': 120, 'min_data': 0, 'Rounds': 63, 'AUC': 0.77919889293117228, 'logloss': 3.0926388514393817, 'FScore': 0.3814652198015871}
{'Leave': 120, 'min_data': 20, 'Rounds': 130, 'AUC': 0.77876150109758746, 'logloss': 3.0903767183646953, 'FScore': 0.38267430750484172}
{'Leave': 120, 'min_data': 50, 'Rounds': 129, 'AUC': 0.78026859099401946, 'logloss': 3.0856083419899378, 'FScore': 0.38233779189551331}

{'Leave': 1000, 'lambda': 0.01, 'Rounds': 64, 'AUC': 0.77889530280400199, 'logloss': 3.0901729544549581, 'FScore': 0.38090865165565924}
{'Leave': 1000, 'lambda': 0.5, 'Rounds': 64, 'AUC': 0.77846141858428497, 'logloss': 3.0914975003028982, 'FScore': 0.38151008541916243}
{'Leave': 1000, 'lambda': 1, 'Rounds': 64, 'AUC': 0.77897710548473342, 'logloss': 3.0892966785467157, 'FScore': 0.38124997349440193}
{'Leave': 1000, 'lambda': 5, 'Rounds': 80, 'AUC': 0.7766087811871818, 'logloss': 3.0960416393364145, 'FScore': 0.38134966996291564}
{'Leave': 1000, 'lambda': 10, 'Rounds': 84, 'AUC': 0.77843461749678267, 'logloss': 3.0888686211565997, 'FScore': 0.38136143411705264}
{'Leave': 120, 'lambda': 0.01, 'Rounds': 138, 'AUC': 0.77917987694281066, 'logloss': 3.0884815615792967, 'FScore': 0.38291119701049153}
{'Leave': 120, 'lambda': 0.5, 'Rounds': 133, 'AUC': 0.77900204163474218, 'logloss': 3.0897653969147623, 'FScore': 0.38188342547714876}
{'Leave': 120, 'lambda': 1, 'Rounds': 135, 'AUC': 0.77974308360817135, 'logloss': 3.0868105987334413, 'FScore': 0.38180197677036348}
## Test 5 and 10 if have time.
"""
lgb_params = {'learning_rate': 0.01,
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': {'binary_logloss'},
                      'num_leaves': 120,
                      'feature_fraction': 0.95,
                      'bagging_fraction': 0.76,
                      'bagging_freq': 5,
                      'max_bin':500,
                      'lambda_l1':0.01}
print('Start training')
model = lgb.train(lgb_params, dtrain, 3000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
print('Start predicting')
pred = model.predict(testing[col],num_iteration=model.best_iteration)
print('Prediction Done')
actual_pred = [1 if each>=0.5 else 0 for each in pred]
AUC=roc_auc_score(actual_pred,testing['reordered'])
LL=log_loss(actual_pred,testing['reordered'])
FScore=Cross_Validation(testing,pred,0.2) 
report = {'Leave':120,'lambda':0.01,'LR':0.01,'Rounds':model.best_iteration,'AUC':AUC,'logloss':LL,'FScore':FScore}
evals.append(report)
print(str(report))

"""
Start predicting
Prediction Done
{'Leave': 120, 'lambda': 0.01, 'LR': 0.01, 'Rounds': 1455, 'AUC': 0.78062988188029281, 'logloss': 3.0845079495103924, 'FScore': 0.38260751334126725}
"""

FI = get_feature_importance_lgb(model)

