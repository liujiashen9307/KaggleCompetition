# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:47:33 2017

@author: Jiashen

@purpose: Try to get all possible features together and export the dataset
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

col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
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
Tune parameters of leaves
"""
AUC=[];LL=[];FS=[]
for leave in [80,100,150,200,240,300,500,1000,1200,1500]:
    print('START')
    lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': leave,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
     }
    print(leave)
    print('Start training')
    model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 50,valid_sets=[dtrain, validdata])
    print('Start predicting')
    pred = model.predict(testing[col],num_iteration=model.best_iteration)
    print('Prediction Done')
    actual_pred = [1 if each>=0.5 else 0 for each in pred]
    print(roc_auc_score(actual_pred,testing['reordered']))
    print(log_loss(actual_pred,testing['reordered']))
    FScore=Cross_Validation(testing,pred,0.2) 
    print(FScore)
    AUC.append(roc_auc_score(actual_pred,testing['reordered']))
    LL.append(log_loss(actual_pred,testing['reordered']))
    FS.append(FScore)

FI = get_feature_importance_lgb(model)
FI.to_csv('FeatureImport86Feature+ID.csv',index=False)


"""
100
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.251036     valid_1's binary_logloss: 0.25123
[60]    training's binary_logloss: 0.243553     valid_1's binary_logloss: 0.244298
[90]    training's binary_logloss: 0.24217      valid_1's binary_logloss: 0.243794
[120]   training's binary_logloss: 0.241215     valid_1's binary_logloss: 0.243731
[150]   training's binary_logloss: 0.240329     valid_1's binary_logloss: 0.243729
[180]   training's binary_logloss: 0.239559     valid_1's binary_logloss: 0.243754
Early stopping, best iteration is:
[138]   training's binary_logloss: 0.240675     valid_1's binary_logloss: 0.243699
Start predicting
Prediction Done
0.778956271182
3.08998955594
0.382365765734
150
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.250499     valid_1's binary_logloss: 0.250855
[60]    training's binary_logloss: 0.242764     valid_1's binary_logloss: 0.244118
[90]    training's binary_logloss: 0.241012     valid_1's binary_logloss: 0.243751
[120]   training's binary_logloss: 0.239681     valid_1's binary_logloss: 0.243737
[150]   training's binary_logloss: 0.23836      valid_1's binary_logloss: 0.243805
Early stopping, best iteration is:
[110]   training's binary_logloss: 0.240097     valid_1's binary_logloss: 0.243717
Start predicting
Prediction Done
0.779914559264
3.08713669138
0.381860907078
200
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.250075     valid_1's binary_logloss: 0.250593
[60]    training's binary_logloss: 0.242072     valid_1's binary_logloss: 0.243987
[90]    training's binary_logloss: 0.239988     valid_1's binary_logloss: 0.243614
[120]   training's binary_logloss: 0.238279     valid_1's binary_logloss: 0.243685
Early stopping, best iteration is:
[90]    training's binary_logloss: 0.239988     valid_1's binary_logloss: 0.243614
Start predicting
Prediction Done
0.780159049004
3.08630120288
0.381862756599
300
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.249396     valid_1's binary_logloss: 0.250351
[60]    training's binary_logloss: 0.240822     valid_1's binary_logloss: 0.24394
[90]    training's binary_logloss: 0.238071     valid_1's binary_logloss: 0.243706
[120]   training's binary_logloss: 0.235705     valid_1's binary_logloss: 0.243823
Early stopping, best iteration is:
[86]    training's binary_logloss: 0.238418     valid_1's binary_logloss: 0.243693
Start predicting
Prediction Done
0.779897739769
3.08674949027
0.382114494487
500
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.248265     valid_1's binary_logloss: 0.250138
[60]    training's binary_logloss: 0.238424     valid_1's binary_logloss: 0.243912
[90]    training's binary_logloss: 0.234347     valid_1's binary_logloss: 0.243891
Early stopping, best iteration is:
[67]    training's binary_logloss: 0.237426     valid_1's binary_logloss: 0.243829
Start predicting
Prediction Done
0.779732316347
3.08850204114
0.382062577886
1000
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.245695     valid_1's binary_logloss: 0.250027
[60]    training's binary_logloss: 0.232927     valid_1's binary_logloss: 0.244196
[90]    training's binary_logloss: 0.225648     valid_1's binary_logloss: 0.244417
Early stopping, best iteration is:
[66]    training's binary_logloss: 0.231388     valid_1's binary_logloss: 0.244189
Start predicting
Prediction Done
0.779207828934
3.08895028042
0.382786099829

FI = get_feature_importance_lgb(model)
FI.to_csv('FeatureImport86Feature.csv',index=False)

col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')

dtrain = lgb.Dataset(training[col],training['reordered'])

validdata = lgb.Dataset(testing[col],testing['reordered'])

AUC=[];LL=[];FS=[]

AUC
Out[22]: []

LL
Out[23]: []

FS
Out[24]: []

AUC=[];LL=[];FS=[]
for leave in [80,100,150,200,240,300,500,1000,1200,1500]:
    print('START')
    lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': leave,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
     }
    print(leave)
    print('Start training')
    model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 50,valid_sets=[dtrain, validdata])
    print('Start predicting')
    pred = model.predict(testing[col],num_iteration=model.best_iteration)
    print('Prediction Done')
    actual_pred = [1 if each>=0.5 else 0 for each in pred]
    print(roc_auc_score(actual_pred,testing['reordered']))
    print(log_loss(actual_pred,testing['reordered']))
    FScore=Cross_Validation(testing,pred,0.2) 
    print(FScore)
    AUC.append(roc_auc_score(actual_pred,testing['reordered']))
    LL.append(log_loss(actual_pred,testing['reordered']))
    FS.append(FScore)
START
80
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.251315     valid_1's binary_logloss: 0.251418
[60]    training's binary_logloss: 0.243784     valid_1's binary_logloss: 0.244309
[90]    training's binary_logloss: 0.242525     valid_1's binary_logloss: 0.243749
[120]   training's binary_logloss: 0.241657     valid_1's binary_logloss: 0.243625
[150]   training's binary_logloss: 0.240912     valid_1's binary_logloss: 0.24361
[180]   training's binary_logloss: 0.240256     valid_1's binary_logloss: 0.24361
[210]   training's binary_logloss: 0.239552     valid_1's binary_logloss: 0.243648
Early stopping, best iteration is:
[169]   training's binary_logloss: 0.240504     valid_1's binary_logloss: 0.243594
Start predicting
Prediction Done
0.779568646443
3.08821670761
0.382038826085
START
100
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.251041     valid_1's binary_logloss: 0.251212
[60]    training's binary_logloss: 0.243409     valid_1's binary_logloss: 0.244183
[90]    training's binary_logloss: 0.24203      valid_1's binary_logloss: 0.243643
[120]   training's binary_logloss: 0.24107      valid_1's binary_logloss: 0.243582
[150]   training's binary_logloss: 0.24018      valid_1's binary_logloss: 0.243571
[180]   training's binary_logloss: 0.239404     valid_1's binary_logloss: 0.243632
Early stopping, best iteration is:
[130]   training's binary_logloss: 0.240755     valid_1's binary_logloss: 0.243549
Start predicting
Prediction Done
0.779405032361
3.08795175268
0.381881647329
START
150
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.250472     valid_1's binary_logloss: 0.250835
[60]    training's binary_logloss: 0.242671     valid_1's binary_logloss: 0.244033
[90]    training's binary_logloss: 0.24091      valid_1's binary_logloss: 0.243584
[120]   training's binary_logloss: 0.239543     valid_1's binary_logloss: 0.243558
[150]   training's binary_logloss: 0.238244     valid_1's binary_logloss: 0.243633
Early stopping, best iteration is:
[129]   training's binary_logloss: 0.239169     valid_1's binary_logloss: 0.243544
Start predicting
Prediction Done
0.780089454213
3.0857102039
0.38242073752
START
200
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.250049     valid_1's binary_logloss: 0.250575
[60]    training's binary_logloss: 0.241972     valid_1's binary_logloss: 0.243899
[90]    training's binary_logloss: 0.239837     valid_1's binary_logloss: 0.243569
[120]   training's binary_logloss: 0.238156     valid_1's binary_logloss: 0.2436
Early stopping, best iteration is:
[90]    training's binary_logloss: 0.239837     valid_1's binary_logloss: 0.243569
Start predicting
Prediction Done
0.779801233503
3.08707553432
0.382272059625
START
240
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.249747     valid_1's binary_logloss: 0.250461
[60]    training's binary_logloss: 0.241422     valid_1's binary_logloss: 0.243912
[90]    training's binary_logloss: 0.239027     valid_1's binary_logloss: 0.243627
[120]   training's binary_logloss: 0.236985     valid_1's binary_logloss: 0.243738
Early stopping, best iteration is:
[92]    training's binary_logloss: 0.238886     valid_1's binary_logloss: 0.243621
Start predicting
Prediction Done
0.779292928802
3.0889095383
0.38161047905
START
300
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.249387     valid_1's binary_logloss: 0.250332
[60]    training's binary_logloss: 0.240713     valid_1's binary_logloss: 0.243878
[90]    training's binary_logloss: 0.237919     valid_1's binary_logloss: 0.243684
[120]   training's binary_logloss: 0.235496     valid_1's binary_logloss: 0.243819
Early stopping, best iteration is:
[84]    training's binary_logloss: 0.238442     valid_1's binary_logloss: 0.243662
Start predicting
Prediction Done
0.779838139276
3.08642341791
0.381989908033
START
500
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.248218     valid_1's binary_logloss: 0.250154
[60]    training's binary_logloss: 0.2383       valid_1's binary_logloss: 0.243879
[90]    training's binary_logloss: 0.234043     valid_1's binary_logloss: 0.243755
[120]   training's binary_logloss: 0.230265     valid_1's binary_logloss: 0.244048
Early stopping, best iteration is:
[82]    training's binary_logloss: 0.2351       valid_1's binary_logloss: 0.243709
Start predicting
Prediction Done
0.778863192308
3.08860377615
0.381734954913
START
1000
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.245606     valid_1's binary_logloss: 0.25001
[60]    training's binary_logloss: 0.232605     valid_1's binary_logloss: 0.244106
[90]    training's binary_logloss: 0.225186     valid_1's binary_logloss: 0.244282
Early stopping, best iteration is:
[63]    training's binary_logloss: 0.231801     valid_1's binary_logloss: 0.244084
Start predicting
Prediction Done
0.778472464682
3.09155863896
0.381466237387
START
1200
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.244636     valid_1's binary_logloss: 0.250048
[60]    training's binary_logloss: 0.230423     valid_1's binary_logloss: 0.244283
[90]    training's binary_logloss: 0.221845     valid_1's binary_logloss: 0.244574
Early stopping, best iteration is:
[62]    training's binary_logloss: 0.229816     valid_1's binary_logloss: 0.244276
Start predicting
Prediction Done
0.778576988139
3.09129373309
0.380917420495
START
1500
Start training
Train until valid scores didn't improve in 50 rounds.
[30]    training's binary_logloss: 0.243239     valid_1's binary_logloss: 0.250072
[60]    training's binary_logloss: 0.227275     valid_1's binary_logloss: 0.244427
[90]    training's binary_logloss: 0.217126     valid_1's binary_logloss: 0.244874
Early stopping, best iteration is:
[55]    training's binary_logloss: 0.22915      valid_1's binary_logloss: 0.244405
Start predicting
Prediction Done
0.778213711517
3.09255715538
0.380179144424
"""


"""
 
"""
