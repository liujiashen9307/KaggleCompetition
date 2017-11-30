# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 00:35:32 2017

@author: jiashen
"""

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
    ,"eta"              : 0.01
    ,"max_depth"        : 10
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
}

watchlist= [(dtrain, "train")]
print('start training')
print(time.ctime())
model = xgb.train(xgb_params,dtrain,num_boost_round=1200,verbose_eval=10, evals=watchlist,early_stopping_rounds=20)
print(time.ctime())
print('Start predicting')
pred = model.predict(dtest)
print('Prediction Done')
print(time.ctime())
test['reordered'] = pred
sub = sub_file(test,0.20,sample_submission)
test_set2 = test[['user_id','product_id','order_id','reordered']]
test_set2.to_csv('save_test_for_exp_azure_86Feats_xgb_depth10.csv',index=False)
sub.to_csv('20170806_Azure_86Feats_0.01_depth10_xgb.csv',index=False)
FI = get_feature_importance(model)
FI.to_csv('Feature_Importance_depth10_xgb.csv',index=False)

"""
product done
Sun Aug  6 00:46:39 2017
User Done
Sun Aug  6 00:47:43 2017
User Ais Dept Done
Sun Aug  6 00:48:48 2017
Data Done 1
Sun Aug  6 00:50:19 2017
Data Done 2
Sun Aug  6 00:51:10 2017
Data Done 3
Sun Aug  6 00:51:44 2017
Data Done
Sun Aug  6 00:52:04 2017
Ends
Sun Aug  6 00:52:04 2017

import xgboost as xgb

dtrain = xgb.DMatrix(train[col],train['reordered'])
dtest = xgb.DMatrix(test[col])
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)

del train
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.01
    ,"max_depth"        : 10
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
}

watchlist= [(dtrain, "train")]
print('start training')
print(time.ctime())
model = xgb.train(xgb_params,dtrain,num_boost_round=1200,verbose_eval=10, evals=watchlist,early_stopping_rounds=20)
print(time.ctime())
print('Start predicting')
pred = model.predict(dtest)
print('Prediction Done')
print(time.ctime())
test['reordered'] = pred
sub = sub_file(test,0.20,sample_submission)
test_set2 = test[['user_id','product_id','order_id','reordered']]
test_set2.to_csv('save_test_for_exp_azure_86Feats_xgb_depth10.csv',index=False)
sub.to_csv('20170806_Azure_86Feats_0.01_depth10_xgb.csv',index=False)
FI = get_feature_importance(model)
FI.to_csv('Feature_Importance_depth10_xgb.csv',index=False)
start training
Sun Aug  6 00:54:54 2017
[0]     train-logloss:0.686019
Will train until train-logloss hasn't improved in 20 rounds.
[10]    train-logloss:0.621924
[20]    train-logloss:0.568691
[30]    train-logloss:0.523986
[40]    train-logloss:0.486121
[50]    train-logloss:0.45385
[60]    train-logloss:0.426174
[70]    train-logloss:0.402337
[80]    train-logloss:0.381737
[90]    train-logloss:0.363888
[100]   train-logloss:0.348375
[110]   train-logloss:0.334871
[120]   train-logloss:0.323113
[130]   train-logloss:0.312855
[140]   train-logloss:0.30389
[150]   train-logloss:0.296061
[160]   train-logloss:0.289208
[170]   train-logloss:0.283212
[180]   train-logloss:0.277968
[190]   train-logloss:0.27338
[200]   train-logloss:0.269357
[210]   train-logloss:0.265838
[220]   train-logloss:0.262753
[230]   train-logloss:0.260052
[240]   train-logloss:0.257686
[250]   train-logloss:0.255612
[260]   train-logloss:0.253791
[270]   train-logloss:0.252191
[280]   train-logloss:0.250791
[290]   train-logloss:0.249564
[300]   train-logloss:0.248479
[310]   train-logloss:0.247516
[320]   train-logloss:0.246669
[330]   train-logloss:0.245922
[340]   train-logloss:0.245262
[350]   train-logloss:0.244674
[360]   train-logloss:0.244146
[370]   train-logloss:0.24368
[380]   train-logloss:0.243264
[390]   train-logloss:0.242888
[400]   train-logloss:0.242552
[410]   train-logloss:0.242245
[420]   train-logloss:0.241969
[430]   train-logloss:0.24171
[440]   train-logloss:0.241474
[450]   train-logloss:0.241257
[460]   train-logloss:0.241055
[470]   train-logloss:0.240859
[480]   train-logloss:0.240684
[490]   train-logloss:0.240514
[500]   train-logloss:0.240356
[510]   train-logloss:0.240205
[520]   train-logloss:0.240062
[530]   train-logloss:0.239922
[540]   train-logloss:0.239789
[550]   train-logloss:0.239655
[560]   train-logloss:0.239534
[570]   train-logloss:0.239415
[580]   train-logloss:0.239294
[590]   train-logloss:0.239179
[600]   train-logloss:0.239063
[610]   train-logloss:0.238953
[620]   train-logloss:0.238843
[630]   train-logloss:0.23874
[640]   train-logloss:0.23864
[650]   train-logloss:0.238537
[660]   train-logloss:0.238434
[670]   train-logloss:0.238337
[680]   train-logloss:0.238238
[690]   train-logloss:0.238143
[700]   train-logloss:0.23805
[710]   train-logloss:0.237943
[720]   train-logloss:0.237849
[730]   train-logloss:0.237755
[740]   train-logloss:0.237666
[750]   train-logloss:0.237577
[760]   train-logloss:0.237479
[770]   train-logloss:0.237392
[780]   train-logloss:0.237312
[790]   train-logloss:0.237223
[800]   train-logloss:0.237135
[810]   train-logloss:0.237047
[820]   train-logloss:0.236968
[830]   train-logloss:0.236878
[840]   train-logloss:0.236789
[850]   train-logloss:0.236699
[860]   train-logloss:0.236606
[870]   train-logloss:0.236519
[880]   train-logloss:0.236437
[890]   train-logloss:0.236356
[900]   train-logloss:0.236274
[910]   train-logloss:0.236196
[920]   train-logloss:0.236111
[930]   train-logloss:0.236024
[940]   train-logloss:0.23593
[950]   train-logloss:0.235855
[960]   train-logloss:0.235771
[970]   train-logloss:0.23569
[980]   train-logloss:0.235605
[990]   train-logloss:0.23552
[1000]  train-logloss:0.235429
[1010]  train-logloss:0.235342
[1020]  train-logloss:0.235273
[1030]  train-logloss:0.235188
[1040]  train-logloss:0.235101
[1050]  train-logloss:0.235022
[1060]  train-logloss:0.234955
[1070]  train-logloss:0.234875
[1080]  train-logloss:0.234803
[1090]  train-logloss:0.234725
[1100]  train-logloss:0.234646
[1110]  train-logloss:0.234574
[1120]  train-logloss:0.234513
[1130]  train-logloss:0.23443
[1140]  train-logloss:0.234358
[1150]  train-logloss:0.23428
[1160]  train-logloss:0.234207
[1170]  train-logloss:0.234136
[1180]  train-logloss:0.234054
[1190]  train-logloss:0.233982
Sun Aug  6 15:56:15 2017
Start predicting
Prediction Done
Sun Aug  6 16:04:55 2017
"""