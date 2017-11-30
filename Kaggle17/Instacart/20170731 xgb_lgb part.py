# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:08:27 2017

@author: Jiashen Liu

@Purpose: 1. Try depth more in xgb and fit-controlling params
          2. Set up lgb for learn
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
prd = prd.merge(products,on='product_id',how='left')
prd = prd.merge(departments,on='department_id',how='left')
prd = prd.merge(aisles,on='aisle_id',how='left')
del prd['department']
del prd['aisle']
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
0730： 加四个关于department和aisle的特征
从数据上看，department和aisle是类似于商品类别的数据，可能会非常有用。因此想知道：
这个department/aisle是否是用户最喜欢的？
这个department/aisle是否是用户最常订购的？
如果好用，可以再加最近订购的参数，类似一样的。
在最后的数据部分，由user_id和department_id/aisle_id一起join，也可以把这些作为一个特征输入进去。看看内存。
"""
priors_orders_detail = priors_orders_detail.merge(prd,on='product_id',how='left')

agg_dict_dept = {'user_id':{'_user_dept_total_orders':'count'},
              'reordered':{'_user_dept_total_reorders':'sum'}
        }
agg_dict_ais = {'user_id':{'_user_ais_total_orders':'count'},
              'reordered':{'_user_ais_total_reorders':'sum'}
        }

user_dept_data =  ka_add_groupby_features_1_vs_n(priors_orders_detail, 
                                                      group_columns_list=['user_id', 'department_id'], 
                                                      agg_dict=agg_dict_dept)
user_ais_data = ka_add_groupby_features_1_vs_n(priors_orders_detail, 
                                                      group_columns_list=['user_id', 'aisle_id'], 
                                                      agg_dict=agg_dict_ais)
user_dept_data['_user_dept_reorder_rate'] = user_dept_data['_user_dept_total_reorders']/user_dept_data['_user_dept_total_orders']
user_ais_data['_user_ais_reorder_rate'] = user_ais_data['_user_ais_total_reorders']/user_ais_data['_user_ais_total_orders']

"""
这几个特征实在没啥用，扔掉
fa_usr_dept = user_dept_data.groupby(['user_id'])['_user_dept_total_orders'].max().reset_index()
fa_usr_dept = fa_usr_dept.merge(user_dept_data[['user_id','_user_dept_total_orders','department_id']],on=['user_id','_user_dept_total_orders'],how='left')
fa_usr_dept['_is_favoriate_dept'] = [1]*len(fa_usr_dept)
del fa_usr_dept['_user_dept_total_orders']

fa_usr_ais = user_ais_data.groupby(['user_id'])['_user_ais_total_orders'].max().reset_index()
fa_usr_ais = fa_usr_ais.merge(user_ais_data[['user_id','_user_ais_total_orders','aisle_id']],on=['user_id','_user_ais_total_orders'],how='left')
fa_usr_ais['_is_favoriate_ais'] = [1]*len(fa_usr_ais)
del fa_usr_ais['_user_ais_total_orders']

re_usr_dept = user_dept_data.groupby(['user_id'])['_user_dept_reorder_rate'].max().reset_index()
re_usr_dept = re_usr_dept.merge(user_dept_data[['user_id','_user_dept_reorder_rate','department_id']],on=['user_id','_user_dept_reorder_rate'],how='left')
re_usr_dept['_is_freq_reorder_dept'] = [1]*len(re_usr_dept)
del re_usr_dept['_user_dept_reorder_rate']

re_usr_ais = user_ais_data.groupby(['user_id'])['_user_ais_reorder_rate'].max().reset_index()
re_usr_ais = re_usr_ais.merge(user_ais_data[['user_id','_user_ais_reorder_rate','aisle_id']],on=['user_id','_user_ais_reorder_rate'],how='left')
re_usr_ais['_is_freq_reorder_ais'] = [1]*len(re_usr_ais)
del re_usr_ais['_user_ais_reorder_rate']
"""

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
del agg_dict,agg_dict_2,agg_dict_4,agg_dict_ais,agg_dict_dept

data = data.merge(user_dept_data,on=['user_id','department_id'],how='left')
del user_dept_data
data = data.merge(user_ais_data,on=['user_id','aisle_id'],how='left')
del user_ais_data
"""
不再需要这几个特征了
data = data.merge(fa_usr_dept,on=['user_id','department_id'],how='left')
data = data.merge(re_usr_dept,on=['user_id','department_id'],how='left')
data = data.merge(fa_usr_ais,on=['user_id','aisle_id'],how='left')
data = data.merge(re_usr_ais,on=['user_id','aisle_id'],how='left')

del fa_usr_ais,fa_usr_dept,re_usr_ais,re_usr_dept
"""
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
del train_set
del train_index,train_indexes,test_index,test_indexes,i
from sklearn.metrics import auc
import xgboost as xgb

col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')
col.remove('department_id')
col.remove('aisle_id')

dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(testing[col])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.10
    ,"max_depth"        : 7
    ,"subsample"        :0.76  #0.76
    ,"colsample_bytree" :0.95
    
}

watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=90,early_stopping_rounds=10,evals=watchlist, verbose_eval=10)

pred = model.predict(dtest)

FScore = Cross_Validation(testing,pred,0.20)
FScore1 = FScore
### Depth 7: 0.37798866537360215, depth 7, subsample 0.5:0.37693398093832503, subsample 1: 
###Boost Rounds Mistakenly set to 300, FScore: 0.36945362708381335, 90: depth 10 thres 0.20: 0.37502356178244522, depth9: 0.37639169237499637, depth 8: 0.37767690225639194

"""
记录： xgb在0730的特征下：把深度设为7，可以有比6更好的效果。

"""

## 把order_id放进去学习？？
col.append('order_id')
dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(testing[col])
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.10
    ,"max_depth"        : 7
    ,"subsample"        :0.76  #0.76
    ,"colsample_bytree" :0.95
    
}

watchlist= [(dtrain, "train")]
model = xgb.train(xgb_params,dtrain,num_boost_round=90,early_stopping_rounds=10,evals=watchlist, verbose_eval=10)

pred = model.predict(dtest)

FScore = Cross_Validation(testing,pred,0.20)

### FScore: 0.37754857304812017

"""
尝试lgb模型，看结果会是怎么样
"""
del model
del dtrain 
del dtest
del xgb_params
del watchlist
del FScore

"""
lightgbm starts from here
"""
import lightgbm as lgb

dtrain = lgb.Dataset(training[col],label=training['reordered'])
lgb_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 100,
    'max_depth': 7,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5
}
ROUNDS = 90

model = lgb.train(lgb_params, dtrain, ROUNDS)

preds = model.predict(testing[col])

FScore2 = Cross_Validation(testing,preds,0.20)

Average = 0.75*pred + 0.25*preds

FScore4 = Cross_Validation(testing,Average,0.20)



"""
Try logistic Regression
"""

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(C = 0.1,random_state=42)
del training['eval_set']
del testing['eval_set']
training=training.fillna(0)
testing = testing.fillna(0)
lg.fit(training[col],training['reordered'])
pred3 = lg.predict_proba(testing[col])
pred3 = [each[1] for each in pred3]

FScore5 = Cross_Validation(testing,pred3,0.20)

Average2 = 0.73*pred + 0.20*preds + 0.07*np.array(pred3)
FScore6 = Cross_Validation(testing,Average2,0.20)


## FScore: 0.37668024730521699/blending: 0.37818238233006812

"""
0.37695513401348385
lgb_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 7,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5
}
ROUNDS = 100
"""