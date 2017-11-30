# -*- coding: utf-8 -*-
"""
Created on Mon Aug 2 20:03:26 2017

@author: Jiashen Liu

@Purpose: Add more features related to aisle and department, and use lightGBM to test!

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
0731加入部门和aisle的recency特征
"""
priors_orders_detail = priors_orders_detail.merge(prd,on='product_id',how='left')

agg_dict_dept = {'user_id':{'_user_dept_total_orders':'count'},
              'reordered':{'_user_dept_total_reorders':'sum'},
              'days_since_prior_order':{'_user_dept_sum_days_since_prior_order':'sum', 
                                        '_user_dept_mean_days_since_prior_order': 'mean',
                                        '_user_dept_std_days_since_prior_order':'std',
                                        '_user_dept_median_days_since_prior_order':'median'}
        }
agg_dict_ais = {'user_id':{'_user_ais_total_orders':'count'},
              'reordered':{'_user_ais_total_reorders':'sum'},
              'days_since_prior_order':{'_user_ais_sum_days_since_prior_order':'sum', 
                                        '_user_ais_mean_days_since_prior_order': 'mean',
                                        '_use_aisr_std_days_since_prior_order':'std',
                                        '_user_ais_median_days_since_prior_order':'median'}
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
#data.to_csv('data_bakeup.csv',index=False)
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

train_set = data[data['eval_set']=='train']

del data

"""
任务1： 用一个单决策树来进行拟合，对参数进行调整。
"""

from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits=5) 
train_indexes = []
test_indexes = []
for i, (train_index, test_index) in enumerate(kf.split(train_set, groups=train_set['user_id'].values)):
    train_indexes.append(train_index)
    test_indexes.append(test_index)
train_index = train_indexes[0]
test_index = test_indexes[0]

training = train_set.iloc[train_index,:]
testing = train_set.iloc[test_index,:]

del train_set
"""
Decision Tree

"""
col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')
col.remove('department_id')
col.remove('aisle_id')


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

"""
for depth in [5,7,10,12]:
    clf = DecisionTreeClassifier(random_state=42,max_depth=depth,max_features=0.95)
    clf.fit(training[col],training['reordered'])
    pred = clf.predict(testing[col])
    print(roc_auc_score(testing['reordered'],pred))
  
"""
"""
0.576273278796
0.577584543037
0.580902983141
0.583134107477

"""

"""
Try lgb directly.

Set: learning rate 0.1, iteration 100, focus: complexity of model.
"""
import lightgbm as lgb
del training,testing
training = train_set.iloc[train_index,:]
testing = train_set.iloc[test_index,:]
dtrain = lgb.Dataset(training[col],training['reordered'])
del training
validdata = lgb.Dataset(testing[col],testing['reordered'])
"""
Try different setting
"""
evals = []
def run_lgb_test(para,ROUNDS=90):
    print('Start training')
    model = lgb.train(para, dtrain, ROUNDS,verbose_eval=4,early_stopping_rounds = 50,valid_sets=[dtrain, validdata])
    print('Start predicting')
    pred = model.predict(testing[col])
    print('Prediction Done')
    actual_pred = [1 if each>=0.5 else 0 for each in pred]
    aucs = roc_auc_score(actual_pred,testing['reordered'])
    loss = log_loss(actual_pred,testing['reordered'])
    FScore = Cross_Validation(testing,pred,0.20)
    evals.append({'auc':aucs,'loss':loss,'FScore':FScore})
    print('AUC: '+str(aucs))
    print('LOSS: '+str(loss))
    print('FScore: '+str(FScore))
    del model
    return evals

"""
original parameters
"""
lgb_params = {
    'learning_rate': 0.1,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 140,
    'max_depth': 6,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
    
}

evals = run_lgb_test(lgb_params)

"""
调整叶子数
"""

for leaf in [1000,500,240,80]:
    print('New param eval')
    lgb_params = {
    'learning_rate': 0.1,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': leaf,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
    }
    evals = run_lgb_test(lgb_params)
    
"""
记录叶子数影响：顺序： 140，1000，500，240，80

Start training
Start predicting
Prediction Done
AUC: 0.779225437614
LOSS: 3.10085153199
FScore: 0.380436868774

New param eval
Start training
Start predicting
Prediction Done
AUC: 0.777794833035
LOSS: 3.09015236875
FScore: 0.380313814677

New param eval
Start training
Start predicting
Prediction Done
AUC: 0.778772245481
LOSS: 3.08856300147
FScore: 0.380959065108

New param eval
Start training
Start predicting
Prediction Done
AUC: 0.779378816507
LOSS: 3.08764606506
FScore: 0.381753978543

New param eval
Start training
Start predicting
Prediction Done
AUC: 0.77944503829
LOSS: 3.09137540698
FScore: 0.381621650232
"""

"""
调整 subsample
"""

for sub in [0.65,0.75,0.85,0.9]:
    print('New param eval')
    lgb_params = {
    'learning_rate': 0.1,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 240,
    'subsample':sub,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
    }
    evals = run_lgb_test(lgb_params)
    
"""
subsample和maxbin没有起作用。尝试参数： scale_pos_weight
效果比较差。 加上了cv和eval，尝试cv一下。
"""
lgb_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 240,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.76,
    'bagging_freq': 5,
    'max_bin':500
 }

"""
0.01/1840 training's binary_logloss: 0.23351      valid_1's binary_logloss: 0.243178 0.38301108974790044/3.08206254955/0.780965752319

0.03/[435]   training's binary_logloss: 0.236483     valid_1's binary_logloss: 0.24331/435
0.38280145016338507/0.780415587538/3.08465056297

"""

print('Start training')
model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=4,early_stopping_rounds = 50,valid_sets=[dtrain, validdata])
print('Start predicting')
pred = model.predict(testing[col])
print('Prediction Done')
actual_pred = [1 if each>=0.5 else 0 for each in pred]
print(roc_auc_score(actual_pred,testing['reordered']))
print(log_loss(actual_pred,testing['reordered']))
Cross_Validation(testing,pred,0.2) 
