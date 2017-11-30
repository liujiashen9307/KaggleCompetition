# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:37:57 2017

@author: jiashen
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 19:24:32 2017

@author: jiashen

@Purpose: 86 Feats + Gap Feats, no train features and no IDs
"""

from functions_in import *
import pandas as pd
import numpy as np
import time

DIR = 'data/'
W2C = pd.read_csv('data/WORD2VEC_Feat_6.csv')
ostreak = pd.read_csv(DIR+'order_streaks.csv')
priors, train, orders, products, aisles, departments, sample_submission = load_data(DIR)
product_name = pd.read_csv(DIR+'product_name_pca_2comp.csv')
product_embedding = pd.read_csv(DIR+'product_embeddings.csv')
col = ['embedding_'+str(each) for each in range(32)]
col = ['product_id', 'product_name', 'aisle_id', 'department_id']+col
product_embedding.columns = col
del product_embedding['product_name']
del product_embedding['aisle_id']
del product_embedding['department_id']
## Only keep first 8
col = ['product_id']+['embedding_'+str(each) for each in range(8)]
product_embedding = product_embedding[col]
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
data = data.merge(product_embedding,on='product_id',how='left')
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

del train_index,train_indexes,test_index,test_indexes,i
del test

import lightgbm as lgb
dtrain = lgb.Dataset(training[col],training['reordered'])
del training
validdata = lgb.Dataset(testing[col],testing['reordered'])

"""
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
"""

## Try kaggler's params

lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
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
FI = get_feature_importance_lgb(model)
FI.to_csv('FI_86_feats+gap+product_name+embedings.csv',index=False)

"""
Early stopping, best iteration is:
[487]   training's binary_logloss: 0.232154     training's auc: 0.85631 valid_1's binary_logloss: 0.242576      valid_1's auc: 0.83739
Start predicting
Prediction Done
0.781280058
3.07688638305
0.384870762977


Early stopping, best iteration is:
[500]   training's binary_logloss: 0.231709     training's auc: 0.857173        valid_1's binary_logloss: 0.242531      valid_1's auc: 0.837453
Start predicting
Prediction Done
0.780690470109
3.08041181435
0.384566710167
"""

evals=[]
for leave in [64,80,100,128,150,180,200,220,280,300,320,350,400,450,512,1024]:
    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': leave,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }
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
{'Leave': 64, 'Rounds': 752, 'AUC': 0.78080690228490135, 'logloss': 3.080717518000216, 'FScore': 0.38409598179038007}
{'Leave': 80, 'Rounds': 1021, 'AUC': 0.78131670784326646, 'logloss': 3.0788427572780805, 'FScore': 0.38442792576740975}
{'Leave': 100, 'Rounds': 816, 'AUC': 0.78136683708059351, 'logloss': 3.0785778415038885, 'FScore': 0.3843988253731756}
{'Leave': 128, 'Rounds': 787, 'AUC': 0.78110174425541667, 'logloss': 3.0780275653012108, 'FScore': 0.38433645728373311}
{'Leave': 150, 'Rounds': 559, 'AUC': 0.78148495690000097, 'logloss': 3.0771309659803681, 'FScore': 0.38446413769498416}
{'Leave': 180, 'Rounds': 511, 'AUC': 0.78163702039787908, 'logloss': 3.0772736450152158, 'FScore': 0.38490546306848639}
{'Leave': 200, 'Rounds': 480, 'AUC': 0.78054491178526286, 'logloss': 3.0810435318585059, 'FScore': 0.3842175360103291}
{'Leave': 220, 'Rounds': 479, 'AUC': 0.78163271492138198, 'logloss': 3.0767234159882952, 'FScore': 0.38448532198846191}
{'Leave': 280, 'Rounds': 459, 'AUC': 0.7805487921856521, 'logloss': 3.0798208021552593, 'FScore': 0.38466144597107838}
{'Leave': 300, 'Rounds': 352, 'AUC': 0.78135483932116379, 'logloss': 3.0779460956852427, 'FScore': 0.38488128087026047}
{'Leave': 320, 'Rounds': 349, 'AUC': 0.78112972219647259, 'logloss': 3.0778237820494181, 'FScore': 0.38407886739431668}
{'Leave': 350, 'Rounds': 339, 'AUC': 0.78100438319520493, 'logloss': 3.0789853528118449, 'FScore': 0.38478011923838867}
{'Leave': 400, 'Rounds': 409, 'AUC': 0.78038967944670556, 'logloss': 3.0804321382296727, 'FScore': 0.38404824176107122}
{'Leave': 450, 'Rounds': 398, 'AUC': 0.78070028686697823, 'logloss': 3.0783331746046052, 'FScore': 0.38404974092502997}
{'Leave': 512, 'Rounds': 361, 'AUC': 0.78042934367369732, 'logloss': 3.0800041898155475, 'FScore': 0.38486043123396885}
{'Leave': 1024, 'Rounds': 283, 'AUC': 0.78013951418389005, 'logloss': 3.0813898979066083, 'FScore': 0.3827314497734568}
"""


evals=[]
for leave in [180,300,512]:
    for weight in [0,10,30]:
        lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': leave,
        'min_sum_hessian_in_leaf': weight,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
        }
        print('Start training')
        model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
        print('Start predicting')
        pred = model.predict(testing[col],num_iteration=model.best_iteration)
        print('Prediction Done')
        actual_pred = [1 if each>=0.5 else 0 for each in pred]
        AUC=roc_auc_score(actual_pred,testing['reordered'])
        LL=log_loss(actual_pred,testing['reordered'])
        FScore=Cross_Validation(testing,pred,0.2) 
        report = {'Leave':leave,'MW':weight,'Rounds':model.best_iteration,'AUC':AUC,'logloss':LL,'FScore':FScore}
        evals.append(report)
        print(str(report))
        
        
"""
{'Leave': 180, 'MW': 0, 'Rounds': 457, 'AUC': 0.78091212303376623, 'logloss': 3.0804933698211574, 'FScore': 0.38440762087507807}
{'Leave': 180, 'MW': 10, 'Rounds': 374, 'AUC': 0.7813032058712136, 'logloss': 3.078761239542843, 'FScore': 0.38452271568238366}
{'Leave': 180, 'MW': 30, 'Rounds': 556, 'AUC': 0.78080227111377787, 'logloss': 3.0798208483875, 'FScore': 0.38506001872018886}
{'Leave': 300, 'MW': 0, 'Rounds': 344, 'AUC': 0.78036678486742794, 'logloss': 3.0812065300601379, 'FScore': 0.38438194178585139}
{'Leave': 300, 'MW': 10, 'Rounds': 358, 'AUC': 0.78074668207712428, 'logloss': 3.0792094731572068, 'FScore': 0.38408975827503927}
{'Leave': 300, 'MW': 30, 'Rounds': 457, 'AUC': 0.78045260548388729, 'logloss': 3.0802894978789435, 'FScore': 0.38465493669131184}
{'Leave': 512, 'MW': 0, 'Rounds': 281, 'AUC': 0.78057617367155507, 'logloss': 3.0801264897704024, 'FScore': 0.38410109255603325}
{'Leave': 512, 'MW': 10, 'Rounds': 340, 'AUC': 0.78059391907369002, 'logloss': 3.0798411892551285, 'FScore': 0.38494805529864556}
{'Leave': 512, 'MW': 30, 'Rounds': 313, 'AUC': 0.78092660059362717, 'logloss': 3.0780071546134637, 'FScore': 0.38437975914667782}
"""


evals=[]
for MW in [35,40,50]:
    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 180,
        'min_sum_hessian_in_leaf': MW,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }
    print('Start training')
    model = lgb.train(lgb_params, dtrain, 2000,verbose_eval=30,early_stopping_rounds = 25,valid_sets=[dtrain, validdata])
    print('Start predicting')
    pred = model.predict(testing[col],num_iteration=model.best_iteration)
    print('Prediction Done')
    actual_pred = [1 if each>=0.5 else 0 for each in pred]
    AUC=roc_auc_score(actual_pred,testing['reordered'])
    LL=log_loss(actual_pred,testing['reordered'])
    FScore=Cross_Validation(testing,pred,0.2) 
    report = {'weight':MW,'Rounds':model.best_iteration,'AUC':AUC,'logloss':LL,'FScore':FScore}
    evals.append(report)
    print(str(report))
    
"""
{'weight': 35, 'Rounds': 645, 'AUC': 0.78132707966796577, 'logloss': 3.076580709119753, 'FScore': 0.38448106673556959}
{'weight': 40, 'Rounds': 431, 'AUC': 0.78101105336917964, 'logloss': 3.0794744460140655, 'FScore': 0.38458847897128062}
{'weight': 50, 'Rounds': 587, 'AUC': 0.78104213321370008, 'logloss': 3.0785370253177264, 'FScore': 0.38443232079503248}
"""