# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 19:51:33 2017

@author: Jiashen Liu

Purpose: 1. Define different thresholds based on users


"""

from functions_in import *
import pandas as np
import numpy as np

data = pd.read_csv('data_bakup.csv')
data['reordered'] = data['reordered'].fillna(0)

train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
del data
submission = pd.read_csv('data/sample_submission.csv')
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

import xgboost as xgb

col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')

dtrain = xgb.DMatrix(training[col],training['reordered'])
dtest = xgb.DMatrix(testing[col])

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
bst = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=90, evals=watchlist, verbose_eval=10)

pred = bst.predict(dtest)

pred_actual = [1 if each>=0.5 else 0 for each in pred]

testing['prob'] = pred
testing['pred_actual'] = pred_actual

correctly_pred = testing[testing['reordered']==1] ##其实是：被回购的产品

"""
看一下： 测试集与训练集的订单号有没有重复的情况。
同时，可以按订单把平均的回订的概率给计算出来，分为高和底两种情况去分类。
"""

train_id = list(set(train['order_id']))
test_id = list(set(test['order_id']))

##找交叉

interaction = set(train_id).intersection(test_id)

##果然没有交叉，我想多了。

"""
两种尝试：按用户做一个threshold，再按产品做一个threshold。
"""
avg_prob_by_user = correctly_pred.groupby('user_id')['prob'].mean()

avg_prob_by_user = pd.DataFrame({'user_id':list(avg_prob_by_user.index),'prob_usr':avg_prob_by_user.tolist()})
test_user = list(set(test['user_id']))

interaction = set.intersection(set(test_user),set(list(avg_prob_by_user.index)))

### 长度是40158

### 按产品：
avg_prob_by_prd = correctly_pred.groupby('product_id')['prob'].mean()

avg_prob_by_prd = pd.DataFrame({'product_id':list(avg_prob_by_prd.index),'prob_prd':avg_prob_by_prd.tolist()})
mean_prd = np.mean(avg_prob_by_prd['prob_prd']) 
mean_usr = np.mean(avg_prob_by_user['prob_usr'])
test_prd = list(set(test['product_id']))
interaction = set.intersection(set(test_prd),set(list(avg_prob_by_prd.index)))
## interaction长度是28598


##尝试两次CV看看，思路分别是：用产品/用户的threshold，和用他们分类出的threshold。估计后者效果会好一些。
testing2 = testing.merge(avg_prob_by_prd,on='product_id',how='left')
testing2 = testing2.merge(avg_prob_by_user,on='user_id',how='left')
testing2['reordered'] = pred
testing2['prob_prd'] = testing2['prob_prd'].fillna(mean_prd)
validation_file = CV_file2(testing)
validation_file['products_o'] = validation_file['products_o'].apply(lambda x: string_to_list(x))
def cluster_prob(prob):
    return mean_prd
testing3 = testing2.copy()
avg_order_prob = testing3.groupby('order_id')['prob_prd'].mean()
avg_order_prob = pd.DataFrame({'order_id':list(avg_order_prob.index),'prod_prd_avg':avg_order_prob.tolist()})
#testing3['prob_prd'] = testing3['prob_prd'].apply(lambda x:cluster_prob(x))
test_order_size = testing3.groupby('order_id')['order_id'].count()
test_order_size = pd.DataFrame({'order_id':list(test_order_size.index),'order_size':test_order_size.tolist()})
testing3 = testing3.merge(avg_order_prob,on='order_id',how='left')
testing3 = testing3.merge(test_order_size,on='order_id',how='left')
def cluster_prob(order_size,prob):
    if order_size<=20:
        return prob
    else:
        return mean_prd
#testing3['prod_prd_avg'] = testing3.apply(lambda row:cluster_prob(row['order_size'],row['prod_prd_avg']),axis=1)
prob = list(testing3['prod_prd_avg'])
order_size = list(testing3['order_size'])
prob_cal = [cluster_prob(x,y) for x,y in zip(order_size,prob)]
testing3['prod_prd_avg'] = prob_cal
## 尝试：1.按产品具体来分
d = dict()
for row in testing3.itertuples():
    if row.reordered > row.prod_prd_avg:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)
for order in validation_file.order_id:
    if order not in d:
        d[order] = 'None'
sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']

valid = validation_file.merge(sub,on='order_id',how='left')
#valid['products_o'] = valid['products_o'].apply(lambda x: string_to_list(x))
valid['products'] = valid['products'].apply(lambda x: string_to_list(x))
f1_score(valid['products'],valid['products_o'])

"""
记录CV
把order_size大于140的算为平均的，而小于的按产品来： 0.37356179761627967
规则：
def cluster_prob(order_size,prob):
    if order_size<=140:
        return prob
    else:
        return mean_prd
如果是以上规则，140变为100： 0.37389688924373915
以上规则，140变为50： 0.37516115441693698 
以上规则，140变为20： 0.37704423713231394
用order平均的threshold: F1-Score为：0.37342332362834713
用平均的threshold：F1-Score为： 0.37682577899501274
用以下规则，F1-Score为：0.36470919398946311
def cluster_prob(prob):
    if prob<=0.1:
        return 0.1
    elif prob>0.1 and prob<=mean_prd:
        return mean_prd
    elif prob>mean_prd and prob<=0.4:
        return 0.25
    else:
        return 0.3
原有概率，空值填写为均值：0.3171107519595332
概率按以下规则来分：
def cluster_prob(prob):
    if prob<= mean_prd:
        return mean_prd
    elif prob> mean_prd and prob< 0.4:
        return 0.25
    else:
        return 0.30
F1为： 0.36821229278531931
按以下规则来分：
def cluster_prob(prob):
    if prob<=mean_prd:
        return mean_prd
    else:
        return 0.22
F1为：0.37578204202154364
按以下规则来分：
def cluster_prob(prob):
    if prob<mean_prd:
        return 0.18
    else:
        return 0.22
F1为：0.37532775057573492
"""


###尝试提交几组数据

dtest = xgb.DMatrix(test[col])
pred = bst.predict(dtest)
test['reordered'] = pred
test_copy = test.copy()
###1. 用平均数为threshold提交
def cluster_prob(prob):
    return mean_prd
def cluster_prob(order_size,prob):
    if order_size<=20:
        return prob
    else:
        return mean_prd
test = test_copy
test = test.merge(avg_prob_by_prd,on='product_id',how='left')
test = test.merge(avg_prob_by_user,on='user_id',how='left')
test['prob_prd'] = test['prob_prd'].fillna(mean_prd)
avg_order_prob = test.groupby('order_id')['prob_prd'].mean()
avg_order_prob = pd.DataFrame({'order_id':list(avg_order_prob.index),'prod_prd_avg':avg_order_prob.tolist()})
#testing3['prob_prd'] = testing3['prob_prd'].apply(lambda x:cluster_prob(x))
test_order_size = test.groupby('order_id')['order_id'].count()
test_order_size = pd.DataFrame({'order_id':list(test_order_size.index),'order_size':test_order_size.tolist()})
test = test.merge(avg_order_prob,on='order_id',how='left')
test = test.merge(test_order_size,on='order_id',how='left')
test2 = test.copy()
prob = list(test['prod_prd_avg'])
order_size = list(test['order_size'])
prob_cal = [cluster_prob(x,y) for x,y in zip(order_size,prob)]
test['prod_prd_avg'] = prob_cal
d = dict()
for row in test.itertuples():
    if row.reordered > row.prod_prd_avg:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)
for order in submission.order_id:
    if order not in d:
        d[order] = 'None'
sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']


sub.to_csv('submission/original_model_average_order_prob_when_order_size_less_than_20_20170724.csv',index=False)#0.3787663
sub.to_csv('submission/original_model_average_order_prob_as_threshold_20170724.csv',index=False)#0.3756796,
sub.to_csv('submission/original_model_average_prob_as_threshold_20170724.csv',index=False)##0.3785197

