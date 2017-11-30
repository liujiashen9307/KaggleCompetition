# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:32:32 2017

@author: Jiashen Liu

Purpose: Try More Method on Features (exclude ID Feature)
"""

import pandas as pd
import numpy as np
from functions_in import *

data = pd.read_csv('data_bakup.csv')
data['reordered'] = data['reordered'].fillna(0)

train = data[data['eval_set']=='train']
test = data[data['eval_set']=='test']
del data


"""
尝试使用一开始Kernel里的特征，加入Word2Vec特征，训练50%的数据并且CV看效果。
"""
import gensim
from sklearn.decomposition import PCA

train_orders = pd.read_csv("data/order_products__train.csv")
prior_orders = pd.read_csv("data/order_products__prior.csv")
products = pd.read_csv("data/products.csv").set_index('product_id')

train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

sentences = prior_products.append(train_products).values

model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=2, workers=4)

vocab = list(model.wv.vocab.keys())
matrix = model.wv.syn0

pca = PCA(n_components=6,random_state=42)
W2V_features = pca.fit_transform(matrix)
W2V_features = pd.DataFrame(W2V_features)
W2V_features.columns = ['W2V_1','W2V_2','W2V_3','W2V_4','W2V_5','W2V_6']
W2V_features['product_id'] = vocab
W2V_features['product_id'] = W2V_features['product_id'].astype(np.uint16)
W2V_features.to_csv('data/WORD2VEC_Feat_6.csv',index=False)

del train_orders,train_products,prior_orders,prior_products,sentences,matrix
del products,vocab

train = train.merge(W2V_features,on='product_id',how='left')
test = test.merge(W2V_features,on='product_id',how='left')

product = pd.read_csv('data/products.csv')
from sklearn.feature_extraction.text import TfidfVectorizer
import re
def clean_data(Desc):
    content = re.sub('<[^>]*>', '', Desc)
    letters_only = re.sub("[^a-zA-Z-0-9]", " ", content)
    letters_only = re.sub('-',' ',letters_only)
    words = letters_only.lower().split()
    return(" ".join(words))
product['product_name'] = product['product_name'].apply(lambda x: clean_data(x))
tfidf = TfidfVectorizer()
name_feature = tfidf.fit_transform(list(product['product_name']))
pca = PCA(n_components=2,random_state=42)
name_feature = pca.fit_transform(name_feature.toarray())
name_feature = pd.DataFrame(name_feature)
name_feature.columns = ['pca_name1','pca_name2']
name_feature['product_id'] = product['product_id']
name_feature.to_csv('data/product_name_pca_2comp.csv',index=False)

train = train.merge(name_feature,on='product_id',how='left')
test = test.merge(name_feature,on='product_id',how='left')

###尝试：不进行stratified的情况下可以走多远,采用10%的数据

from sklearn.model_selection import train_test_split

training,testing = train_test_split(train,test_size=0.9,random_state=42)

import xgboost as xgb

col = list(training.columns)
col.remove('reordered')
col.remove('eval_set')
col.remove('user_id')
col.remove('product_id')
col.remove('order_id')

###不带PCA特征，尝试：有了W2C特征是否会有帮助
col.remove('pca_name1')
col.remove('pca_name2')

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

from sklearn.metrics import log_loss
log_loss(pred_actual,testing['reordered'])

### 3.1140985924529501

##CV###

validation_file = CV_file2(testing)
actual_label = testing['reordered']

testing['reordered'] = pred
test_file = sub_file(testing,0.21,validation_file)

validation_file['products_o'] = validation_file['products_o'].apply(lambda x: string_to_list(x))
test_file['products'] = test_file['products'].apply(lambda x: string_to_list(x))
valid = validation_file.merge(test_file,on='order_id',how='left') 
f1_score(valid['products'],valid['products_o'])

FI = get_feature_importance(bst)

"""
 Feature  Score
6                 _user_reorder_ratio    425
10              time_since_last_order    372
1          _up_order_since_last_order    369
4                  _prod_reorder_prob    351
0                      _up_order_rate    300
12  _user_mean_days_since_prior_order    297
5    _up_order_rate_since_first_order    284
3                 _prod_reorder_ratio    281
2                     _up_order_count    212
13               _user_average_basket    201
8    _user_sum_days_since_prior_order    185
19     _prod_buy_first_time_total_cnt    184
15          _up_average_cart_position    172
9             _user_distinct_products    161
17                              W2V_1    149
23                              W2V_2    135
7               _up_last_order_number    132
14                     _prod_tot_cnts    125
16             _prod_reorder_tot_cnts    122
18               _user_total_products     91
20             _up_first_order_number     90
11    _prod_buy_second_time_total_cnt     88
21                 _user_total_orders     49
22                _prod_reorder_times     24

[0]     train-logloss:0.625628
[10]    train-logloss:0.335016
[20]    train-logloss:0.268004
[30]    train-logloss:0.250511
[40]    train-logloss:0.24571
[50]    train-logloss:0.244086
[60]    train-logloss:0.243313
[70]    train-logloss:0.242803
[80]    train-logloss:0.242377
3.1140985924529501
0.37711373727954334
"""


"""

将不放入W2V特征的数据集放入模型。

"""

col.remove('W2V_1')
col.remove('W2V_2')

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

from sklearn.metrics import log_loss
log_loss(pred_actual,actual_label)

FI = get_feature_importance(bst)
testing['reordered'] = actual_label
F_Score = Cross_Validation(testing,pred,0.21)


"""
[0]     train-logloss:0.625588
[10]    train-logloss:0.335078
[20]    train-logloss:0.268026
[30]    train-logloss:0.250489
[40]    train-logloss:0.245716
[50]    train-logloss:0.244146
[60]    train-logloss:0.24344
[70]    train-logloss:0.242884
[80]    train-logloss:0.242443
3.1139491129843964
0.37668700963777585
                              Feature  Score
15                _user_reorder_ratio    435
1          _up_order_since_last_order    413
4               time_since_last_order    382
5                  _prod_reorder_prob    371
0                      _up_order_rate    313
16  _user_mean_days_since_prior_order    308
3                 _prod_reorder_ratio    297
6    _up_order_rate_since_first_order    278
19     _prod_buy_first_time_total_cnt    227
2                     _up_order_count    223
21               _user_average_basket    215
9    _user_sum_days_since_prior_order    203
12          _up_average_cart_position    192
10            _user_distinct_products    163
7               _up_last_order_number    123
18    _prod_buy_second_time_total_cnt    122
11                     _prod_tot_cnts    111
14             _prod_reorder_tot_cnts    105
20             _up_first_order_number     96
17               _user_total_products     87
8                  _user_total_orders     58
13                _prod_reorder_times     37
"""


## 最后尝试放入PCA特征

testing['reordered'] = actual_label
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

from sklearn.metrics import log_loss
log_loss(pred_actual,actual_label)

FI = get_feature_importance(bst)
#testing['reordered'] = actual_label
F_Score = Cross_Validation(testing,pred,0.21)


"""
[0]     train-logloss:0.625584
[10]    train-logloss:0.33531
[20]    train-logloss:0.26797
[30]    train-logloss:0.250474
[40]    train-logloss:0.245683
[50]    train-logloss:0.24411
[60]    train-logloss:0.243357
[70]    train-logloss:0.242786
[80]    train-logloss:0.242394
3.1161906414068188
0.37722919384505416


                              Feature  Score
7                 _user_reorder_ratio    394
4               time_since_last_order    356
1          _up_order_since_last_order    352
5                  _prod_reorder_prob    340
0                      _up_order_rate    287
6    _up_order_rate_since_first_order    282
3                 _prod_reorder_ratio    281
18  _user_mean_days_since_prior_order    233
2                     _up_order_count    226
16                          pca_name2    203
23               _user_average_basket    190
10     _prod_buy_first_time_total_cnt    174
9    _user_sum_days_since_prior_order    171
24                          pca_name1    170
17          _up_average_cart_position    158
22            _user_distinct_products    157
25                              W2V_2    141
19                              W2V_1    118
8               _up_last_order_number    109
12                     _prod_tot_cnts    103
20               _user_total_products     90
11    _prod_buy_second_time_total_cnt     83
21             _up_first_order_number     76
13             _prod_reorder_tot_cnts     74
14                 _user_total_orders     52
15                _prod_reorder_times     28
"""


dtest = xgb.DMatrix(test[col])
pred = bst.predict(dtest)
test['reordered'] = pred
sample_submission = pd.read_csv('data/sample_submission.csv')
sub = sub_file(test,0.21,sample_submission)
sub.to_csv('submission/xgb_pca2names_W2C_10%data.csv',index=False)


