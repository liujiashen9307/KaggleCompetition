# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:02:42 2017

@author: LUI01
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

train = pd.read_csv('input/en_train.csv')
test = pd.read_csv('input/en_test_2.csv')

max_num_features = 10
pad_size = 1
boundary_letter = -1
space_letter = 0
max_data_size = 320000


"""
sentence_id  token_id  class         before          after
0            0         0  PLAIN  Brillantaisia  Brillantaisia
1            0         1  PLAIN             is             is
2            0         2  PLAIN              a              a
3            0         3  PLAIN          genus          genus
4            0         4  PLAIN             of             of

"""

x_data = []
y_data = pd.factorize(train['class'])
labels = y_data[1]
y_data = y_data[0]

for x in train['before'].values:
    x_row = np.ones(max_num_features, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi) ## ord: transfer string to unidoce. ord('a') = 97 ,ord('s') = 115
    x_data.append(x_row)
    



def context_window_transform(data, pad_size):
    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(pad_size)]
    data = pre + data + pre
    neo_data = []
    for i in np.arange(len(data) - pad_size * 2):
        row = []
        for x in data[i : i + pad_size * 2 + 1]:
            row.append([boundary_letter])
            row.append(x)
        row.append([boundary_letter])
        neo_data.append([int(x) for y in row for x in y])
    return neo_data

x_data = x_data[:max_data_size]
y_data = y_data[:max_data_size]
x_data = np.array(context_window_transform(x_data, pad_size))

x_data = np.array(x_data)
y_data = np.array(y_data)

print('Total number of samples:', len(x_data))
print('Use: ', max_data_size)
#x_data = np.array(x_data)
#y_data = np.array(y_data)

print('x_data sample:')
print(x_data[0])
print('y_data sample:')
print(y_data[0])
print('labels:')
print(labels)

x_train = x_data
y_train = y_data

x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)


num_class = len(labels)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

param = {'objective':'multi:softmax',
         'eta':'0.3', 'max_depth':10,
         'silent':1, 'nthread':-1,
         'num_class':num_class,
         'eval_metric':'merror',
         'seed':42}
model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
                  verbose_eval=10)


### Prediction Area

pred_data = []
for x in test['before'].values:
    x_row = np.ones(max_num_features, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi) ## ord: transfer string to unidoce. ord('a') = 97 ,ord('s') = 115
    pred_data.append(x_row)
pred_data = pred_data 

pred_data = np.array(context_window_transform(pred_data, pad_size))

pred_data = np.array(pred_data)

dpred = xgb.DMatrix(pred_data)

pred = model.predict(dpred)

pred = [labels[int(x)] for x in pred]

test['class'] = pred

test.to_csv('input/test2_with_class.csv',index=False)

