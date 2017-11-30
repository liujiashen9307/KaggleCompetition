# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:39:10 2017

@author: Jiashen Liu

"""

import tensorflow as tf
from functions import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,auc,accuracy_score
import numpy as np

from FeedforwardNNClass import *

nnet = feed_forward_NN(30,0.01,0.005,57,2,2,20,15)
nnet.Cross_validation(train,'target',col,5)

def report_performance(ytrue, ypred):
    """
    Function Purpose: Report the Performance of Classifier
    
    """
    #print(log_loss(ytrue, ypred))
    #print(auc(ytrue, ypred))
    #print(gini_normalized(ytrue, ypred))
    #print(gini(ytrue,ypred))
    print(accuracy_score(ytrue,ypred))

def PrepareTarget(lst):
    return np.array(lst, dtype='int8').reshape(-1, 1)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

col = list(test.columns)
col.remove('id')

nnet = feed_forward_NN(30,0.01,0.005,57,2,2,20,15)
nnet.Cross_validation(train,'target',col,5,plot=False)


"""
Tensorflow Code

"""

x = tf.placeholder(tf.float32, [None, len(col)])

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def NNet2(num_features,n_hidden_1,n_hidden_2,weight):
    w1 = weight_variable([num_features, n_hidden_1])
    b1 = bias_variable([n_hidden_1])
    w2 = weight_variable([n_hidden_1, n_hidden_2])
    b2 = bias_variable([n_hidden_2])
    w3 = weight_variable([n_hidden_2, 1])
    b3 = bias_variable([1])
    # 1st Hidden layer with dropout
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    #h1_dropout = tf.nn.dropout(h1, keep_prob)
    # 2nd Hidden layer with dropout
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    #h2_dropout = tf.nn.dropout(h2, keep_prob)
    output = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)
    reg = weight * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2))+tf.reduce_mean(tf.square(w3)))
    return output,reg

learning_rate =5e-4
threshold = 0.5
n_hidden = 30
## If necessary
n_hidden2 = 15
weight_l1 = 0.01

y,regularization=NNet2(len(col),n_hidden,n_hidden2,weight_l1)
#y,regularization=NNet(n_features,n_hidden,weight_l1)
y_ = tf.placeholder(tf.float32, [None, 1])
### For Multilabel: Cross Entropy Might be different
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))
#tf.reduce_mean(-y_*tf.log(tf.maximum(0.00001, y)) - (1.0 - y_)*tf.log(tf.maximum(0.00001, 1.0-y)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy+regularization)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
step_record = []
accuracy_record = []

for _ in range(300):
    kf = KFold(n_splits=5,shuffle=True)
    train = train.reset_index(drop=True)
    for train_index, test_index in kf.split(train):
        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
        batch_xs = X_train[col]
        batch_ys = PrepareTarget(list(X_train['target']))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if _%5==0:
            pred = sess.run(y, feed_dict={x:X_test[col]})
            pred = [1 if each>=0.5 else 0 for each in pred]
            report_performance(X_test['target'],pred)
       """
        #Accu.append(accu)
        print('Step '+str(_)+': Accuracy: '+str(accu))
        step_record.append(_)
        accuracy_record.append(accu)
    if _==299:
        probs = sess.run(y, feed_dict={x:test_features})
        """