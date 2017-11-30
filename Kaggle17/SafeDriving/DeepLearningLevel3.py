# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:53:15 2017

@author: Jiashen Liu
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import KFold

train = pd.read_csv('level2_stack_train.csv')
test = pd.read_csv('level2_stack_test.csv')

def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def create_place_holder(n_feat,n_class):
    
    """
      Create place holders for original input and output
    """
    
    X = tf.placeholder(tf.float32,[None,n_feat],name='X')
    Y = tf.placeholder(tf.float32,[None,n_class],name='Y')
    
    return X,Y


def initialize_parameters(n_feat,hidden1,hidden2,nclass):
    
    """
    #Build a two-hidden-layer nnet. First layer with 25 neurons and second with 13
    """
    W1 = tf.Variable(tf.truncated_normal([n_feat,hidden1],stddev=0.01))
    b1 = tf.Variable(tf.constant(0.01, shape=[hidden1]))
    W2 = tf.Variable(tf.truncated_normal([hidden1,hidden2],stddev=0.01))
    b2 = tf.Variable(tf.constant(0.01, shape=[hidden2]))
    W3 = tf.Variable(tf.truncated_normal([hidden2,nclass],stddev=0.01))
    b3 = tf.Variable(tf.constant(0.01, shape=[nclass]))
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    return params





def forward_prop(placeholder,parameters):
    """
      Calculate the output of neural network
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.matmul(placeholder,W1)+b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(A1,W2)+b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(A2,W3) + b3
    output = tf.nn.sigmoid(Z3)
    return Z3,output


def cost_function(Z3,label):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label,logits = Z3))


def PrepareTarget(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

"""
   PARAMETER ZONE
"""

nfeat = 5
n_class = 2
nhidden1 = 25
nhidden2 = 12
nepoch = 300
learning_rate = 0.0025

K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)


sess = tf.Session()
X,Y = create_place_holder(nfeat,n_class)
parameters = initialize_parameters(nfeat,nhidden1,nhidden2,n_class)
Z3,prob = forward_prop(X,parameters)
#prob = tf.nn.sigmoid(Z3)
cost = cost_function(Z3,Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.85).minimize(cost) 

col = list(test.columns)

for i, (train_index,test_index) in enumerate(kf.split(train)):
    init = tf.global_variables_initializer()
    sess.run(init)
    training,valid = train.iloc[train_index,:],train.iloc[test_index,:]
    ylabel = PrepareTarget(list(training['target']))
    for j in range(nepoch):
        sess.run(optimizer,feed_dict={X:training[col],Y:ylabel})
        pb = sess.run(prob,feed_dict = {X:valid[col]})[:,1]
        gini = eval_gini(valid['target'],pb)
        print("Gini: ", gini)