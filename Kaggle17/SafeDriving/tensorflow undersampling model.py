# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:35:51 2017

@author: J Liu

@Purpose: undersampling techniques, by tensorflow
"""

import warnings
warnings.filterwarnings("ignore")

from functions import *
from preprocessing import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

train,test = feature_engineering()
train = train.fillna(-1)
test = test.fillna(-1)


def create_place_holder(n_feat,n_class):
    
    """
      Create place holders for original input and output
    """
    
    X = tf.placeholder(tf.float32,[None,n_feat],name='X')
    Y = tf.placeholder(tf.float32,[None,n_class],name='Y')
    
    return X,Y

def initialize_parameters():
    
    """
    Build a two-hidden-layer nnet. First layer with 25 neurons and second with 13
    """
    W1 = tf.get_variable(name='W1',shape=[50,25],initializer=tf.contrib.layers.xavier_initializer(seed=42))
    W2 = tf.get_variable(name='W2',shape=[25,13],initializer=tf.contrib.layers.xavier_initializer(seed=42))
    W3 = tf.get_variable(name='W3',shape=[13,2],initializer=tf.contrib.layers.xavier_initializer(seed=42))
    b1 = tf.get_variable(name='b1',shape=[1,25],initializer=tf.zeros_initializer())
    b2 = tf.get_variable(name='b2',shape=[1,13],initializer=tf.zeros_initializer())
    b3 = tf.get_variable(name='b3',shape=[1,2],initializer=tf.zeros_initializer())
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
    """
      Make target friendly to tensorflow.
    """
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

"""
  Run Model, with undersampling
"""

n_feats = 50
n_class = 2
learning_rate = 0.00001
num_epoch = 800

sess = tf.Session()

col = list(test.columns)
col.remove('id')

f_cats = [f for f in train.columns if "_cat" in f]

y_valid_pred = 0*train['target']
y_test_pred = 0

kf = KFold(n_splits=5,shuffle=True,random_state=42)

"""
  Tensorflow initialization
"""
sess = tf.Session()
num_feats = 50
n_class = 2
X,Y = create_place_holder(n_feats,n_class)
parameters = initialize_parameters()
output,prob = forward_prop(X,parameters)
cost = cost_function(output,Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()
sess.run(init)

Record = {}
for i, (train_index, test_index) in enumerate(kf.split(train)):
    
    # Create data for this fold
    X_train = train.iloc[train_index,:]
    X_valid = train.iloc[test_index,:]
    X_test = test
    print( "\nFold ", i)
    
    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_valid[f],
                                                        tst_series=X_test[f],
                                                        target=X_train['target'],
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )
    ## Update the list of features
    col = list(X_test.columns)
    col.remove('id')
    for i in range(num_epoch):
        ylabel = PrepareTarget(list(X_train['target']))
        feats = X_train[col]
        sess.run(optimizer,feed_dict={X:feats,Y:ylabel})
        cost_v = sess.run(cost,feed_dict={X:feats,Y:ylabel})
        eval_prob = sess.run(prob,feed_dict={X:X_valid[col]})[:,0]
        print(eval_gini(list(X_valid['target']),eval_prob))
        """
        if i%50==0:
            eval_prob = sess.run(prob,feed_dict={X:X_valid[col]})[:,0]
            print(eval_gini(list(X_valid['target']),eval_prob))
        """
            
        
            
    
    