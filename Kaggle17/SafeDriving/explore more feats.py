# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:13:26 2017

@author: LUI01
"""

from Frame_Model import *
import numpy as np
import tensorflow as tf

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





def forward_prop(placeholder,parameters,num_layers):
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


def initialize_parameters2(n_feat,hidden1,nclass):
    
    """
    #Build a two-hidden-layer nnet. First layer with 25 neurons and second with 13
    """
    W1 = tf.Variable(tf.truncated_normal([n_feat,hidden1],stddev=0.01))
    b1 = tf.Variable(tf.constant(0.01, shape=[hidden1]))
    W2 = tf.Variable(tf.truncated_normal([hidden1,nclass],stddev=0.01))
    b2 = tf.Variable(tf.constant(0.01, shape=[nclass]))
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2
              }
    return params





def forward_prop2(placeholder,parameters):
    """
      Calculate the output of neural network
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = tf.matmul(placeholder,W1)+b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(A1,W2)+b2
    output = tf.nn.sigmoid(Z2)
    return Z2,output

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

nfeat = 50
n_class = 2
num_h_layers = 1
nhidden1 = 500
nhidden2 = 6
nepoch = 300
learning_rate = 0.001#0.00005#0.000005

K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)


sess = tf.Session()
X,Y = create_place_holder(nfeat,n_class)
if num_h_layers==2:
    parameters = initialize_parameters(nfeat,nhidden1,nhidden2,n_class)
    Z3,prob = forward_prop(X,parameters)
else:
    parameters = initialize_parameters2(nfeat,nhidden1,n_class)
    Z3,prob = forward_prop2(X,parameters)
#prob = tf.nn.sigmoid(Z3)
cost = cost_function(Z3,Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



FEATS = FEATS.fillna(-1)
test_df = test_df.fillna(-1)

for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = FEATS.iloc[train_index,:].copy(), FEATS.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    
    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_valid[f],
                                                        tst_series=X_test[f],
                                                        target=y_train,
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )
    
    """
       #Tensorflow Part
    """
    #sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(nepoch):
        ylabel = PrepareTarget(list(y_train))
        sess.run(optimizer,feed_dict={X:X_train,Y:ylabel})
        pb = sess.run(prob,feed_dict={X:X_valid})[:,0]
        if i%10==0:
            print('epoch: ',i,' gini:', eval_gini(y_valid,pb))
        
        
    
    
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    #y_test_pred += fit_model.predict(X_test)
    
    #del X_test, X_train, X_valid, y_train
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)

