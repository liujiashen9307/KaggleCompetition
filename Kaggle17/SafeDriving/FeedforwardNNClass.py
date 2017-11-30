# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:43:42 2017

@author: Jiashen Liu

@Purpose: Wraper of tensorflow for feed-forward neural network training and validation
          Up to 4 layers are supported
          Either Cross Validation or Simple spliting is supported.
          Also, you can train and make prediction on specific data set.
"""


import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

class ANN(object):
    
    """
    arguments:
        n_hidden1 --> size of first hidden layer
        weight_l1 --> weight of l1 regularization
        num_features --> number of features
        num_class --> number of unique labels
        n_layers --> number of layers
        num_epoch --> number of epoches
        n_hidden2 --> Optional
        n_hidden3 --> Optional
        n_hidden4 --> Optional
        
    """
    
    def __init__(self,n_hidden1,weight_l1,learning_rate,num_features,num_class,n_layers,num_epoch,n_hidden2=6,n_hidden3=5,n_hidden4=4,batch_size=0):
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.weight_l1 = weight_l1
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_class = num_class
        self.num_epoch = num_epoch
        self.n_layers = n_layers
        self.n_hidden3 = n_hidden3
        self.n_hidden4 = n_hidden4
        self.batch_size = batch_size
           
    def weight_variable(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def report_performance(self,ytrue, ypred):
        print(accuracy_score(ytrue,ypred))
        
    def outputs_1layer(self,placeholder):
        w1 = self.weight_variable([self.num_features, self.n_hidden1])
        b1 = self.bias_variable([self.n_hidden1])
        w2 = self.weight_variable([self.n_hidden1,self.num_class])  
        b2 = self.bias_variable([self.num_class])
        h1 = tf.nn.relu(tf.matmul(placeholder, w1) + b1)
        output = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
        reg = self.weight_l1*(tf.reduce_mean(tf.square(w1))+tf.reduce_mean(tf.square(w2)))
        return output,reg
    
    def outputs_2layer(self,placeholder):
        w1 = self.weight_variable([self.num_features, self.n_hidden1])
        b1 = self.bias_variable([self.n_hidden1])
        w2 = self.weight_variable([self.n_hidden1,self.n_hidden2])
        b2 = self.bias_variable([self.n_hidden2])
        w3 = self.weight_variable([self.n_hidden2, self.num_class])
        b3 = self.bias_variable([self.num_class])
        h1 = tf.nn.relu(tf.matmul(placeholder, w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        output = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)
        reg = self.weight_l1 * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2))+tf.reduce_mean(tf.square(w3)))
        return output, reg
    
    def outputs_3layer(self,placeholder):
        w1 = self.weight_variable([self.num_features, self.n_hidden1])
        b1 = self.bias_variable([self.n_hidden1])
        w2 = self.weight_variable([self.n_hidden1,self.n_hidden2])
        b2 = self.bias_variable([self.n_hidden2])
        w3 = self.weight_variable([self.n_hidden2, self.n_hidden3])
        b3 = self.bias_variable([self.n_hidden3])
        w4 = self.weight_variable([self.n_hidden3,self.num_class])
        b4 = self.bias_variable([self.num_class])
        h1 = tf.nn.relu(tf.matmul(placeholder, w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
        output = tf.nn.sigmoid(tf.matmul(h3, w4) + b4)
        reg = self.weight_l1 * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2))+tf.reduce_mean(tf.square(w3))+tf.reduce_mean(tf.square(w4)))
        return output, reg
    
    def outputs_4layer(self,placeholder):
        w1 = self.weight_variable([self.num_features, self.n_hidden1])
        b1 = self.bias_variable([self.n_hidden1])
        w2 = self.weight_variable([self.n_hidden1,self.n_hidden2])
        b2 = self.bias_variable([self.n_hidden2])
        w3 = self.weight_variable([self.n_hidden2, self.n_hidden3])
        b3 = self.bias_variable([self.n_hidden3])
        w4 = self.weight_variable([self.n_hidden3,self.n_hidden4])
        b4 = self.bias_variable([self.n_hidden4])
        w5 = self.weight_variable([self.n_hidden4,self.num_class])
        b5 = self.bias_variable([self.num_class])
        h1 = tf.nn.relu(tf.matmul(placeholder, w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
        h4 = tf.nn.relu(tf.matmul(h3 ,w4) + b4)
        output = tf.nn.sigmoid(tf.matmul(h4, w5) + b5)
        reg = self.weight_l1 * (tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2))+tf.reduce_mean(tf.square(w3))+tf.reduce_mean(tf.square(w4))+tf.reduce_mean(tf.square(w5)))
        return output, reg
    
    def PrepareTarget(self,y):
        N = len(y)
        K = len(set(y))
        ind = np.zeros((N, K))
        for i in range(N):
            ind[i, y[i]] = 1
        return ind
    
    def Cross_validation(self,training_set,label_name,col,num_split):
        x = tf.placeholder(tf.float32, [None,self.num_features])
        if self.n_layers==1:
            y,regularization= self.outputs_1layer(x)
        elif self.n_layers==2:    
            y,regularization= self.outputs_2layer(x)
        elif self.n_layers==3:
            y,regularization= self.outputs_3layer(x)
        else:
            y,regularization= self.outputs_4layer(x)
        y_ = tf.placeholder(tf.float32, [None, self.num_class])
        if self.num_class==2:
           cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
        else:
           cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy+regularization)
        sess = tf.Session()
        init=tf.global_variables_initializer()
        sess.run(init)
        for _ in range(self.num_epoch):
            kf = KFold(n_splits=num_split,shuffle=True)
            for train_index, test_index in kf.split(training_set):
                accu = []
                X_train, X_test = training_set.iloc[train_index,:], training_set.iloc[test_index,:]
                batch_xs = X_train[col]
                batch_ys = self.PrepareTarget(list(X_train[label_name]))
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                if _%5==0:
                    test_y_batch = X_test[col]
                    pred = sess.run(y, feed_dict={x:test_y_batch})
                    pred = np.argmax(pred,1)
                    accu.append(accuracy_score(pred,X_test[label_name]))
            if _%5==0:
                print('Iteration '+str(_)+' '+str(np.mean(accu)))
     
    def Simple_validation(self,training_set,label_name,col,test_size):
        x = tf.placeholder(tf.float32, [None,self.num_features])
        if self.n_layers==1:
            y,regularization= self.outputs_1layer(x)
        elif self.n_layers==2:    
            y,regularization= self.outputs_2layer(x)
        elif self.n_layers==3:
            y,regularization= self.outputs_3layer(x)
        else:
            y,regularization= self.outputs_4layer(x)
        y_ = tf.placeholder(tf.float32, [None, self.num_class])
        if self.num_class==2:
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
        else:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy+regularization)
        sess = tf.Session()
        init=tf.global_variables_initializer()
        sess.run(init)
        for _ in range(self.num_epoch):
            training,validation = train_test_split(training_set,random_state=42,test_size=test_size)
            batch_xs = training[col]
            batch_ys = self.PrepareTarget(list(training[label_name]))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if _%5==0:
                test_y_batch = validation[col]
                pred = sess.run(y, feed_dict={x:test_y_batch})
                pred = np.argmax(pred,1)
                score = accuracy_score(pred,validation[label_name])
                print('iteration '+str(_)+' accuracy '+str(score))
            if _==self.num_epoch-1:
                pred = sess.run(y, feed_dict={x:test_y_batch})
                pred = np.argmax(pred,1)
                print(confusion_matrix(pred,validation[label_name]))

    def train_predict(self,training_set,prediction_set,label_name,col):
        x = tf.placeholder(tf.float32, [None,self.num_features])
        if self.n_layers==1:
            y,regularization= self.outputs_1layer(x)
        elif self.n_layers==2:    
            y,regularization= self.outputs_2layer(x)
        elif self.n_layers==3:
            y,regularization= self.outputs_3layer(x)
        else:
            y,regularization= self.outputs_4layer(x)
        y_ = tf.placeholder(tf.float32, [None, self.num_class])
        if self.num_class==2:
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
        else:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy+regularization)
        sess = tf.Session()
        init=tf.global_variables_initializer()
        sess.run(init)
        for _ in range(self.num_epoch):
            batch_xs = training_set[col]
            batch_ys = self.PrepareTarget(list(training_set[label_name]))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        pred = sess.run(y,feed_dict={x:prediction_set[col]})
        pred = np.argmax(pred,1)
        prediction_set['pred'] = pred
        return prediction_set