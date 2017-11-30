# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 21:24:14 2017

@author: LUI01
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from Frame_Model import *
import tensorflow as tf
from sklearn.model_selection import KFold

class FeatureBinarizatorAndScaler:
    """ This class needed for scaling and binarization features
    """
    NUMERICAL_FEATURES = list()
    CATEGORICAL_FEATURES = list()
    BIN_FEATURES = list()
    binarizers = dict()
    scalers = dict()

    def __init__(self, numerical=list(), categorical=list(), binfeatures = list(), binarizers=dict(), scalers=dict()):
        self.NUMERICAL_FEATURES = numerical
        self.CATEGORICAL_FEATURES = categorical
        self.BIN_FEATURES = binfeatures
        self.binarizers = binarizers
        self.scalers = scalers

    def fit(self, train_set):
        for feature in train_set.columns:

            if feature.split('_')[-1] == 'cat':
                self.CATEGORICAL_FEATURES.append(feature)
            elif feature.split('_')[-1] != 'bin':
                self.NUMERICAL_FEATURES.append(feature)

            else:
                self.BIN_FEATURES.append(feature)
        for feature in self.NUMERICAL_FEATURES:
            scaler = StandardScaler()
            self.scalers[feature] = scaler.fit(np.float64(train_set[feature]).reshape((len(train_set[feature]), 1)))
        for feature in self.CATEGORICAL_FEATURES:
            binarizer = LabelBinarizer()
            self.binarizers[feature] = binarizer.fit(train_set[feature])


    def transform(self, data):
        binarizedAndScaledFeatures = np.empty((0, 0))
        for feature in self.NUMERICAL_FEATURES:
            if feature == self.NUMERICAL_FEATURES[0]:
                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape(
                    (len(data[feature]), 1)))
            else:
                binarizedAndScaledFeatures = np.concatenate((
                    binarizedAndScaledFeatures,
                    self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]),
                                                                                       1)))), axis=1)
        for feature in self.CATEGORICAL_FEATURES:

            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,
                                                         self.binarizers[feature].transform(data[feature])), axis=1)

        for feature in self.BIN_FEATURES:
            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((
                len(data[feature]), 1))), axis=1)
        print(binarizedAndScaledFeatures.shape)
        return binarizedAndScaledFeatures
    
def preproc(X_train):
    # Adding new features and deleting features with low importance
    multreg = X_train['ps_reg_01'] * X_train['ps_reg_03'] * X_train['ps_reg_02']
    ps_car_reg = X_train['ps_car_13'] * X_train['ps_reg_03'] * X_train['ps_car_13']
    X_train = X_train.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12',
                            'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
                            'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_car_10_cat', 'ps_ind_10_bin',
                            'ps_ind_13_bin', 'ps_ind_12_bin'], axis=1)
    X_train['mult'] = multreg
    X_train['ps_car'] = ps_car_reg
    X_train['ps_ind'] = X_train['ps_ind_03'] * X_train['ps_ind_15']
    return X_train

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

target = train['target']
train = train.drop(['id','target'],axis=1)
test = test.drop(['id'],axis=1)

bscaler = FeatureBinarizatorAndScaler()
bscaler.fit(train)


train = bscaler.transform(train)
test = bscaler.transform(test)

train = pd.DataFrame(train)
test = pd.DataFrame(test)


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





def forward_prop(placeholder,parameters,dropout=0.2):
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
    A1 = tf.nn.dropout(A1,keep_prob=(1-dropout))
    Z2 = tf.matmul(A1,W2)+b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(A2,W3) + b3
    output = tf.nn.sigmoid(Z3)
    return Z3,output


def initialize_parameters2(n_feat,hidden1,hidden2,hidden3,nclass):
    
    """
    #Build a two-hidden-layer nnet. First layer with 25 neurons and second with 13
    """
    W1 = tf.Variable(tf.truncated_normal([n_feat,hidden1],stddev=0.01))
    b1 = tf.Variable(tf.constant(0.01, shape=[hidden1]))
    W2 = tf.Variable(tf.truncated_normal([hidden1,hidden2],stddev=0.01))
    b2 = tf.Variable(tf.constant(0.01, shape=[hidden2]))
    W3 = tf.Variable(tf.truncated_normal([hidden2,hidden3],stddev=0.01))
    b3 = tf.Variable(tf.constant(0.01, shape=[hidden3]))
    W4 = tf.Variable(tf.truncated_normal([hidden3,nclass],stddev=0.01))
    b4 = tf.Variable(tf.constant(0.01, shape=[nclass]))
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4":b4}
    return params





def forward_prop2(placeholder,parameters,dropout=0.2):
    """
      Calculate the output of neural network
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    Z1 = tf.matmul(placeholder,W1)+b1
    A1 = tf.nn.relu(Z1)
    A1 = tf.nn.dropout(A1,keep_prob=(1-dropout))
    Z2 = tf.matmul(A1,W2)+b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(A2,W3) + b3
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(A3,W4) + b4
    output = tf.nn.sigmoid(Z4)
    return Z4,output,Z3



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

nfeat = 226
n_class = 2
nhidden1 = 128
nhidden2 = 64
nhidden3 = 12
nepoch = 65#70 # previously 130
learning_rate = 0.0025
batch_size = 128
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)
hlayer = 3

sess = tf.Session()
X,Y = create_place_holder(nfeat,n_class)
if hlayer==2:
    parameters = initialize_parameters(nfeat,nhidden1,nhidden2,n_class)
    Z3,prob = forward_prop(X,parameters)
else:
    parameters = initialize_parameters2(nfeat,nhidden1,nhidden2,nhidden3,n_class)
    Z3,prob,special_feats = forward_prop2(X,parameters)
#prob = tf.nn.sigmoid(Z3)
cost = cost_function(Z3,Y)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.85).minimize(cost) 

kf = KFold(n_splits=5,random_state=42,shuffle=True)

for ii, (train_index, test_index) in enumerate(kf.split(train)):
    y_train, y_valid = target.iloc[train_index].reset_index(drop=True), target.iloc[test_index].reset_index(drop=True)
    X_train, X_valid = train.iloc[train_index,:].reset_index(drop=True), train.iloc[test_index,:].reset_index(drop=True)
    test = test
    init = tf.global_variables_initializer()
    sess.run(init)
    N_batch = len(X_train)//batch_size
    for i in range(nepoch):
        ylabel = PrepareTarget(list(y_train))
        for bt in range(N_batch+1):
            startidx = bt*batch_size
            endidx = min((bt+1)*batch_size,len(X_train))
            batch_target = ylabel[startidx:endidx]
            batch_feat = X_train.iloc[startidx:endidx,:]
            sess.run(optimizer,feed_dict={X:batch_feat,Y:batch_target})
            if bt==N_batch:
                pb = sess.run(prob,feed_dict={X:X_valid})[:,1]
            #print(eval_gini(y_valid,pb))
                print('epoch: ',i,' gini: ',eval_gini(y_valid,pb))
        #if i== nepoch-1:
    feats = sess.run(special_feats,feed_dict={X:X_valid})
    tmp = pd.DataFrame(feats)
    tmp.columns = ['nn_feat_'+str(i) for i in range(nhidden3)]
    tmp['index'] = list(test_index)
    feats_test = sess.run(special_feats,feed_dict={X:test})
    if ii==0:
        final = tmp
        final_test = feats_test    
    else:
        final = pd.concat([final,tmp])
        final_test = final_test+feats_test

final_test = final_test/5
final_test = pd.DataFrame(final_test)
final_test.columns = ['nn_feat_'+str(i) for i in range(nhidden3)]
final = final.sort_values(['index'],ascending=True)
del final['index']
final.to_csv('valid_nn_features.csv',index=False)
final_test.to_csv('test_nn_features.csv',index=False)
         