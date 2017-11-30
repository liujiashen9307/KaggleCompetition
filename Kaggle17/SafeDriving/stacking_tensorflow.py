# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:34:55 2017

@author: LUI01
"""

from Frame_Model import *

training = pd.DataFrame()
testing = pd.DataFrame()

xgb_sub = pd.read_csv('stacking2/xgb_undersample.csv')
xgb_valid = pd.read_csv('stacking2/xgb_valid.csv')

training['xgb'] = xgb_valid['target']
testing['xgb'] = xgb_sub['target']

forza_sub= pd.read_csv('stacking2/Froza_and_Pascal.csv')
forza_valid = pd.read_csv('stacking2/forza_pascal_oof.csv')

training['forza'] = forza_valid['target']
testing['forza'] = forza_sub['target']

stacker_sub = pd.read_csv('stacking2/stacked_1.csv')
stacker_valid = pd.read_csv('stacking2/stacker_oof_preds_1.csv')

training['stacker'] = stacker_valid['target']
testing['stacker'] = stacker_sub['target']

rgf_sub = pd.read_csv('stacking2/rgf_submit.csv')
rgf_valid = pd.read_csv('stacking2/rgf_valid.csv')

training['rgf'] = rgf_valid['target']
testing['rgf'] = rgf_sub['target']

gp_sub = pd.read_csv('stacking2/gpari.csv')
gp_valid = pd.read_csv('stacking2/gp_pseu_val.csv')

training['gp'] = gp_valid['target']
testing['gp'] = gp_sub['target']

lgb_sub = pd.read_csv('stacking2/single_lgb.csv')
lgb_valid = pd.read_csv('stacking2/stacking_lgb.csv').sort_values(['id'],ascending=1)

training['lgb'] = lgb_valid['target']
testing['lgb'] = lgb_sub['target']


cat_sub = pd.read_csv('stacking2/single_catboost.csv')
cat_valid = pd.read_csv('stacking2/stacking_cat.csv').sort_values(['id'],ascending=1)

training['cat'] = cat_valid['target']
testing['cat'] = cat_sub['target']

target = pd.read_csv('data/train.csv')['target']

training['target']=target


"""
  Tensorflow Part
"""
import tensorflow as tf
from sklearn.model_selection import KFold

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

## PARAMETER ZONE

num_epoch = 200
learning_rate = 0.0001
n_feat = 7
n_hidden1 = 20
n_hidden2 = 10
n_class = 2

X,Y = create_place_holder(n_feat,n_class)
params = initialize_parameters(n_feat,n_hidden1,n_hidden2,n_class)

Z3,prob = forward_prop(X,params)
cost = cost_function(Z3,Y)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()

y_valid_pred = 0*training['target']
kf = KFold(n_splits=5,random_state=42)
y_valid_pred = 0*training['target']
for i, (train_index, test_index) in enumerate(kf.split(training)):
    
    X_train,X_test = training.iloc[train_index,:].copy(), training.iloc[test_index,:].copy()
    y_train,y_test = X_train['target'],X_test['target']
    del X_train['target']
    del X_test['target']
    
    init = tf.global_variables_initializer()
    sess.run(init)
    evals = []
    for i in range(num_epoch):
        ylabel = PrepareTarget(list(y_train))
        sess.run(optimizer,feed_dict={X:X_train,Y:ylabel})
        pb = sess.run(prob,feed_dict={X:X_test})[:,0]
        gini = eval_gini(y_test,pb)
        evals.append(gini)
    print(gini)
    y_valid_pred.iloc[test_index] = pb        
    pred = sess.run(prob,feed_dict={X:testing})[:,0]
    y_test_pred+=pred
y_test_pred/=5
print(eval_gini(training['target'],y_valid_pred))

"""
0.282928364586
0.286630323972
0.28647796885
0.289622829433
0.279174155206
"""

sub = pd.DataFrame()
sub['target'] = y_test_pred
sub['id'] = xgb_sub['id']

sub.to_csv('submission/stack_seven_models_tensorflow_unsubmitted.csv',index=False)











