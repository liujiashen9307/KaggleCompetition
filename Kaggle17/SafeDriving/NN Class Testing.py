# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:14:38 2017

@author: Jiashen Liu
"""

from FeedforwardNNClass import *

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

col = list(test.columns)
col.remove('id')

nn = feed_forward_NN(30,0.01,0.0005,57,1,2,200,15)

nn.Cross_validation(train,'target',col,5)