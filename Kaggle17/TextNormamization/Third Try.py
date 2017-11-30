# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:11:58 2017

@author: Jiashen Liu
"""


import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from datetime import datetime
import string
import roman
import num2words
import inflect
p = inflect.engine()


train = pd.read_csv("input/en_train.csv")

def bagOfWords(cls):
    d = defaultdict(list)
    if cls=='All':
        train_cls = train
    else:
        train_cls = train[train["class"]==cls]
    train_list = [(train_cls.iloc[i,3],train_cls.iloc[i,4]) for i in range(train_cls.shape[0])]
    for k,v in train_list:
        d[k].append(v)
    counter_dict = {}
    for key in d:
        c = Counter(d[key]).most_common(1)[0][0]
        counter_dict[key] = c
    return counter_dict

train_bow = bagOfWords('All')

len(train)
len(train_bow)


test = pd.read_csv('input/test_rf.csv')

test.columns

test.drop("Unnamed: 0", axis=1,inplace=True)

test['id'] = test[['sentence_id','token_id']].apply(lambda x: str(x[0])+"_"+str(x[1]),axis=1)

def transform(x,cls):
    if x in train_bow:
        return train_bow[x]
    else:
        if cls=='CARDINAL':
            return cardinal(x)
        elif cls=='DIGIT':
            return digit(x)
        elif cls=='LETTERS':
            return letters(x)
        elif cls=='TELEPHONE':
            return telephone(x)
        elif cls=='ELECTRONIC':
            return electronic(x)
        elif cls=='MONEY':
            return money(x)
        elif cls=='FRACTION':
            return fraction(x)
        elif cls =='ORDINAL':
            return ordinal(x)
        elif cls =='ADDRESS':
            return address(x)
        else:
            return plain(x)
        
test['after'] = test.apply(lambda x:transform(x['before'],x['class']),axis=1)

test = test[['id','after']]

test.to_csv('input/sub_11200937.csv',index=False)