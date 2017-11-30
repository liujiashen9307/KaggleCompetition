# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:24:05 2017

@author: LUI01
"""

import pandas as pd

PATH = "input/"

train = pd.read_csv(PATH + 'en_train.csv',encoding = 'latin')
sub = pd.read_csv(PATH+'submission_kernel_forked.csv',encoding = 'latin')
types = pd.read_csv(PATH+'pred_test.csv',encoding = 'latin')

types['id'] = types.apply(lambda x:str(x['sentence_id'])+'_'+str(x['token_id']),axis=1)

sub = sub.merge(types,on='id',how='left')

#Further Processing of the data 

#label = list(set(sub['predict']))

"""
['DECIMAL',
 'PUNCT',
 'ELECTRONIC',
 'DIGIT',
 'VERBATIM',
 'DATE',
 'CARDINAL',
 'MONEY',
 'PLAIN',
 'TIME',
 'TELEPHONE',
 'LETTERS',
 'FRACTION',
 'ORDINAL',
 'MEASURE',
 'ADDRESS']

"""
train_plain = train[train['before']!=train['after']]
train_plain = train_plain[train_plain['class']=='PLAIN']
#train_plain_before = list(train_plain['before'])

"""
Possible Improvement 1: 
    Check PLAIN TEXT
"""

plain = sub[sub['predict']=='PLAIN']
plain = plain[plain['data']!=plain['after']][['id','data','after']]
plain.columns = ['id','before','after_sub']

train_plain = train[train['class']=='PLAIN'][['before','after']]
train_plain = train_plain.drop_duplicates(['before'])

# (37457, 3)

plain = plain.merge(train_plain,on='before',how = 'inner')



good  = plain[plain['after']==plain['after_sub']]
bad = plain[plain['after']!=plain['after_sub']]

bad = bad[['id','before','after']]
del bad['before']
bad.columns = ['id','after_mod']

sub = sub.merge(bad,on='id',how='left')

sub = sub.fillna('fillnawithplaceholder')

def fill_an_col(original,aftermod):
    if aftermod== 'fillnawithplaceholder':
        return original
    else:
        return aftermod

sub['true_after'] = sub.apply(lambda x: fill_an_col(x['after'],x['after_mod']),axis=1)

sub['after'] = sub['true_after']
sub = sub[['id','after']]

sub.to_csv('submission_modify_some_plain.csv',index=False)

"""
0.9878 So Bad...
"""

