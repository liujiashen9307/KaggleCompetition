# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:06:56 2017

@author: LUI01
"""

import pandas as pd
from num2words import num2words 

train = pd.read_csv('en_train.csv')
def compare(a,b):
    if a==b:
        return 0
    else:
        return 1

train['diff'] = train.apply(lambda x:compare(x['before'],x['after']),axis=1)

def cal_diff(types):
    tmp = train[train['class']==types]
    #tmp['diff'] = tmp.apply(lambda x:compare(x['before'],x['after']),axis=1)
    all_diff = sum(tmp['diff'])
    print('calculating '+types+' '+str(len(tmp))+' '+str(len(tmp)/len(train)))
    print(all_diff/len(tmp))
    print('=='*20)
    
types = list(set(train['class']))

for each in types:
    cal_diff(each)
    
"""
calculating DATE 258348 0.026047238673900464
1.0
========================================
calculating TIME 1465 0.00014770466447297514
1.0
========================================
calculating ADDRESS 522 5.26292388088007e-05
1.0
========================================
calculating MEASURE 14783 0.001490456010173373
0.998511804099
========================================
calculating MONEY 6128 0.000617839033372281
0.999510443864
========================================
calculating PUNCT 1880507 0.18959703445329765
0.0
========================================
calculating ELECTRONIC 5162 0.0005204446948870291
0.961642774119
========================================
calculating FRACTION 1196 0.00012058346669602612
1.0
========================================
calculating TELEPHONE 4024 0.00040570892139198085
1.0
========================================
calculating DECIMAL 9821 0.0009901757746000606
1.0
========================================
calculating LETTERS 152795 0.015405142804196748
0.944854216434
========================================
calculating VERBATIM 78108 0.007875027940378936
0.330785578942
========================================
calculating CARDINAL 133744 0.01348437723226866
1.0
========================================
calculating PLAIN 7353693 0.7414162165203181
0.00496594024254
========================================
calculating DIGIT 5442 0.0005486749379262325
1.0
========================================
calculating ORDINAL 12703 0.001280745633310719
1.0
========================================

"""
res = dict()
train = open('en_train.csv', encoding='UTF8')
line = train.readline().strip()
pos = line.find('","')
text = line[pos + 2:]
arr = text.split('","')
res[arr[0]] = dict()
res[arr[0]][arr[1]] = 1