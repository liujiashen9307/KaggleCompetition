# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 00:42:18 2017

@author: jiashen
"""

import pandas as pd
from functions_in import *

sub1 = pd.read_csv('save_test_for_exp_azure_86Feats.csv')
sub2 = pd.read_csv('save_test_for_exp_azure_103_Feats.csv')

sample = pd.read_csv('data/sample_submission.csv')

Average = 0.6*sub1.reordered + 0.4*sub2.reordered

sub = sub1
sub['reordered'] = Average
submission = sub_file(sub,0.2,sample)
submission.to_csv('20170804_azure_blending86Feats_103Feats_0.6_0.4.csv',index=False)