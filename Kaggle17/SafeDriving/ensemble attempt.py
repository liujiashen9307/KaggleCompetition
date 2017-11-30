# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:20:24 2017

@author: Jiashen Liu
"""

import pandas as pd

df1 = pd.read_csv('submission/kagglemix.csv')
df2 = pd.read_csv('submission/xgb_undersample.csv')

average = 0.8*df1['target'] + 0.2*df2['target']

new_sub = pd.DataFrame({'id':df1['id'],'target':average})

new_sub.to_csv('mix_kglmix_and_best_my.csv',index=False)


average2 = 0.85*df1['target'] + 0.15*df2['target']

new_sub2 = pd.DataFrame({'id':df1['id'],'target':average2})

new_sub2.to_csv('submission/mix_kglmix_and_best_my_2.csv',index=False)

df3 = pd.read_csv('submission/sub_xgb_284_other_seed.csv')

average3 = 0.9*average2 + 0.1*df3['target']

new_sub3 = pd.DataFrame({'id':df1['id'],'target':average3})

new_sub3.to_csv('submission/mix_kglmix_and_best_my_3.csv',index=False)

## 20171122
df1 = pd.read_csv('submission/xgb_undersample.csv')
df2 = pd.read_csv('submission/single_lgb.csv')
df3 = pd.read_csv('submission/single_catboost.csv')

average1 = 0.5*df1['target'] + 0.3*df2['target'] + 0.2*df3['target']

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average1

sub.to_csv('submission/ensemble_boost_models.csv',index=False)


### How about all those files?

df3 = pd.read_csv('submission/Froza_and_Pascal.csv')
df4 = pd.read_csv('submission/gpari.csv')
df5 = pd.read_csv('submission/rgf_submit.csv')
df6 = pd.read_csv('submission/stacked_1.csv')

df_all = pd.read_csv('submission/kagglemix.csv')

average = 0.8*df_all['target'] + 0.1*df2['target']+0.1*df3['target']

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/ensemble_kaggle_and_lgb_cat.csv',index=False)

## 20171122 Final Try

df7 = pd.read_csv('submission/gpx.csv')

average = 0.85*average + 0.15*df7['target']

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/ensemble_best_and_LR.csv',index=False)

## 20171123

df1 = pd.read_csv('submission/ensemble_best_and_LR.csv')
df2 = pd.read_csv('submission/NN_simple.csv')

average = 0.85*df1['target'] + 0.15*df2['target']

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/try_add_simple_NN.csv',index=False)


df1 = pd.read_csv('submission/ensemble_best_and_LR.csv')
df2 = pd.read_csv('submission/stack_seven_models.csv')

average = 0.85*df1['target'] + 0.15*df2['target']

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/stack_and_blend_average_inc_blend.csv',index=False)


### 20171124 Blend three stacked models

df1 = pd.read_csv('submission/stack_seven_models_lgb_meta_undersampled_attepmt.csv')
df2 = pd.read_csv('submission/stack_seven_models_xgb_meta.csv')
df3 = pd.read_csv('submission/stack_seven_models.csv')

average = np.exp(0.4*np.log(df1['target'])+0.4*np.log(df2['target'])+0.2*np.log(df3['target']))

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/log_average_three_stacked_model.csv',index=False)


### 20171124 Blend best and best stacked average

df1 = pd.read_csv('submission/ensemble_best_and_LR.csv')

df2 = pd.read_csv('submission/log_average_three_stacked_model.csv')

average = 0.75*df1['target'] + 0.25*df2['target']

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/stack_and_current_best_blend.csv',index=False)



### 20171124 log-average seven results
df1 = pd.read_csv('submission/gpx.csv')
df2 = pd.read_csv('submission/xgb_undersample.csv')
df3 = pd.read_csv('submission/Froza_and_Pascal.csv')
df4 = pd.read_csv('submission/gpari.csv')
df5 = pd.read_csv('submission/rgf_submit.csv')
df6 = pd.read_csv('submission/stacked_1.csv')
df7 = pd.read_csv('submission/single_lgb.csv')
df8 = pd.read_csv('submission/single_catboost.csv')

sums = np.log(df1['target'])+np.log(df2['target'])+np.log(df3['target'])+np.log(df4['target'])+np.log(df5['target'])+np.log(df6['target'])+np.log(df7['target'])+np.log(df8['target'])
avg = sums/8

final = np.exp(avg)

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = final

sub.to_csv('submission/log_average_eight_results.csv',index=False)


## 20171124 Final Try: Average 0.287 Results

df1 = pd.read_csv('submission/ensemble_best_and_LR.csv')
df2 = pd.read_csv('submission/stack_and_blend_average_inc_blend.csv')
df3 = pd.read_csv('submission/stack_and_blend_average.csv')
df4 = pd.read_csv('submission/stack_and_current_best_blend.csv')
df5 = pd.read_csv('submission/try_add_simple_NN.csv')

average = (df1['target']+df2['target']+df3['target']+df4['target']+df5['target'])/5

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/blend_five_287_results.csv',index=False)

### 20171125 log average everything

xgb_sub = pd.read_csv('stacking2/xgb_undersample.csv')
forza_sub= pd.read_csv('stacking2/Froza_and_Pascal.csv')
stacker_sub = pd.read_csv('stacking2/stacked_1.csv')
rgf_sub = pd.read_csv('stacking2/rgf_submit.csv')
gp_sub = pd.read_csv('stacking2/gpari.csv')

average = (np.log(xgb_sub['target'])+np.log(forza_sub['target'])+np.log(stacker_sub['target'])+np.log(rgf_sub['target'])+np.log(gp_sub['target']))/5

df2 = pd.read_csv('submission/single_lgb.csv')
df3 = pd.read_csv('submission/single_catboost.csv')

average1 = 0.8*average + 0.1*np.log(df2['target']) + 0.1*np.log(df3['target'])

df7 = pd.read_csv('submission/gpx.csv')

average2 = 0.85*average1 + 0.15*np.log(df7['target'])

final = np.exp(average2)

sub = pd.DataFrame()
sub['id'] = df7['id']
sub['target'] = final

sub.to_csv('submission/log_result_current_best.csv',index=False)

## Add NN Results

NN = pd.read_csv('submission/NN_simple.csv')
average3 = 0.95*average2 + 0.05*np.log(NN['target'])
final2 = np.exp(average3)

sub = pd.DataFrame()
sub['id'] = df7['id']
sub['target'] = final2

sub.to_csv('submission/log_result_current_best_and_little_NN.csv',index=False)

## Another Try

df1 = pd.read_csv('submission/log_average_three_stacked_model.csv')
df2 = pd.read_csv('submission/log_result_current_best.csv')

average = np.exp(np.log(df2['target'])*0.9+np.log(df1['target'])*0.1)

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/log_average_blending_stacking_best_adjust_weights.csv',index=False)


## 20171126: Based on 1125 best results: add a NN model

NN = pd.read_csv('submission/NN.csv')

average3 = 0.9*average2 + 0.1*np.log(NN['target'])

pred =np.exp(average3)

sub = pd.DataFrame()
sub['id'] = NN['id']
sub['target'] = pred

sub.to_csv('submission/current_best_add_NN.csv',index=False)

### hmean attempt

from scipy.stats import hmean

testing = pd.DataFrame()

xgb_sub = pd.read_csv('stacking2/xgb_undersample.csv')

testing['xgb'] = xgb_sub['target']

forza_sub= pd.read_csv('stacking2/Froza_and_Pascal.csv')

testing['forza'] = forza_sub['target']

stacker_sub = pd.read_csv('stacking2/stacked_1.csv')

testing['stacker'] = stacker_sub['target']

rgf_sub = pd.read_csv('stacking2/rgf_submit.csv')

testing['rgf'] = rgf_sub['target']

gp_sub = pd.read_csv('stacking2/gpari.csv')

testing['gp'] = gp_sub['target']

hmean1 = hmean(testing,axis=1)

testing = pd.DataFrame()

lgb_sub = pd.read_csv('stacking2/single_lgb.csv')

testing['lgb'] = 0.1*lgb_sub['target']


cat_sub = pd.read_csv('stacking2/single_catboost.csv')

testing['cat'] = 0.1*cat_sub['target']

testing['hmean1'] = 0.8*hmean1

hmean2 = hmean(testing,axis=1)

df7 = pd.read_csv('submission/gpx.csv')

testing = pd.DataFrame()

testing['gpx'] = df7['target']*0.15

testing['hmean2'] = 0.85*hmean2

hmean3 = hmean(testing,axis=1)

NN = pd.read_csv('submission/NN.csv')


testing = pd.DataFrame()

testing['NN'] = NN['target']*0.1

testing['hmean3'] = 0.9*hmean3

hmean4 = hmean(testing,axis=1)


sub = pd.DataFrame()
sub['id'] = NN['id']
sub['target'] = hmean4

sub.to_csv('submission/current_best_by_hmean.csv',index=False)

## gmean attempt


from scipy.stats import gmean

testing = pd.DataFrame()

xgb_sub = pd.read_csv('stacking2/xgb_undersample.csv')

testing['xgb'] = xgb_sub['target']

forza_sub= pd.read_csv('stacking2/Froza_and_Pascal.csv')

testing['forza'] = forza_sub['target']

stacker_sub = pd.read_csv('stacking2/stacked_1.csv')

testing['stacker'] = stacker_sub['target']

rgf_sub = pd.read_csv('stacking2/rgf_submit.csv')

testing['rgf'] = rgf_sub['target']

gp_sub = pd.read_csv('stacking2/gpari.csv')

testing['gp'] = gp_sub['target']

gmean1 = gmean(testing,axis=1)

testing = pd.DataFrame()

lgb_sub = pd.read_csv('stacking2/single_lgb.csv')

testing['lgb'] = 0.1*lgb_sub['target']


cat_sub = pd.read_csv('stacking2/single_catboost.csv')

testing['cat'] = 0.1*cat_sub['target']

testing['gmean1'] = 0.8*gmean1

gmean2 = gmean(testing,axis=1)

df7 = pd.read_csv('submission/gpx.csv')

testing = pd.DataFrame()

testing['gpx'] = df7['target']*0.15

testing['gmean2'] = 0.85*hmean2

gmean3 = gmean(testing,axis=1)

NN = pd.read_csv('submission/NN.csv')


testing = pd.DataFrame()

testing['NN'] = NN['target']*0.1

testing['hmean3'] = 0.9*hmean3

gmean4 = gmean(testing,axis=1)


sub = pd.DataFrame()
sub['id'] = NN['id']
sub['target'] = gmean4

sub.to_csv('submission/current_best_by_gmean.csv',index=False)


### 20171127 At least make some submission today

df1 = pd.read_csv('submission/current_best_add_NN.csv')
df2 = pd.read_csv('submission/NN_EntityEmbed_10fold-sub.csv')

average = 0.9*df1['target'] + 0.1*df2['target']

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/add_entity_embedding_NN.csv',index=False)


### 20171129 Final Day. Quickily do five ensembles

import numpy as np

df1 = pd.read_csv('submission/add_entity_embedding_NN.csv')
df2 = pd.read_csv('submission/Level2_stacking_log_average.csv')

average = 0.65*np.log(df1['target']) + 0.35*np.log(df2['target'])
average = np.exp(average)

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/log_average_best_blending_and_stacking.csv',index=False)

df1 = pd.read_csv('submission/Level2_stacking_log_average.csv')
df2 = pd.read_csv('submission/Level3_stacking_LR_XGB_LGB_Log_Average.csv')

average = (df1['target'] + df2['target'])/2

sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/average_two_best_stacking.csv',index=False)


df3 = pd.read_csv('submission/level3_stacking_LR_only.csv')

average = np.exp(0.65*np.log(average) + 0.35*np.log(df3['target']))
sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/log_average_three_best_stacking.csv',index=False)


df3 = pd.read_csv('submission/level3_stacking_LR_only.csv')

average = 0.65*average + 0.35*df3['target']
sub = pd.DataFrame()
sub['id'] = df1['id']
sub['target'] = average

sub.to_csv('submission/average_three_best_stacking.csv',index=False)





