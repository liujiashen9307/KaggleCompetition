# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:13:52 2017

@author: Jiashen Liu

Purpose: Store useful functions here in order to boost productivity

"""

"""
Function 1: Load data

Purpose: Load data with minimum cost

"""

import pandas as pd
import numpy as np

DIR = 'data/'

def load_data(path_data):
    '''
    --------------------------------order_product--------------------------------
    * Unique in order_id + product_id
    '''
    priors = pd.read_csv(path_data + 'order_products__prior.csv', 
                     dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    train = pd.read_csv(path_data + 'order_products__train.csv', 
                    dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    '''
    --------------------------------order--------------------------------
    * This file tells us which set (prior, train, test) an order belongs
    * Unique in order_id
    * order_id in train, prior, test has no intersection
    * this is the #order_number order of this user
    '''
    orders = pd.read_csv(path_data + 'orders.csv', 
                         dtype={
                                'order_id': np.int32,
                                'user_id': np.int64,
                                'eval_set': 'category',
                                'order_number': np.int16,
                                'order_dow': np.int8,
                                'order_hour_of_day': np.int8,
                                'days_since_prior_order': np.float32})

    '''
    --------------------------------product--------------------------------
    * Unique in product_id
    '''
    products = pd.read_csv(path_data + 'products.csv', 
                           dtype={
                                'product_id': np.uint32,
                                'order_id': np.int32,
                                'aisle_id': np.uint8,
                                'department_id': np.uint8},
                                usecols=['product_id', 'aisle_id', 'department_id'])
    aisles = pd.read_csv(path_data + "aisles.csv")
    departments = pd.read_csv(path_data + "departments.csv")
    sample_submission = pd.read_csv(path_data + "sample_submission.csv")
    
    return priors, train, orders, products, aisles, departments, sample_submission


"""
Function 2: Feature Creation: Create Useful features

目的： 创造一系列特征在数据集中以供训练

函数的作用： 在原数据中：以一系列的特征为组，用特定计算方式来创建新特征。

举例：

agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)

在训练集里，以product为中心，分别找出： 每个产品卖了多少个，每个产品被重新订购多少个，每个产品被买一次的
次数与被买两次的次数。

然后可以选择join回去。

"""

def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    df_new = df.copy()
    grouped = df_new.groupby(group_columns_list)
    the_stats = grouped.agg(agg_dict)
    the_stats.columns = the_stats.columns.droplevel(0)
    the_stats.reset_index(inplace=True)
    if only_new_feature:
        df_new = the_stats
    else:
        df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
    return df_new

"""
Function 3: String to List

Only for validation

"""

def string_to_list(string):
    return string.split(' ')

"""
Function 4: Create Submission File/validation file

函数作用： 创建与提交文件相同格式的数据集。同时：也可用来创建CV数据集

"""

def sub_file(test_set,threshold,submission):
    d = dict()
    for row in test_set.itertuples():
        if row.reordered > threshold:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)
    for order in submission.order_id:
        if order not in d:
            d[order] = 'None'
    sub = pd.DataFrame.from_dict(d, orient='index')
    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']
    return sub

"""
Function 5: Create Benchmark File from cv set. Run this function can help you to create
            a file for validating the results.
"""

def CV_file(test):
    d = dict()
    for row in test.itertuples():
        if row.reordered==1:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)
    #else:
        #d[row.order_id] = 'None'
    T_test = pd.DataFrame.from_dict(d, orient='index')
    T_test.reset_index(inplace=True)
    T_test.columns = ['order_id', 'products_o']
    return T_test

"""
Correct Way of Creating Validation File

"""
def CV_file2(test):
    d = dict()
    for row in test.itertuples():
        if row.reordered==1:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)
    for order in test.order_id:
        if order not in d:
            d[order]='None'
    T_test = pd.DataFrame.from_dict(d, orient='index')
    T_test.reset_index(inplace=True)
    T_test.columns = ['order_id', 'products_o']
    return T_test

"""
Function 6: F1-Score Calculation, based on list

"""

def f1_score_single(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)
    
def f1_score(y_true, y_pred):
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])

"""
Function 7: Feature Importance of XGB Model

"""

def get_feature_importance(model):
    Importance = model.get_fscore()
    Importance = list(Importance.items())
    Feature= []
    Score = []
    for each in Importance:
        Feature.append(each[0])
        Score.append(each[1])
    df = pd.DataFrame({'Feature':Feature,'Score':Score}).sort_values(by=['Score'],ascending=[0])
    return df  

"""
Function 8: CV 函数

"""  

def Cross_Validation(test,pred,threshold):
    testing = test.copy()
    validation_file = CV_file2(testing)
    testing['reordered'] = pred
    testing_file = sub_file(testing,threshold,validation_file)
    valid = validation_file.merge(testing_file,on='order_id',how='left')
    valid['products_o'] = valid['products_o'].apply(lambda x: string_to_list(x))
    valid['products'] = valid['products'].apply(lambda x: string_to_list(x))
    return f1_score(valid['products'],valid['products_o'])


"""
Function 9: Feature Importance of LGB Model

"""

def get_feature_importance_lgb(model):
    Importance = list(model.feature_importance())
    Feature= model.feature_name()
    df = pd.DataFrame({'Feature':Feature,'Score':Importance}).sort_values(by=['Score'],ascending=[0])
    return df  
