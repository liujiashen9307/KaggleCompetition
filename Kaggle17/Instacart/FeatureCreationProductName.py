# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:53:04 2017

@author: Jiashen Liu

Purpose: Create TFIDF Features for Product Name

"""

import pandas as pd

product = pd.read_csv('data/products.csv')

product = product[['product_id','product_name']]

## Do some data cleansing and preprocessing
import re
def clean_data(name):
    content = re.sub('<[^>]*>', '', name)
    letters_only = re.sub("[^a-zA-Z-0-9]", " ", content)
    letters_only = re.sub('-',' ',content)
    words = letters_only.lower().split()
    return(" ".join(words))

product['product_name'] = product['product_name'].apply(lambda x: clean_data(x))

## Extract TFIDF Features

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

Input = list(product['product_name'])

tfidf_matrix = tfidf.fit_transform(Input)

tfidf_matrix = tfidf_matrix.toarray()

from sklearn.decomposition import PCA

PCA = PCA(n_components=10,random_state=42)

pca_features = PCA.fit_transform(tfidf_matrix)

for i in range(1,11):
    product['pca_product_name_'+str(i)] = pca_features[:,i-1]
    
product.to_csv('data/product_pca_features.csv',index=False)