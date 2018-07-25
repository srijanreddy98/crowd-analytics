# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:16:23 2018

@author: Rutvik
"""

import pandas as pd
import os
import numpy as np

dataset = pd.read_excel('idea_export.xlsx')


from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.DESCRIPTION.values,dataset.status.values,stratify=dataset.status.values,test_size = 0.25,random_state = 42)

dict = {'good':0,'bad':1}
dataset['status'] = dataset['status'].map(dict)

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
count_vect = CountVectorizer(ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer  = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)