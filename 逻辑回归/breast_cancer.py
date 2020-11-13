# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:49:41 2019

@author: admin
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

bc = load_breast_cancer()
X, y = bc.data, bc.target
num_negetive, num_positive = np.bincount(y) 
print('data shape: {0}, num. positive: {1}; num. negetive: {2}'.format(X.shape, 
                                            num_positive, num_negetive))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
train_score = lgr.score(X_train, y_train)
test_score = lgr.score(X_test, y_test)
print('train score: {0:.6f}; test score: {1:.6f}'.format(train_score, test_score))

    
    
    