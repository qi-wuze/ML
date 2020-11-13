# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:54:36 2019

@author: admin
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from indicator import indicator

X = np.array([[0, 1, 1, 1],
     [1, 0, 1, 0],
     [0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, 1, 1, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, 0, 0, 0],
     [1, 1, 0, 0],
     [0, 0, 1, 1],
     [0, 0, 1, 1]])
y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
classes = {0: '嫁', 1: '不嫁'}
nb = BernoulliNB()
nb.fit(X, y)
X_new = np.array([[1, 1, 1, 1]])
for k in range(len(X_new)):
    print('Classification result: {}'.format(classes[nb.predict(X_new)[k]]))
    print('Probobility: {}'.format(nb.predict_proba(X_new)[k]))
    
print(indicator(np.array([['a', 'c'],
                          ['b', 'c']])))