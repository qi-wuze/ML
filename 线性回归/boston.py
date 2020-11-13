# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:03:08 2019

@author: john
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

boston = load_boston()
X = boston.data
y = boston.target
print('X shape: %s, y shape: %s' % (X.shape, y.shape))

X0 = X
for degree in [1, 2, 3]:
    pf = PolynomialFeatures(degree = degree, include_bias = False)
    X = pf.fit_transform(X0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    reg = LinearRegression(normalize = True)
    reg.fit(X_train, y_train)
    train_score = reg.score(X_train, y_train)
    test_score = reg.score(X_test, y_test)
    print('degree = %d, train score: %.6f, test score: %.6f' % (degree, train_score, test_score))
