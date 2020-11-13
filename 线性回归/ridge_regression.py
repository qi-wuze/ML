# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from loadData import load
from normalization import MinMaxNorm
from holdout import holdout_split

class RidgeRegression(object):
    def __init__(self, alpha = 0.2):
        self.alpha = alpha
        
    def fit(self, X, y):
        self.X = np.insert(X, obj = 0, values = 1, axis = 1) #X列方向0位置插入1
        self.y = y        
        A = self.X.T * self.X + self.alpha * np.eye(len(self.X.T))
        if np.linalg.det(A) == 0:
            print('(xTx + alpha * I) is singular, can not do inverse')
            return None
        self.w = A.I * self.X.T * self.y
        self.PearsonCorrCoef = np.corrcoef((self.X * self.w).T[0], (self.y).T[0])[0, 1]
        return self
    
    def predict(self, X):
        X = np.insert(X, obj = 0, values = 1, axis = 1)
        return X * self.w

if __name__ == '__main__':
    #input data        
    X, y = load('abalone.txt', sep = '\t') 
    #normalization
    normalize = MinMaxNorm() 
    normalize.getMinMax(X)
    X = normalize.transform(X)
    #transform to matrix
    X = np.mat(X) 
    y = np.mat(y.astype(float)).T
    # dataset split
    X_train, X_test, y_train, y_test = holdout_split(X, y, testsize = 0.3)
    
    for alpha in [0, 0.001, 0.01, 0.1, 1, 10, 100]:
        ridge = RidgeRegression(alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        average_err = np.average(np.abs(y_pred - y_test))
        print('while alpha = %f, average error = %f\n' % (alpha, average_err))
    #    print('w values: ', ridge.w)
    
