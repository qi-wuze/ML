# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

class BatchGradientDescent(object):
    def __init__(self, learning_rate = 0.0001, epsilon = 1.e-6, maxIter = 1.e8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.maxIter = maxIter
    def fit(self, X, y, w0 = None):
        self.X = np.insert(X, obj = 0, values = 1, axis = 1) #X列方向0位置插入1
        self.y = y
        n, m = np.shape(self.X)
        if w0 is None: w0 = np.mat(np.zeros((m, 1), dtype = float))
        iter_num = 0
        while iter_num < self.maxIter:
            self.w = w0 - self.learning_rate / n * self.X.T * (self.X * w0 - self.y)
            diff = np.max(np.abs(self.w - w0))
            if diff < self.epsilon:
                print('Iteration number: %d\n' % iter_num)
                break
            iter_num += 1
            w0 = self.w.copy()
        else: print('The maxmium iteration number is reached')
        return self
    def predict(self, X):
        X = np.insert(X, obj = 0, values = 1, axis = 1)
        return X * self.w
    
if __name__ == '__main__':
    X = np.mat([[52], [55], [60], [75], [78], [80], [84], [92], [95], [98]])
    y = np.mat([[160], [152], [250], [280], [310], [350], [320], [345], [380], [350]])
    bgd = BatchGradientDescent()
    bgd.fit(X, y)
    print('w =\n', bgd.w)
    
    
        
        
        
        
        
        
