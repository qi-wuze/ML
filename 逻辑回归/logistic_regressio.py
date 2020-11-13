# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:49:41 2019

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt

def g(z): return 1/(1+np.exp(-z))

class LogisticRegression(object):
    def __init__(self, learning_rate = 0.0001, epsilon = 1.e-6, maxIter = 1.e8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon   #误差范围
        self.maxIter = maxIter   #最大迭代次数
    def fit(self, X, y, w0 = None):
        self.X = np.insert(X, obj = 0, values = 1, axis = 1) #X列方向0位置插入1
        self.y = y
        n, m = np.shape(self.X)  #n个样本，m个属性，
        if w0 is None: w0 = np.mat(np.zeros((m, 1), dtype = float))
        iter_num = 0
        while iter_num < self.maxIter:   #梯度下降迭代
            self.w = w0 + self.learning_rate * self.X.T * (self.y - g(self.X * w0))
            diff = np.max(np.abs(self.w - w0))
            if diff < self.epsilon:    #判断是否收敛
                print('Iteration number: %d\n' % iter_num)
                break
            iter_num += 1
            w0 = self.w.copy()
        else: print('The maxmium iteration number is reached')
        return self
    def predict(self, X):
        X = np.insert(X, obj = 0, values = 1, axis = 1)
        return g(X * self.w)

if __name__ == '__main__':
    X = np.mat([[1], [2], [3], [4], [6], [7], [8], [9],[20]])
    y = np.mat([[0], [0], [0], [0], [1], [1], [1], [1],[1]])
    lgr = LogisticRegression(learning_rate = 0.1, epsilon = 1.e-3)
    lgr.fit(X, y)
    print('w =\n', lgr.w)
    X_plot = np.mat(np.linspace(0, 10, 101)).T
    y_plot = lgr.predict(X_plot)
    plt.plot(X[:, 0], y[:, 0], 'r*')
    plt.plot(X_plot[:, 0], y_plot[:, 0], 'b-')
    plt.show()

    
    
    