# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 06:56:19 2019

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    def fit(self, X, y):
        self.X = np.insert(X, obj=0, values=1, axis=1)  # X列方向(每行插入)0位置插入1
        self.y = y
        xTx = self.X.T * self.X
        if np.linalg.det(xTx) == 0:
            print('xTx is singular, can not do inverse')
            return None
        self.w = xTx.I * self.X.T * self.y  #.I是逆
        self.PearsonCorrCoef = np.corrcoef((self.X * self.w).T[0], (self.y).T[0])
        return self

    def predict(self, X):
        X = np.insert(X, obj=0, values=1, axis=1)
        return X * self.w


if __name__ == '__main__':
    reg = LinearRegression()
    X = np.mat([[52], [55], [60], [75], [78], [80], [84], [92], [95], [98]])
    y = np.mat([[160], [152], [250], [280], [310], [350], [320], [345], [380], [350]])
    reg.fit(X, y)
    print("W values:\n", reg.w)
    print("Correlation Coefficient: %.4f" % reg.PearsonCorrCoef)
    y_pred = reg.predict(X)
    plt.plot(np.asarray(reg.X[:, 1].T)[0], np.asarray(reg.y.T)[0], 'b*', label='sample')
    plt.plot(np.asarray(reg.X[:, 1].T)[0], np.asarray(y_pred.T)[0], 'r-', label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
