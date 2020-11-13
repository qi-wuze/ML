# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 06:38:48 2019

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

X = np.mat([[52], [55], [60], [75], [78], [80], [84], [92], [95], [98]], 
           dtype = float)
y = np.mat([[160], [152], [250], [280], [310], [350], [320], [345], [380], [350]], 
           dtype = float)

x_plot = np.linspace(np.min(X), np.max(X), 50)
x_for_plot = np.mat(x_plot.copy()).T
train_indices = [0, 2, 4, 6, 8, 9]
test_indices = [1, 3, 5, 7]
for n in range(1, 6):
    if n > 1: X = np.insert(X, n-1, np.mat(np.array(X[:, 0]) * np.array(X[:, -1])).T, 
                            axis = 1)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    err_train = np.average(np.abs(y_train_pred - y_train))
    y_test_pred = reg.predict(X_test)
    err_test = np.average(np.abs(y_test_pred - y_test))
    print('while n = %d,\n' % n)
    print('Correlation Coefficient: %.4f\n' % reg.PearsonCorrCoef)
    print('Average train error: %.2f\n' % err_train)
    print('Average test error: %.2f\n' % err_test)
    if n > 1: x_for_plot = np.insert(x_for_plot, n-1, np.mat(np.array(x_for_plot[:, 0]) * 
                                     np.array(x_for_plot[:, -1])).T, axis = 1)
    y_pred_plot = reg.predict(np.mat(x_for_plot))
    plt.plot(x_plot, np.array(y_pred_plot.T)[0], 'r-')
    plt.plot(np.array(X[:, 0].T)[0], np.array(y[:, 0].T)[0], '*')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

