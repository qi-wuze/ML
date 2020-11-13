# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 08:35:05 2019

@author: john
"""

#import numpy as np


X = [[52], [55], [60], [75], [78], [80], [84], [92], [95], [98]]
y = [160, 152, 250, 280, 310, 350, 320, 345, 380, 350]    

# from sklearn.linear_model import LinearRegression
# reg = LinearRegression().fit(X, y)
# print('intercept: %.6f\ncoefficient: %.6f' % (reg.intercept_, reg.coef_))

# from sklearn.linear_model import Ridge
# for alpha in [0, 0.1, 1.0, 10., 100, 1000]:
#     reg = Ridge(alpha = alpha).fit(X, y)
#     print('alpha = %f, intercept: %.4f, coefficient: %.4f' % (alpha, reg.intercept_, reg.coef_))

from sklearn.linear_model import Lasso
for alpha in [0.1, 1.0, 10., 100, 1000]:
    reg = Lasso(alpha = alpha).fit(X, y)
    print('alpha = %f, intercept: %.4f, coefficient: %.4f' % (alpha, reg.intercept_, reg.coef_))
