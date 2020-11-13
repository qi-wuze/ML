# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:47:01 2020

@author: 18131
"""

import numpy as np
import pandas as pd

yangben = np.array(pd.read_table("abalone.txt",sep="\t",header=None))
class linear:
    def fit(self, x_train, y_train):
        x_train = np.mat(np.insert(x_train, obj=0, values=1,axis=1))
        y_train = np.mat(y_train).T
        xtx = x_train.T*x_train
        if np.linalg.det(x_train.T * x_train) == 0:
            print("解不出来")
        else:
            w = xtx.I*x_train.T * y_train
            return w
    def cov(self,w,x_train,y_train):
        w = np.mat(w)
        x_train = np.mat(np.insert(x_train, obj=0, values=1,axis=1))
        y_train = np.mat(y_train)
        cov = np.corrcoef((x_train*w).T, y_train)
        return cov

linear = linear()
x_train = yangben[:,0:-1]
y_train = (yangben[:,-1]).T
w = linear.fit(yangben[:,0:-1],yangben[:,-1])
cov = linear.cov(w,x_train,y_train)
print(cov)

        
    