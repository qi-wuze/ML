# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:44:29 2018

@author: admin
"""
import numpy as  np
from loadData import load
from normalization import MinMaxNorm
from kNeighbors import kNN

X, y = load("datingTestSet.txt", sep = '\t')
normalize = MinMaxNorm()
normalize.getMinMax(X)
X_norm = normalize.transform(X)

dating = kNN(k = 3)
dating.fit(X_norm, y)
accuracy = np.sum(dating.predict(X_norm) == y) / len(y)
print('预测精度:', accuracy)

