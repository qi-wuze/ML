# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:44:29 2018

@author: admin
"""
import numpy as np
from loadData import load
from kNeighbors import kNN

X, y = load("datingTestSet.txt", sep = '\t')
dating = kNN(k = 3)
dating.fit(X, y)
accuracy = np.sum(dating.predict(X) == y) / len(y)
print('预测精度:', accuracy)

        
    
    
    


