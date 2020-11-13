# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:44:29 2018

@author: admin
"""
import numpy as np
from loadData import load

def holdout_split(X, y, testsize = 0.3, shuffle = True):
    num = len(X)
    indices = np.arange(num)
    if shuffle: np.random.shuffle(indices)

    testNum = int(num * testsize)
    testIndices, trainIndices = indices[:testNum], indices[testNum:]
    return X[trainIndices], X[testIndices], y[trainIndices], y[testIndices]

if __name__ == '__main__':
    X, y = load("datingTestSet.txt", sep = '\t')
    X_train, X_test, y_train, y_test = holdout_split(X, y, testsize = 0.4)
    print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
        
    
    
    


