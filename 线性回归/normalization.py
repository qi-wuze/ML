# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:44:29 2018

@author: admin
"""

import numpy as np
from loadData import load

class MinMaxNorm(object):
    def __init__(self, minFeatures = None, maxFeatures = None):
        self.minFeatures = minFeatures
        self.maxFeatures = maxFeatures
    def getMinMax(self, X):
        if np.ndim(X) != 2:
            print('the matrix X for input should be 2-dimensional')
            return None
        self.minFeatures = np.min(X, axis = 0)
        self.maxFeatures = np.max(X, axis = 0)
        return None
    def transform(self, X):
        if self.minFeatures is None:
            print("first 'getMinMax', then 'transform'")
            return X
        if np.ndim(X) != 2:
            print('the matrix X for input should be 2-dimensional')
            return X
        rows = len(X)
        return (X - np.tile(self.minFeatures, (rows, 1)))/np.tile(self.maxFeatures - self.minFeatures, (rows, 1))

if __name__ == "__main__":
    X, y = load("datingTestSet.txt", sep = '\t')
    print('old-X:\n', X)
    normalize = MinMaxNorm()
    normalize.getMinMax(X)
    X = normalize.transform(X)
    print('new-X:\n', X)
