# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:44:29 2018

@author: admin
"""
import numpy as np
from loadData import load


def cv_split(X, kFold=10, shuffle=True):
    num = len(X)
    indices = np.arange(num)
    if shuffle: np.random.shuffle(indices)

    valNum = int(num / kFold)
    valIndexArray = [np.arange(valNum) + i * valNum for i in range(kFold)]  #ndarray 可以广播+1
    train_val_indexArray = []
    for valIndices in valIndexArray:
        trainIndices = np.array(list(set(indices) - set(valIndices)))
        train_val_indexArray.append((trainIndices, valIndices))
    return train_val_indexArray


if __name__ == '__main__':
    X, y = load("datingTestSet.txt", sep='\t')
    for traInd, valInd in cv_split(X, kFold=10):
        X_train, X_val, y_train, y_val = (X[traInd], X[valInd], y[traInd], y[valInd])
        print(np.shape(X_train), np.shape(X_val), np.shape(y_train), np.shape(y_val))
