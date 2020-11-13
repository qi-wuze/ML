# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:00:40 2019

@author: admin
"""

import numpy as np

class NaiveBayes(object):
    def fit(self, X, y):
        n, m = np.shape(X) #n:num of samples, m: num of features
        indices_zero, indices_nonzero = np.where(y == 0), np.where(y == 1)
        num_zero, num_nonzero = len(indices_zero[0]), len(indices_nonzero[0])
        self.prior = np.array([num_zero/n, num_nonzero/n])
        prob = np.zeros((2, 2, m), dtype = float) #class:0/1, feature:0/1, features
        for j in range(m):
            prob[0, 0, j] = np.count_nonzero(X[indices_zero, j] == 0) / num_zero
            prob[0, 1, j] = 1 - prob[0, 0, j]
            prob[1, 0, j] = np.count_nonzero(X[indices_nonzero, j] == 0) / num_nonzero
            prob[1, 1, j] = 1 - prob[1, 0, j]
        self.prob = prob
        self.num_features = m
        return self
        
    def predict(self, X):
        new_classes, new_prob = [], []
        for sample in X:
            prob = self.prior.copy()
            for ind, value in enumerate(sample):
                prob[0] *= self.prob[0, value, ind]
                prob[1] *= self.prob[1, value, ind]
            prob /= np.sum(prob)
            new_prob.append(prob)
            new_classes.append(0 if prob[0] > prob[1] else 1)
        self.new_prob = new_prob
        return new_classes    
    
if __name__ == '__main__':
    X = np.array([[0, 1, 1, 1],
                 [1, 0, 1, 0],
                 [0, 0, 1, 0],
                 [1, 0, 0, 0],
                 [0, 1, 1, 0],
                 [1, 1, 1, 1],
                 [0, 0, 0, 1],
                 [1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 1, 1],
                 [0, 0, 1, 1]])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
    classes = {0: '嫁', 1: '不嫁'}
    nb = NaiveBayes()
    nb.fit(X, y)
    new_samples = np.array([[1, 1, 1, 1]])
    for k in range(len(new_samples)):
        print(classes[nb.predict(new_samples)[k]])
        print('prob of the new sample: {}'.format(nb.new_prob[k]))    
    
