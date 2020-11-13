# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:17:41 2018

@author: john
"""
import numpy as np


def calcEntropy(X, y):
    class_num = {}
    for eachClass in y: class_num[eachClass] = class_num.get(eachClass, 0) + 1
    nums = np.array(list(class_num.values()))
    nums = nums / np.sum(nums)
    Entropy = -np.sum(nums * np.log2(nums))
    return Entropy


def splitDataSet_byAttr(X, y, attribute):
    valuesOfAttribute = set([sample[attribute] for sample in X])
    sub_DataSet = {}
    for value in valuesOfAttribute:
        indices = [i for i in range(len(X)) if X[i][attribute] == value]
        sub_DataSet[value] = [X[i] for i in indices], [y[i] for i in indices]
    return sub_DataSet


def generateTree(X, y, attributes):
    classes = set(y)
    # all samples are same class
    if len(classes) == 1: return classes.pop()
    # attributes is null, or X[i] is same at all attributes
    if (not attributes) or all([(X[i] == X[0]) for i in range(1, len(X))]):
        class_num = {}
        for eachClass in y: class_num[eachClass] = class_num.get(eachClass, 0) + 1
        max_num = max(class_num.values())
        for key, value in class_num.items():
            if value == max_num: return key
    # select the best attribute
    infoGain = {}
    baseEntropy = calcEntropy(X, y)
    for attr in attributes:
        infoGain[attr] = baseEntropy
        sub_DataSet = splitDataSet_byAttr(X, y, attr)
        for sub_X, sub_y in sub_DataSet.values():
            infoGain[attr] -= len(sub_X) / len(X) * calcEntropy(sub_X, sub_y)
    max_infoGain = max(infoGain.values())
    for key, value in infoGain.items():
        if value == max_infoGain: bestAttribute = key
    sub_DataSet = splitDataSet_byAttr(X, y, bestAttribute)
    sub_attributes = [i for i in attributes if i != bestAttribute]
    tree = {bestAttribute: {}}
    print(tree)
    for key, value in sub_DataSet.items():
        sub_X, sub_y = value
        tree[bestAttribute][key] = generateTree(sub_X, sub_y, sub_attributes)
    #print(tree)
    return tree


class DecisionTree(object):
    def fit(self, X, y):
        attributes = list(range(len(X[0])))
        self.tree = generateTree(X, y, attributes)
        return self

    def predict(self, X):
        y_predict = []
        for each_X in X:
            track_tree = self.tree
            while type(track_tree) is dict:
                for key, value in track_tree.items(): track_tree = value[each_X[key]]
            y_predict.append(track_tree)
        return y_predict


if __name__ == '__main__':
    classes = {0: 'bear', 1: 'eagle', 2: 'penguin', 3: 'dolphin'}
    features = {0: 'have feather?', 1: 'can fly?', 2: 'have fin?'}
    answers = {0: 'No', 1: 'Yes'}
    X = [[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]]
    y = [0, 1, 2, 3]

    dt = DecisionTree()
    dt.fit(X, y)
    X_new = [[0, 0, 1]]
    predict_classes = dt.predict(X_new)
    for each_X, each_y in zip(X_new, predict_classes):
        print('The animal with features:')
        for attr, attr_value in enumerate(each_X):
            print('{0}: {1}'.format(features[attr], answers[attr_value]))
        print('belongs to the class: {}'.format(classes[each_y]))
