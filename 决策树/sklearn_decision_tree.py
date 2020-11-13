# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:17:41 2018

@author: john
"""
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
print('X shape: {0}, y shape: {1}'.format(X.shape, y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#dt = tree.DecisionTreeClassifier(criterion = 'entropy')
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
print('Accuracy on the training set is: {0}\nAccuracy on the testing set is: {1}'.format(
        dt.score(X_train, y_train), dt.score(X_test, y_test)))

tree.export_graphviz(dt, out_file = 'iris.dot',
                     feature_names = iris.feature_names, 
                     class_names = iris.target_names,
                     filled = True)

    
    
        
    