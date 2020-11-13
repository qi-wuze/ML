# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:54:36 2019

@author: admin
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB

def transform(X, vocab):
    X_trans = np.zeros((len(X), len(vocab)), dtype = bool)
    for ind, sentence in enumerate(X):
        for word in sentence.split(' '):
            if word in vocab: X_trans[ind, vocab.index(word)] = True
    return X_trans

X0 = ['my dog has flea problem help please',
      'maybe not take him to dog park stupid',
      'my dalmation is so cute I love him',
      'stop posting stupid worthless garbage',
      'mr licks ate ny steak how to stop him',
      'quit buying worthless dog food stupid'] 
y0 = ['normal', 'abusive', 'normal', 'abusive', 'normal', 'abusive']

#construction of vocabulary
vocab = set()
for sentence in X0:
    for word in sentence.split(' '):
        vocab.add(word)
vocab = list(vocab)
#transform X0 to numerical array
X = transform(X0, vocab)
#transform y0 to numerical array
classes = {0: 'normal', 1: 'abusive'}
y = (np.array(y0) == 'abusive')

nb = BernoulliNB()
nb.fit(X, y)
X_new = ['love my dog', 'stupid garbage']
X_new = transform(X_new, vocab)
for k in range(len(X_new)):
    print('Classification result for {0} sample: {1}'.format(k, classes[nb.predict(X_new)[k]]))
    print('Probobility for {0} sample: {1}'.format(k, nb.predict_proba(X_new)[k]))