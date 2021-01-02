# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:53:43 2020

@author: 18131
"""

import pandas as pd
import numpy as np


class nb:
    def zhibiao(self, x):
        dic = {}
        x = pd.DataFrame(x, columns=[x for x in range(1, x.shape[1] + 1)])
        for i in range(0, len(x.columns)):
            lie = x.iloc[:, i]
            lie = lie.value_counts()
            dic[lie.index[0]] = 1
            if len(lie.index) > 1:
                dic[lie.index[1]] = 0
            x[i + 1] = x[i + 1].map(dic)
        return x, dic

    def fit(self, x_input, x_samplee):
        x_sample, dic = self.zhibiao(x_samplee)
        x_sample.columns = [x for x in range(1, x_sample.shape[1] + 1)]
        x0 = x_sample[x_sample[x_sample.shape[1]] == 0]
        x1 = x_sample[x_sample[x_sample.shape[1]] == 1]
        py0 = 1 - x_sample.iloc[:, -1].sum() / x_sample.shape[0]
        for i in range(0, x_sample.shape[1] - 1):
            t1 = x0.iloc[:, i]
            p = t1.sum() / len(t1)
            if dic[x_input[i]] == 1:
                p = p
            else:
                p = 1 - p
            py0 = py0 * p
        py1 = x1.shape[0] / (x1.shape[0] + x0.shape[0])
        for i in range(0, x_sample.shape[1] - 1):
            t1 = x1.iloc[:, i]
            p = t1.sum() / len(t1)
            if dic[x_input[i]] == 1:
                p = p
            else:
                p = 1 - p
            py1 = py1 * p
        dic_new = {}
        t3 = pd.DataFrame(x_samplee).iloc[:, -1].unique()
        dic_new[dic[t3[0]]] = t3[0]
        dic_new[dic[t3[1]]] = t3[1]
        if py0 > py1:
            print("选择是:{:}，概率为：{:}".format(dic_new[0], py0 / (py0 + py1)))
        else:
            print("选择是:{:}，概率为：{:}".format(dic_new[1], py1 / (py0 + py1)))


def predict():
    n_b = nb()
    a = np.array(pd.read_table('data.txt', sep=" ", header=None))
    b = ['不帅', '不好', '高', '不上进']
    n_b.fit(b, a)


predict()
