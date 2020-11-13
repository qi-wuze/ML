# -*- coding: utf-8 -*-
"""
Created on 2020/11/2 0:13

@author: Qi
"""
import numpy as np
import pandas as pd


def xinxishang(x):
    x_count = x.iloc[:, -1].value_counts()  # 对最后一列进行统计
    x_count = x_count.map(lambda x: x / x_count.sum())  # 计算统计后各项百分比
    result = x_count.map(lambda x: -x * np.log2(x)).sum()  # 每项转化为单项信息熵再求和
    return result


def fenlei(x_input):
    dic = {}
    result = []
    for item in x_input.columns[0:-1]:
        item_class = x_input[item].unique()  # 获取每一种属性的取值
        xinxis = 0
        for clas_s in item_class:
            temp_df = x_input[x_input[item] == clas_s]  # 布尔索引获得某种属性下对应的数据
            clas_s_number = temp_df.shape[0]
            xinxis += (clas_s_number / x_input.shape[0]) * xinxishang(temp_df)
        dic[xinxis] = item
    xinxis_max = np.array(list(dic.keys())).max()
    sort_class = dic[xinxis_max]
    item_class = x_input[sort_class].unique()
    for clas_s in item_class:
        temp_df = x_input[x_input[sort_class] == clas_s]
        result.append((temp_df.drop(sort_class, axis=1), clas_s))  # 封装成元组放入列表
    return result, sort_class


class Tree:
    def fit(self, data):
        indicator = 0
        for item in data.columns[0:-1]:
            indicator += len(data[item].unique())  # 计算每列是否取值都相同
        if len(data.iloc[:, -1].unique()) == 1:  # 结果取值都一样的情况
            return data.iloc[0, -1]
        elif indicator == len(data.columns) - 1:  # 每列取值都一样的情况
            temp = data[data.columns[-1]].value_counts()
            return temp.index[0]
        else:
            x, y = fenlei(data)
            bestAttribute = y
            tree1 = {bestAttribute: {}}  # 得到分类特征
            for i in range(0, len(x)):
                s, t = x[i]
                tree1[bestAttribute][t] = self.fit(s)
            return tree1

    def predict1(self, d_tree, x_input):   #用于大量数据测试
        if isinstance(d_tree, dict) == 0:  # 判断是不是字典，不是字典说明已经到了叶节点，就输出结果
            return d_tree
        else:
            temp = list(d_tree.keys())[0]  # 取出当前问题
            next_tree = d_tree[temp][x_input[temp]]  # 根据输入的数据对应的答案找到下一个问题
            return self.predict1(next_tree, x_input)  # 迭代，每一次next_tree都会减小

    def predict2(self, d_tree):            #用于让用户输入
        if isinstance(d_tree, dict) == 0:  # 判断是不是字典，不是字典说明已经到了叶节点，就输出结果
            print(d_tree)
        else:                              #道理与predict1相同
            temp = list(d_tree.keys())[0]
            x_input = input("请问"+str(temp)+"? (回答“是”或”否“)")
            next_tree = d_tree[temp][x_input]
            self.predict2(next_tree)


x_train = pd.read_table("data.txt", sep=" ")
x_train.set_index(['编号'], inplace=True)
tree = Tree()
tree_dic = tree.fit(x_train)
# x_input = {"有没有羽毛": "是", "会不会飞": "否", "有没有鳍": "否"}
tree.predict2(tree_dic)
