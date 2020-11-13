# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:49:24 2018

@author: admin
"""

import numpy as np
from loadData import load

class kNN(object):
    def __init__(self, k = 5):  #kNN类的构造函数，指定k值，默认k=5
        """
        kNN类构造函数 \n
        输入参数：\n
            k: kNN算法中的k值，默认k = 5 \n
        """
        self.k = k  #将k值赋值给kNN实例属性self.k
        self.X = self.y = None  #实例属性self.X及self.y分别用于存放样本的特征矩阵及标签向量，先赋值为None
        
    def fit(self, X, y):  #fit函数用于训练，参数包括特征矩阵及标签向量
        """
        fit方法用于模型训练 \n
        输入参数： \n
            X: 样本集特征矩阵 \n
            y: 样本集标签向量 \n
        """
        self.X, self.y = np.asarray(X), np.asarray(y)  #分别将特征矩阵及标签向量赋值给实例属性self.X, self.y
        
    def predict(self, new):  #predict函数用于对新的样本预测其类别，参数new中存放新样本的特征值
        """
        predict方法用于预测 \n
        输入参数：\n
            new: 新样本特征向量 \n
        输出：新样本类别标签 \n
        """
        if self.X is None:   #如果self.X为None
            print("first fit, then predict") #提醒用户要先进行fit，然后再predict
            return None
        if np.ndim(new) < 2:
            print("The input of predict showed be 2-dimentional")
            return None
        predictClasses = []
        for each in new:
            dist = self.distance(self.X, np.tile(each, (len(self.X), 1)))  #计算新样本与样本集中每个样本的距离
            sortedIndices = np.argsort(dist)   #对距离从小到大排序，并将排序后的索引值存到sortedIndices
            classCount = {}  #classCount字典用于对k个近邻的类别计数
            for i in range(self.k):  #对于前k个样本
                label = self.y[sortedIndices[i]] #先取出每个样本的标签即类别，存到label
                classCount[label] = classCount.get(label, 0) + 1  #字典中键为label的项对应的值加1
            maxNum = max(classCount.values())  #取得字典中values中的最大值
            for key, value in classCount.items(): #从字典中依次取出键值对
                if value == maxNum:  ##如果该值等于最大值
                    predictClass = key  #则其键即为预测的类别
                    break
            predictClasses.append(predictClass)
        return np.asarray(predictClasses)  #返回预测类别
    
    def distance(self, x1, x2): 
        """
        distence方法用于距离计算 \n
        输入参数：\n
            x1, x2: 用于计算距离的两个特征向量或特征矩阵，二者要有相同的shape \n
        输出：x1与x2之间的欧氏距离 \n
        """
        return np.sqrt(np.sum((np.asarray(x1)-np.asarray(x2))**2, axis = -1))
        
if __name__ == "__main__":
    X, y = load("movieData.txt", sep = "\t")  #装载梯形本集数据
    test = kNN(k = 3)  #产生kNN实例test,k值为3
    test.fit(X, y)  #执行fit方法
    new = np.array([[18, 90]]) #将新样本特征赋值给new
    newClass = test.predict(new) #预测new所属类别将其赋值给newClass
    print("预测新电影类别为：", newClass)  #打印预测类别
    