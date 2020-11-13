# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:49:24 2018

@author: admin
"""

import numpy as np

def load(file, sep = ' '):
    """
    load函数用于读入数据。\n
    输入参数：\n
        file: 文件名；\n
        sep: 文件中数据之间的分隔符，默认为空格。\n
    输出参数：属性矩阵及标签向量。 \n
    示例：\n
        X, y = load(file = "movieData.txt", sep = ",")    
    """
    X, y = [], []  #X， y分别用于存放属性及标签值
    for lines in open(file):  #将file中每一行以字符串形式取出赋给lines
        L = lines.strip().split(sep)  #将lines字符串修剪掉分行符后以空格分隔，得到的list赋给L
        X.append(list(map(float, L[0:-1]))) #将L中属性值用map函数转换为数值型，存入X
        y.append(L[-1]) #将L中最后一列即标签存入y
    return np.asarray(X), np.asarray(y)  #将X, y转换为numpy数组后返回
        
if __name__ == "__main__":
    X, y = load(file = "movieData.txt", sep = "\t")
    print(X, '\n', y)
