import numpy as np
import pandas as pd
import random
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_table("datingTestSet.txt", sep="\t", header=None, encoding="GBK")
KNN = KNeighborsClassifier()
guiyi = MinMaxScaler(feature_range=(0, 1), copy=True)
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
t = guiyi.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(t, y, test_size=0.5)  # shuffle = True
KNN.fit(x_train, y_train)
print(KNN.predict(x_test))  # 预测
print(KNN.score(x_test, y_test))  # 计算精确度
a = eval(input("飞行距离："))
b = eval(input("游戏时间比"))
c = eval(input("冰淇淋："))
input = guiyi.transform([(a - x_train[:, 0].min()) / (x_train[:, 0].max() - x_train[:, 0].min()), ])
