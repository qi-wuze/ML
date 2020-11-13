import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = pd.read_table("datingTestSet.txt", sep="\t", header=None, encoding="GBK")


class KNN:
    def __init__(self, ceshi, k=3):
        self.k = k
        self.ceshi = ceshi

    def panduan(self, y, p):
        self.ceshi.columns = ["1", "2", "3", "4"]
        juli_0 = self.ceshi.drop(labels="4", axis=1).sub(np.tile(y, (len(self.ceshi), 1)))
        juli_1 = juli_0.apply(lambda x: x ** p)
        juli_2 = juli_1.sum(axis=1).apply(lambda x: x ** (1 / p))
        juli = pd.DataFrame(juli_2)
        juli.columns = ["juli"]
        juli["xihao"] = list(self.ceshi.iloc[:, -1])
        juli.sort_values(by="juli", inplace=True)
        kgezuixiaode = juli.iloc[0:self.k, -1]
        kgezuixiaode.index = [x for x in range(0, len(kgezuixiaode))]
        jieguo = kgezuixiaode.value_counts()
        zhenjieguo = jieguo.sort_values(ascending=False)
        return zhenjieguo.index[0]


def guiyi(data):
    new_data = pd.DataFrame(index=data.index, columns=data.columns)
    for i in data.columns[0:-1]:
        d = data[i]
        min = d.min()
        max = d.max()
        new_data[i] = (d - min) / (max - min)
    new_data[new_data.columns[-1]] = data[new_data.columns[-1]]
    return new_data


y = list(a.index)
random.shuffle(y)
whole = guiyi(a).reindex(y)


def pridict(data_biaozhun=whole[0:150], data_ceshi=whole[150:750], p=2, k=3):
    text = KNN(data_biaozhun, k)
    zhengque = 0
    data_biaozhun.index = [x for x in range(0, len(data_biaozhun))]
    data_ceshi.index = [x for x in range(0, len(data_ceshi))]
    for i in range(0, len(data_ceshi)):
        canshu = np.array(data_ceshi.iloc[i, 0:-1]).reshape(1, 3)
        if text.panduan(canshu, p) == data_ceshi.iat[i, -1]:
            zhengque += 1
    return zhengque / len(data_ceshi)


# print(pridict(data_ceshi=whole[750:1000], k=3, p=2))
# print(pridict(data_ceshi=whole[500:1000], k=9, p=4))
answer = []
for j in range(1,10):
    jishu = 0
    for i in range(0, 400):
        jishu += pridict(data_biaozhun=whole.drop(i, axis=0),
                     data_ceshi=pd.DataFrame(np.array(whole.iloc[i, :]).reshape(1, 4)), k=3, p=2)
    answer.append(jishu/400)

print(np.mean(answer))
