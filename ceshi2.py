import numpy as np
import pandas as pd

def guiyi(data):
    new_data = pd.DataFrame(index=data.index, columns=data.columns)
    for i in data.columns[0:-1]:
        d = data[i]
        min = d.min()
        max = d.max()
        new_data[i] = (d-min)/(max-min)
        new_data[new_data.columns[-1]] = data[new_data.columns[-1]]
    return new_data


a = pd.read_table("datingTestSet.txt", sep="\t", header=None, encoding="GBK")
print(guiyi(a))


text = KNN(new_a[0:150], k=5)
zhengque = 0
for i in range(150, 750):
    canshu = np.array(new_a.iloc[i, 0:3]).reshape(1, 3)
    if text.panduan(canshu, 2) == a.iat[i, -1]:
        zhengque += 1
print(zhengque / 600)