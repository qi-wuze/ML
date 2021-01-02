import numpy as np
import pandas as pd

A = np.random.randint(0, 10, size=36).reshape(6, 6)
names = [_ for _ in ('very_long_name_aaaaaaaaaaaaaaaaaaaa','a','b','c','d','f')]
df = pd.DataFrame(A, index=names, columns=names)
df.to_csv('df.csv', index=True, header=True, sep=' ')