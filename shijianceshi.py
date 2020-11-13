import time
a = 1
starttime = time.time()
for i in range(1, 1000000):
    pass
endtime = time.time()
print(round(endtime - starttime, 2))
