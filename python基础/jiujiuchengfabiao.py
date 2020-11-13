a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
col_2 =[row[1]for row in a]
print(col_2)
print("\n".join(["\t".join(["{}*{}={}".format(x, y, x*y) for y in range(1, x+1)]) for x in range(1, 10)]))
