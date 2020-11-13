# 计数用全局变量来计数，防止数字被每一次递归改变。


movetimes = 0


def hanoi(n, A, B, C):
    global movetimes
    if n == 1:
        print(A, "->", B)
        movetimes += 1
    else:
        hanoi(n - 1, A, C, B)
        print(A, "->", B)
        movetimes += 1
        hanoi(n - 1, C, B, A)


hanoi(2, "A", "B", "C")
