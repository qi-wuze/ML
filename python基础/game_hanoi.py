def atob(a, b):
    b.insert(0, a.pop(0))
    return a, b


k = int(input("请输入汉诺塔层数： "))
a = [x for x in range(1, k + 1)]
b = []
c = []


def caozuo(a, b, c):
    caozuo = input("请操作：")
    atob(eval(caozuo[0]), eval(caozuo[-1]))
    if sorted(a) != a or sorted(b) != b or sorted(c) != c:
        print("error,please 重新输入：")
        atob(eval(caozuo[-1]), eval(caozuo[0]))
    else:
        return a, b, c


while b != [x for x in range(1, k + 1)]:
    caozuo(a, b, c)

print("you win")
