


a = [1, 2, 3]
b = [4, 5, 6]
c = zip(a, b)

for index, (a_i, b_i) in enumerate(c):
    print(index, a_i, b_i)