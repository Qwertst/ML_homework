with open('data.txt', 'r') as file:
    data = file.read().splitlines()
a, b = data[0], data[1]
n, m = len(a), len(b)
match = 1
mismatch = -2
gap = -1
f = [[0] * (m + 1) for i in range(n + 1)]

for i in range(1, n + 1):
    f[i][0] = f[i - 1][0] + gap

for i in range(1, m + 1):
    f[0][i] = f[0][i - 1] + gap

for i in range(1, n + 1):
    for j in range(1, m + 1):
        x = match if a[i - 1] == b[j - 1] else mismatch
        f[i][j] = max(f[i - 1][j - 1] + x, f[i - 1][j] + gap, f[i][j - 1] + gap)

new_a, new_b = "", ""
s = ""
i, j = n, m
while (i, j) != (0, 0):
    if max(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) == f[i - 1][j - 1]:
        new_a = a[i - 1] + new_a
        new_b = b[j - 1] + new_b
        if a[i - 1] == b[j - 1]:
            s = a[i - 1] + s
        else:
            s = a[i - 1] + b[j - 1] + s
        i, j = i - 1, j - 1
    elif max(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) == f[i][j - 1]:
        new_a = "-" + new_a
        new_b = b[j - 1] + new_b
        s = b[j - 1] + s
        i, j = i, j - 1
    else:
        new_a = a[i - 1] + new_a
        new_b = "-" + new_b
        s = a[i - 1] + s
        i, j = i - 1, j

f = open("output.txt", "w")
f.write("%s\n" % new_a)
f.write("%s\n" % new_b)
f.write("%s\n" % s)
f.close()
