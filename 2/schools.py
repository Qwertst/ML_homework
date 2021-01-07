import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.neighbors import KDTree

with open('input.txt', 'r') as file:
    data = file.read().splitlines()

k_x = list(map(int, data[0].split()))
k_y = list(map(int, data[1].split()))

schoolnumber = int(data[2])
school = []
for i in range(3, 3 + schoolnumber):
    school.append(list(map(int, data[i].split())))
school = np.array(school)
pointnumber = int(data[3 + schoolnumber])
points = []
for i in range(4 + schoolnumber, 4 + schoolnumber + pointnumber):
    points.append(list(map(int, data[i].split())))
points = np.array(points)

tree = KDTree(school, metric='manhattan')
dist, n = tree.query(points)

for i in range(schoolnumber):
    print(points[i], school[n[i]])

fig, ax = plt.subplots()
ax.scatter(school[:, 0], school[:, 1])
ax.scatter(points[:, 0], points[:, 1])
ax.set_xticks(k_x)
ax.set_yticks(k_y)
plt.grid(True)
plt.show()
