import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

with open('input.txt', 'r') as file:
    data = file.read().splitlines()

city = list(map(int, data[0].split()))
klen = city[0]
k_x = [klen * i for i in range(city[1] + 1)]
k_y = [klen * i for i in range(city[2] + 1)]
schoolnumber = int(data[1])
school = []
for i in range(2, 2 + schoolnumber):
    school.append(list(map(int, data[i].split())))
school = np.array(school)
pointnumber: int = int(data[2 + schoolnumber])
points = []
for i in range(3 + schoolnumber, 3 + schoolnumber + pointnumber):
    points.append(list(map(int, data[i].split())))
points = np.array(points)

tree = KDTree(school, metric='manhattan')
dist, n = tree.query(points)
for i in range(pointnumber):
    print(points[i], school[n[i]])

fig, ax = plt.subplots()
ax.scatter(school[:, 0], school[:, 1])
ax.scatter(points[:, 0], points[:, 1])
ax.set_xticks(k_x)
ax.set_yticks(k_y)
plt.grid(True)
plt.show()
