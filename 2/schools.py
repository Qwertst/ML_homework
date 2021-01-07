import matplotlib.pyplot as plt
import random
import numpy as np


k_x=[10,20,35,40,60,75,100]
k_y=[5,20,30,55,70,85,100]



from sklearn.neighbors import KDTree
points = np.array([[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)]])
people=np.array([[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)],[random.randint(0,100), random.randint(0,100)]])


tree = KDTree(points,metric='manhattan')
dist, n=tree.query(people)


for i in range(len(people)):
  print(people[i],points[n[i]])

fig, ax = plt.subplots() 
ax.scatter(points[:,0],points[:,1])
ax.scatter(people[:,0],people[:,1])
ax.set_xticks(k_x) 
ax.set_yticks(k_y)
plt.grid(True)
plt.show()

