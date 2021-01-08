from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv("data.txt")
X = data['0'].values
Y = data['1'].values
X = np.stack((X, Y), axis = 1)
nclusters = 9

X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X[:, 0], X[:, 1], test_size=0.1, random_state=666)
New_X = np.stack((X_test, Y_test), axis = 1)
print(New_X)
New_X = StandardScaler().fit_transform(New_X)

hierarchy = cluster.AgglomerativeClustering(
    linkage="ward", affinity="euclidean",
    n_clusters=nclusters)
hierarchy.fit(New_X)
y_pred = hierarchy.fit_predict(New_X)

plt.scatter(New_X[:, 0], New_X[:, 1], c=y_pred, s=nclusters, cmap='viridis')

sum, cen_x, cen_y = [0] * nclusters, [0] * nclusters, [0] * nclusters
for x_, y_, i in zip(New_X[:, 0], New_X[:, 1], y_pred):
    sum[i] += 1
    cen_x[i] += x_
    cen_y[i] += y_
centroids = []
for i in range(nclusters):
    centroids.append([cen_x[i] / sum[i], cen_y[i] / sum[i]])
centroids = np.array(centroids)
plt.show()

kmeans = KMeans(init=centroids, n_clusters=nclusters, n_init=1).fit(X)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=nclusters, cmap='viridis')
plt.show()
