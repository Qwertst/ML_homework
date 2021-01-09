import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.neighbors import kneighbors_graph

data = pd.read_csv("dataset_3.txt")
X = data['0'].values
Y = data['1'].values
X = np.stack((X, Y), axis = 1)
nclusters = 2
plt.scatter(X[:,0],X[:, 1])
plt.savefig('data_3.png')
plt.show()
X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X[:, 0], X[:, 1], test_size=0.01, random_state=666)
New_X = np.stack((X_test, Y_test), axis = 1)
New_X = StandardScaler().fit_transform(New_X)

connectivity = kneighbors_graph(
    New_X, n_neighbors=10, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)

hierarchy = cluster.AgglomerativeClustering(
    linkage="average", affinity="cityblock",
    n_clusters=nclusters,connectivity=connectivity)
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
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", color='r')
plt.savefig('h3.png')
plt.show()
kmeans = KMeans(init=centroids, n_clusters=nclusters, n_init=1).fit(X)
y_kmeans = kmeans.fit_predict(X)
centers = np.array(kmeans.cluster_centers_)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=nclusters, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='r')
plt.savefig('k3.png')
plt.show()
