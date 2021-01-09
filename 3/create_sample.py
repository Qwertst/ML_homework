from sklearn import datasets
import numpy as np


np.random.seed(0)
n_samples = 100000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=80,cluster_std=0.2)
no_structure = np.random.rand(n_samples, 2), None
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
dataset=datasets.make_blobs(n_samples=n_samples,centers=6,random_state=666,cluster_std=0.1)
X, y = noisy_moons
f = open("dataset_3.txt", "w")
f.write("0,1\n")
for i in range(n_samples):
    s=str(X[i][0])+","+str(X[i][1])
    f.write("%s\n" % s)
f.close()
