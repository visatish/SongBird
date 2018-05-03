import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load('10000_songs_03_31_18.npz')
X = data['features']

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print("Finished scaling")
# Find best K
params = []
errs = []
for i in range(1, 30) : 
    kmeans = KMeans(n_clusters=i) 
    kmeans.fit(X_scaled)
    err = kmeans.inertia_
    params.append(i)
    errs.append(err)
    print(i)

# show errs vs parameters
plt.plot(params, errs)
plt.xlabel("k")
plt.ylabel("error")
plt.show()


# project data using PCA
print("computing svd")
u, s, d = np.linalg.svd(X_scaled)
print("done")
pca = u[:, 0:2].T
plt.scatter(pca[0], pca[1], 0.1)
plt.show()
