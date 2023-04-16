import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# defining the game assets
e1 = np.array([[1.0, 1.0], [1.0, 6.0], [4.0, 6.0], [8.0, 1.0], [8.0, 9.0], [9.0, 1.0], [9.0, 9.0], [10.0, 3.0]])
e2 = np.array([[2.0, 1.0], [3.0, 9.0], [3.0, 10.0], [5.0, 6.0], [7.0, 2.0], [9.0, 10.0], [10.0, 5.0]])

# performing K-means clustering with Euclidean distance
kmeans_euclidean = KMeans(n_clusters=3, init='random', algorithm='full', max_iter=2, random_state=42).fit(np.vstack((e1, e2)))
labels_euclidean = kmeans_euclidean.labels_

# performing K-means clustering with Mahalanobis distance
kmeans_mahalanobis = KMeans(n_clusters=3, init='random', algorithm='full', max_iter=2, random_state=42).fit(np.vstack((e1, e2)), y=None, sample_weight=None)
labels_mahalanobis = kmeans_mahalanobis.labels_

# plotting the results
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.scatter(e1[:, 0], e1[:, 1], c=labels_euclidean[:8], cmap='viridis')
plt.scatter(e2[:, 0], e2[:, 1], c=labels_euclidean[8:], cmap='viridis', marker='*')
plt.title('K-means Clustering with Euclidean Distance')
plt.xlabel('X co-ordinate')
plt.ylabel('Y co-ordinate')

plt.subplot(122)
plt.scatter(e1[:, 0], e1[:, 1], c=labels_mahalanobis[:8], cmap='viridis')
plt.scatter(e2[:, 0], e2[:, 1], c=labels_mahalanobis[8:], cmap='viridis', marker='*')
plt.title('K-means Clustering with Mahalanobis Distance')
plt.xlabel('X co-ordinate')
plt.ylabel('Y co-ordinate')

plt.show()
