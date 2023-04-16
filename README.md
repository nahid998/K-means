# K-means
This python code use Euclidean Distance and Mahalanobis Distance for K-means clusterin 
The code performs the K-means clustering algorithm on two sets of points, "e1" and "e2", which represent the game assets (or any other thing one may want to imagine).

The algorithm is executed twice, once using the Euclidean distance (with the instance "kmeans_euclidean") and once using the Mahalanobis distance (with the instance "kmeans_mahalanobis"). Both instances are randomly initialized ("init='random'"), use the full clustering algorithm ("algorithm='full'"), and are run for only two iterations ("max_iter=2") for simplicity. The cluster results are stored in two label arrays ("labels_euclidean" and "labels_mahalanobis").

Finally, the code uses the matplotlib library to visualize the clustering results, plotting the points of "e1" as blue dots and the points of "e2" as yellow stars. In both plots, the points are colored based on their assigned cluster by the clustering algorithm, using a "viridis" colormap. The left plot shows the clustering results with the Euclidean distance, while the right plot shows the clustering results with the Mahalanobis distance.
