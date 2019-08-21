# This program reads in the data file "Clustering_Data_Sets.csv" 
# and inputs this data into the k-means clustering algorithm. 
# The output of the program is a scatter plot with all of the
# data points and the centroids for the 20 different clusters
# that are created. The coordinates of the 20 centroids are 
# also printed. 

# Step 1 - import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2 - import the dataset 
data = pd.read_csv("clustering_dataset.csv").values

# Step 3 - create an instance of the k-means algorithm with the number
# of clusters set to 20 and the intial centroids set to random 
k_mean = KMeans(n_clusters=20, init='random')

# Step 4 - fit the algorithm to the dataset 
k_mean.fit(data)

# Step 5 - get the centroids of the clusters and print them 
centroids = k_mean.cluster_centers_
print("\nCentroids of clusters: \n\n", centroids)

# Step 6 - get the labels of the clusters (in case you want to look at these)
labels = k_mean.labels_

# Step 7 - create a plot of the 20 clusters with their centroids and show it
plt.scatter(data[:,0], data[:,1], s = 20)
plt.scatter(centroids[:,0], centroids[:,1], c = 'r', s = 20)
plt.xlabel("x")
plt.ylabel("y")
plt.suptitle("20 clusters with centroids using k-means")
plt.show()


