"""
Sources :  
- https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875
- https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
- https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25

Videos :  
- https://www.youtube.com/watch?v=8nOIB5LDiWo  
Partie 1
- https://www.youtube.com/watch?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&time_continue=114&v=H4JSN_99kig&feature=emb_logo  
Partie 2  
https://www.youtube.com/watch?v=HRoeYblYhkg&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=38
"""

"""
Algorithm :
1. Randomly initialize K centroids
2. Assign each data point to the centroid it is closest to
3. Recompute the centroid based off of the mean of the poinst assigned to the original centroid
4. Repeat steps two and three until we the stopping creteria has been reached

Optimizing :
- Euclidean (squared) distance
- Sum of squared distances of each point assigned to its cluster
- r is an indicator function for whether th epoint is assigned to cluster k
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1.5,1.8], [5, 8], [8, 8], [1, 0.6],[9, 11]])

#clf = KMeans(n_clusters=8)
#clf.fit(X)

#centroids = clf.cluster_centers_
#labels = clf.labels_

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

colors = 10*['g','r','c','b','k']

class K_means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self,data):

        sef.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i]
# for every iterations we are going clearing out the classification because :
# centroid, always going to have a 0 & 1 index or key
# but the value is going to change so you're always going to have the same number of centroids
# but here for the classifcations, that's going to change every time the centroid changes
# So we emplty tat out and redo the classification every single time
        for i in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
# X should be the data               
            for featureset in X:
# Creating a list that is being populated with k number of values because for centroid
# and self centroids that contrains K numbers of centroids so 0 and 1 so the zeoeth index
# in this list will be the 0 basically the distance to the zero(th) with centroid
# And then the first(th) element will be the distance from that data point to the centroid one 
                distances = [np.linalg.norm(featureset-self.centroids[centroids]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
# we have to do this because object inheritance
# we are trying to compare the two centroids so we can find how much they have changed
# So we can use the tolerance value 
            prev_centroids = dict(self.centroids)
#
            for classification in self.classifications:
                pass
# it is going to take that basically array value and it's going to take the average of all of the classifications that we have 
# So, I am going to take the average dataset
# It is going to find the centroid for all of the values that are of that previous centroid classification
# This is findind the mean of all the features for any given class
# And then it's remaking that centroid, redefines the centroid now

                #self.centroids[classfication] = np.average(self.classifications[classification], axis=0)
            optimized = True
        
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
    
    def predict(self,data):
        pass
    
 