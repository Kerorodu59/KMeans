from KMeans import *
import unittest
import types

kmeans = KMeans(n_clusters=4,random_state=0,max_iter=500).fit(X)
center,cluster,label = k_means(4,X,500)

class test_kmeans_functions(unittest.TestCase):
    
    def test_kmeans_centers(self):
        self.assertEquals(kmeans.cluster_centers_.sort(),center.sort())
        
    def test_kmeans_cluster(self):
        self.assertEqual(rand_index(kmeans.labels_,label),1)
        
        
if __name__ == '__main__':
    unittest.main()