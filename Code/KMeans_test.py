from KMeans import *
import unittest
import types

np.random.seed(0)
kmeans = KMeans(n_clusters=3,random_state=0,max_iter=500).fit(X)
center,cluster,label = k_means(3,X,500)

class test_kmeans_functions(unittest.TestCase):
    
    def test_kmeans_centers(self):
        self.assertTrue(kmeans.cluster_centers_.sort()==center.sort())
        
    def test_kmeans_cluster(self):
        self.assertEqual(rand_index(kmeans.labels_,label),1)
        
        
if __name__ == '__main__':
    unittest.main()