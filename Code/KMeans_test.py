"""
Fonction testant les informations calculés dans le fichier KMeans.py 
Usage:
======
    python KMeans_test.py 
"""

__authors__ = ("Audrey")
__contact__ = ("audrey.bilon.etu@univ-lille.fr")
__copyright__ = "CRISTAL"
__date__ = "2023-04-19"

from KMeans import *
import unittest


np.random.seed(0)
kmeans = KMeans(n_clusters=3,random_state=0,max_iter=500).fit(X)
center,cluster,label = k_means(3,X,500)


# Montre que le KMeans implémenté fonctionne comme le KMeans de la librairie
class test_kmeans_functions(unittest.TestCase):
    
    def test_kmeans_centers(self):
        self.assertTrue(np.allclose(np.sort(kmeans.cluster_centers_,axis=0),np.sort(center,axis=0)))
        
    def test_kmeans_cluster(self):
        """
        Comme les clusters des 2 sont identiques, ce sera la même chose pour ICV
        """
        self.assertEqual(rand_index(kmeans.labels_,label),1)
        
        
if __name__ == '__main__':
    unittest.main()