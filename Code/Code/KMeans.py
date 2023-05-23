"""
Fonction calculant les informations relatives au k-means
Usage:
======
    python KMeans.py 
"""

__authors__ = ("Audrey")
__contact__ = ("audrey.bilon.etu@univ-lille.fr")
__copyright__ = "CRISTAL"
__date__ = "2023-04-19"

import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from Metriques import *
from Outils import *

np.random.seed(0)
iris = datasets.load_iris()
X = iris["data"]


def k_means(K,data,max_iter,function_initial_centers):
    """
    Renvoie un tuple contenant:
    - la liste des centres de chaque cluster
    - un dictionnaire contenant les clusters (sous forme de liste) associés à leurs numéros de cluster 
    - une liste (label) contenant le numéro du cluster auquel est associé chaque point.
    """
    centroid = []
    cluster_not_the_same = True
    c = 0
    
    # Calcul du centre aléatoire    
    centroid = function_initial_centers(K,data)

    # Tant que la condition d'arrêt n'est pas respecté 
    while(c!=max_iter and cluster_not_the_same):
        # Remise à zéro de clusters
        cluster = dict((i,[]) for i in range(K))
        c += 1
        label = []
        
        #Assignation des points en fonction de leurs clusters
        for point in data:
            # Aucun cluster n'est assigné au point
            clust = (None,float('inf')) # (numéro du cluster,distance au cluster)
            #Choix du cluster
            for i in range(K):
                dist = np.linalg.norm(point-centroid[i])
                #Si le centre du cluster est plus proche que ceux étudiés avant nous le remplaçons
                if dist<clust[1]:
                    clust = (i,dist)
            
            cluster[clust[0]].append(point)
            label.append(clust[0])
        
        # Recalcule les centres
        new_centroid = [np.mean(cluster[i],axis=0) for i in range(K)]
        
        # On regarde si les centres sont toujours les mêmes
        if (np.array_equal(new_centroid,centroid)): 
            cluster_not_the_same= False
        else:
            centroid=new_centroid
    
    return (centroid,cluster,label)


if __name__ == '__main__':
    center,cluster,label = k_means(2,X,500)
    kmeans = KMeans(n_clusters=2,random_state=0,max_iter=500).fit(X)
    print("Intra-variance cluster of the function : ",IVC_Kmeans(center,cluster))
    print("Rand Index of the function :", rand_index(kmeans.labels_,label))