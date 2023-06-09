"""
KMeans, fonction qui permet de calculer les clusters de données numériques uniquement
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
    Renvoie les informations concernant les K clusters obtenus à partie du dataset numérique
    
    Args:
        K (int) : le nombre de clusters que nous voulons avoir 
        data (list(list(float))) : Dataset représenté par une liste d'échantillons (sous liste de nombres)
        max_iter (int) : nombre qui indique le nombre d'itérations maximum que la fonction peut réaliser.
        function_initial_centers ((int,list(list(float)))->list(list(float))) : fonction qui permet de calculer 
        les centres initiaux

    Returns:
        (list(list(float)),dict(int,list(list(float))),list(int)) :
        les éléments retournés dans le tuple sont les suivants :
            - liste des centres (sous forme de liste de floattant)
            - dictionnaire des clusters avec leurs numéros associés 
            - labels de chaque échantillons contenus dans une liste
        
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