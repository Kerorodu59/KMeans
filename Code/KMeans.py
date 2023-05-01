"""
Fonction calculant le k-mean des données
Usage:
======
    python nom_de_ce_super_script.py argument1 argument2

    argument1: un entier signifiant un truc
    argument2: une chaîne de caractères décrivant un bidule
"""

__authors__ = ("Audrey")
__contact__ = ("audrey.bilon.etu@univ-lille.fr")
__copyright__ = "CRISTAL"
__date__ = "2023-04-19"

import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(0)
iris = datasets.load_iris()
X = iris["data"]


def k_means(n_clusters,data,max_iter):
    """
    Renvoie un tuple contenant:
    - la liste des centres de chaque cluster
    - label contenant les clusters correspondants de chaque points
    chaque liste est associé à la valeur du numéro du cluster auquel il correspond 
    """
    centroid = []
    cluster_not_the_same = True
    c = 0
    
    # Calcul du centre aléatoire    
    centroid = [np.random.uniform(data.min(axis=0),data.max(axis=0)) for i in range(n_clusters)]

    # Tant que la condition d'arrêt n'est pas respecté 
    while(c!=max_iter and cluster_not_the_same):
        # Remise à zéro de clusters
        cluster = dict((i,[]) for i in range(n_clusters))
        c += 1
        label = []
        
        #Assignation des points en fonction de leurs clusters
        for point in data:
            # Aucun cluster n'est assigné au point
            clust = (None,float('inf')) # (numéro du cluster,distance au cluster)
            #Choix du cluster
            for i in range(n_clusters):
                dist = np.linalg.norm(point-centroid[i])
                #Si le centre du cluster est plus proche que ceux étudiés avant nous le remplaçons
                if dist<clust[1]:
                    clust = (i,dist)
            
            cluster[clust[0]].append(point)
            label.append(clust[0])
        
        # Recalcule les centres
        new_centroid = [np.mean(cluster[i],axis=0) for i in range(n_clusters)]
        
        # On regarde si les centres sont toujours les mêmes
        if (np.array_equal(new_centroid,centroid)): 
            cluster_not_the_same= False
        else:
            centroid=new_centroid
    
    return (centroid,cluster,label)


def intra_cluster_variance(centroid: list[list[float]],cluster: dict[int,list[float]]) -> float:
    """
    Calcul l'intra-cluster-variance
    """
    res = 0
    for i in range(len(centroid)):
        distance = np.array(cluster[i]-centroid[i])
        res += np.sum(np.linalg.norm(distance,axis=1)**2)
    return res

def rand_index(labelTrue,labelPredicted):
    """
    Calcule le taux de représentativité de labelPredicted par rapport à labelTrue
    Plus la valeur est proche de 1 et plus les labels sont bien prédits
    """
    cluster = list(set(labelTrue))
    cluster1 = [0 for i in range(len(cluster))]
    cluster2 = [0 for i in range(len(cluster))]
    
    for i in range(len(labelTrue)):
        
        for j in range(len(cluster)) :
            if cluster[j] == labelTrue[i]:
                cluster1[j]+= 1
            if cluster[j] == labelPredicted[i]:
                cluster2[j]+= 1
        
    cluster1.sort()
    cluster2.sort()
    
    cluster = np.min(np.array([cluster1,cluster2]),axis=0)
    
    return (np.sum(cluster))/len(labelTrue)  


if __name__ == '__main__':
    center,cluster,label = k_means(2,X,500)
    kmeans = KMeans(n_clusters=2,random_state=0,max_iter=500).fit(X)
    print("Intra-variance cluster of the function : ",intra_cluster_variance(center,cluster))
    print("Rand Index of the function :", rand_index(kmeans.labels_,label))