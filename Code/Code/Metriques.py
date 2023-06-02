"""
Fonctions calculant la justesse des informations obtenues par les algorithmes KMeans, KModes et KPrototype 
"""

__authors__ = ("Audrey")
__contact__ = ("audrey.bilon.etu@univ-lille.fr")
__copyright__ = "CRISTAL"
__date__ = "2023-05-02"

import numpy as np
from math import *

# Métrique de KMeans
def IVC_Kmeans(centroid,cluster):
    """Calcul l'intra-cluster variance pour KMeans
    
    Args:
        centroid (list(list(float))) : liste représentant les centres sous forme de liste d'entiers.
        cluster (dict(int,list(list(float)))): dictionnaire représentant le cluster par son label(int) 
        et les éléments qui compose le cluster (sa valeur associée) sous forme de liste d'échantillons (list(int))

    Returns:
        int : intra cluster variance
    """
    res = 0
    for i in range(len(centroid)):
        distance = np.array(cluster[i]-centroid[i])
        res += np.sum(np.linalg.norm(distance,axis=1)**2)
    return res


# Métrique de KModes
def IVC_KModes(centroid,cluster):
    """
    Calcul l'intra-cluster-variance pour KModes
    
    Args:
        centroid (DataFrame) : DataFrame des centres
        cluster (dict(int,DataFrame)): dictionnaire représentant le cluster par son label(int) 
        et les éléments qui compose le cluster (sa valeur associée)

    Returns:
        int : intra cluster variance 
    """
    res = 0
    for i in range(len(centroid)):
        for j in range(len(cluster[i])) :
            res += dissimilarity(cluster[i].loc[j],centroid.iloc[i])**2
    return res

def dissimilarity(pointA,pointB):
    """
    Renvoie l'indice de dissimilarité 
    Plus il est élevé et plus la distance entre les deux points est grande
    (pour plus de documentation regarder Outils.py)
    """
    
    res = sum([1 for k in pointA.index if pointA[k] != pointB[k]])
    
    return res

# Métriques utilisables pour toutes les fonctions 
def rand_index(labelTrue,labelPredicted):
    """Renvoie le rand index calculé 

    Args:
        labelTrue (list(int)): liste des labels de chaque point lorsque le cluster est bien calculé
        labelPredicted (list(int)): liste des labels de chaque point lorsque le cluster est calculé avec 
        l'algorithme

    Returns:
        float: rand index, plus il est proche de 1 et plus les résultats obtenus par l'algorithme sont bons.
    """
    # Initialisation de la table de groupement et séparation des données
    table = np.zeros((2,2))
    
    #Parcours toutes les combinaisons de paires de points par partitionnement
    for i in range(len(labelTrue)):
        for j in range(i+1,len(labelTrue)):
            
            if labelTrue[i]==labelTrue[j]:
                if labelPredicted[i]==labelPredicted[j]:
                    table[0,0]+=1
                else:
                    table[0,1]+=1
            else:
                if labelPredicted[i]==labelPredicted[j]:
                    table[1,0]+=1
                else:
                    table[1,1]+=1
                    
    return (table[0,0]+table[1,1])/table.sum()   


def Accuracy(labelTrue,labelPredicted):
    """
    Calcule le taux de représentativité de labelPredicted par rapport à labelTrue
    Plus la valeur est proche de 1 et plus les labels sont bien prédits

    Args:
        labelTrue (list(int)): liste des labels de chaque point lorsque le cluster est bien calculé
        labelPredicted (list(int)): liste des labels de chaque point lorsque le cluster est calculé avec 
        l'algorithme

    Returns:
        float: plus il est proche de 1 et plus les résultats obtenus par l'algorithme sont bons.   
    """
    clusterT = list(set(labelTrue))
    clusterP = list(set(labelPredicted))
    clusterTrue = [set() for i in range(len(clusterT))]
    clusterPredicted = [set() for i in range(len(clusterP))]
    
    for i in range(len(labelTrue)):
        clusterTrue[clusterT.index(labelTrue[i])].add(i)
        clusterPredicted[clusterP.index(labelPredicted[i])].add(i)
        
    acc = 0
    for i in clusterPredicted:
        max_similitude = 0
        for j in range(len(clusterTrue)):
            index = 0
            similitude = len(i.intersection(clusterTrue[j]))
            if max_similitude < similitude:
                max_similitude = similitude
                index = j
        acc+=max_similitude
        del(clusterTrue[index])
    
    return acc/len(labelTrue)


def Adjusted_Rand_Index(labelTrue,labelPredicted):
    """Calcul l'Adjusted Rand Index des deux partitions

    Args:
        labelTrue (list(int)): liste des labels de chaque point lorsque le cluster est bien calculé
        labelPredicted (list(int)): liste des labels de chaque point lorsque le cluster est calculé avec 
        l'algorithme

    Returns:
        float: plus adjusted rand index est proche de 1 et plus les résultats obtenus par l'algorithme sont exactes.
    
    """
    #Initialisation de la "contingency table"
    clusterPredicted, clusterTrue = list(set(labelPredicted)),list(set(labelTrue))
    X = np.zeros((len(clusterTrue),len(clusterPredicted)))
    
    # Remplissage du tableau de contingeance
    for i in range(len(labelPredicted)):
        t=clusterTrue.index(labelTrue[i])
        p=clusterPredicted.index(labelPredicted[i])
        X[t,p] += 1
    
    X = X.astype(int)
    A = X.sum(axis=1)
    B = X.sum(axis=0)        
    
    sommeB = sum([comb(n,2) for n in B])
    E = (sum([comb(n,2)*sommeB for n in A])/comb(len(labelTrue),2))
    RI = sum([comb(n,2) for ligne in X for n in ligne])
    max_RI = 0.5 * (sum([comb(n,2) for n in A])+sommeB)
    
    return (RI-E)/(max_RI-E)