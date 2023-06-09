"""
KPrototype, Fonction calculant les clusters de données catégoriques et numériques.
"""

__authors__ = ("Audrey")
__contact__ = ("audrey.bilon.etu@univ-lille.fr")
__copyright__ = "CRISTAL"
__date__ = "2023-05-02"


import numbers
from sklearn.preprocessing import minmax_scale
from numpy.random import randint
import numpy as np
import pandas as pd
from math import *
from Outils import *
from Metriques import *


def KPrototype(K,data,max_iter,gamma):
    """Renvoie des informations concernant les clusters formés par l'algorithme KPrototype

    Args:
        K (int): nombre de clusters voulus
        data (DataFrame): datasets étudié
        max_iter (int): nombre d'itération réalisable par l'algorithme
        gamma (float) : poids des données catégorielles

    Returns:
        (DataFrame,dict(int,DataFrame),list(int)) :
        les éléments retournés dans le tuple sont les suivants :
            - DataFrame comportant les centres des clusters
            - dictionnaire des clusters avec leurs numéros associés 
            - labels de chaque échantillons contenus dans une liste
    """
    # Tri des clés ayant des valeurs catégoriques et numériques
    number_keys = []
    categorical_keys = []
    
    for k in data :
        if isinstance(data.iloc[0][k],numbers.Real) :
            number_keys.append(k)
        else:
            categorical_keys.append(k)
    
    # Préparation des 2 ensembles de données
    categorical = data[categorical_keys]
    numerical = pd.DataFrame(minmax_scale(np.array(list(list(data[k]) for k in number_keys))).T,columns=number_keys)
    
    dataset = pd.concat([numerical,categorical],axis=1)
    centers = dataset.iloc[randint(0,data.shape[0],K)] 
    
    i = 0
    not_centers_same = True
    
    while (i!=max_iter and not_centers_same):
        label = []
        i+=1
        
        # Remise à zéro des clusters
        cluster = dict((i,pd.DataFrame(columns=(number_keys+categorical_keys))) for i in range(K))
        
        for d in range(data.shape[0]):
            value_clusters = []
            for i in range(K):
                distance = np.linalg.norm(np.array(numerical.iloc[d])-np.array(centers.iloc[i][number_keys]))+gamma*dissimilarity(centers.iloc[i][categorical_keys],categorical.iloc[d])
                value_clusters.append((distance,i))
            
            minimum_center = min(value_clusters)
            cluster[minimum_center[1]].loc[len(cluster[minimum_center[1]])] = dataset.iloc[d]
            label.append(minimum_center[1])
    
        
        # Recalcule les centres
        new_centers= [new_centroid_KProt(cluster[i],number_keys,categorical_keys) for i in range(K)]
        new_centers = pd.concat(new_centers,axis=0)
        
        if (np.array_equal(new_centers,centers)): 
            not_centers_same = False
        else:
            centers=new_centers
        
    return (new_centers,cluster,label) 



if __name__ == '__main__':
    Df = pd.read_csv("Données/bank.csv",sep=";")
    ncenters,cluster,label = KPrototype(2,Df,1000,1)