"""
Fonction calculant les informations relatives au k-mode
Usage:
======
    python KMode.py 
"""

__authors__ = ("Audrey")
__contact__ = ("audrey.bilon.etu@univ-lille.fr")
__copyright__ = "CRISTAL"
__date__ = "2023-05-02"

import pandas as pd
import numpy as np
from Metriques import *
from Outils import *

np.random.seed(0)


def k_modes(n_clusters,data,max_iter):
    
    #Initialisation des centres choisis aléatoirement
    centers= [data.iloc[np.random.randint(0,len(data)-1)] for i in range(n_clusters)] 
    not_same_as_before = True
    i=0

    while (not_same_as_before and i!=max_iter):
        
        label = []
        i+=1
        # Remise à zéro de clusters
        cluster = dict((i,pd.DataFrame(columns=data.columns)) for i in range(n_clusters))
        
        for d in range(data.shape[0]):
            
            # Recherche le centre le plus proche du point
            minimum_center = min([(dissimilarity(centers[i],data.iloc[d]),i) for i in range(n_clusters)])
            # L'associe à son cluster et ajoute le label du point à la liste des labels
            cluster[minimum_center[1]].loc[len(cluster[minimum_center[1]])] = data.iloc[d]
            label.append(minimum_center[1])
            
        # Recalcule les centres
        new_center = []
        for i in range(n_clusters):
            new_center.append(new_centroid(cluster[i]))
        
        # On regarde si les centres sont toujours les mêmes
        if (np.array_equal(new_center,centers)): 
            not_same_as_before = False
        else:
            centers=new_center
    
    return (centers,cluster,label)


if __name__ == '__main__':
    Df = pd.read_csv("Données/bank.csv",sep=";")
    Df = Df[['age','job', 'marital', 'education', 'default', 'housing', 
'loan','contact','month','poutcome']]
    print("Test de la fonction K-Mode : ")
    center2, cluster2, label2 = k_modes(2,Df,100)
    center3, cluster3, label3 = k_modes(3,Df,100)
    print("L'IVC avec K=2 des données est : ", IVC_KModes(center2,cluster2))
    print("L'IVC avec K=3 des données est : ", IVC_KModes(center3,cluster3))
    print("K=3 nous donne un cluster avec une meilleure IVC")