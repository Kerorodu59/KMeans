"""
KMode, Fonction calculant les clusters de donnée numérique
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


def k_modes(K,data,max_iter):
    """Renvoie les informations relatives aux clusters obtenu grâce à KModes

    Args:
        K (int): nombre de clusters voulus
        data (DataFrame): datasets étudié
        max_iter (int): nombre d'itération réalisable par l'algorithme

    Returns:
        (DataFrame,dict(int,DataFrame),list(int)) :
        les éléments retournés dans le tuple sont les suivants :
            - DataFrame comportant les centres des clusters
            - dictionnaire des clusters avec leurs numéros associés 
            - labels de chaque échantillons contenus dans une liste
    """
    
    #Initialisation des centres choisis aléatoirement
    centers= k_modes_center(K,data)
    not_same_as_before = True
    i=0
    
    while (not_same_as_before and i!=max_iter):
        
        label = []
        i+=1
        # Remise à zéro de clusters
        cluster = dict((i,pd.DataFrame(columns=data.columns)) for i in range(K))
        
        for d in range(data.shape[0]):
            
            minimum_center = min([(dissimilarity(centers.iloc[i],data.iloc[d]),i) for i in range(K)])
            
            cluster[minimum_center[1]].loc[len(cluster[minimum_center[1]])] = data.iloc[d]
            label.append(minimum_center[1])
            
        # Recalcule les centres
        new_center = pd.concat([new_centroid_KMode(cluster[i]) for i in range(K)],ignore_index=True)
        
        # On regarde si les centres sont toujours les mêmes
        if (new_center.equals(centers)): 
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