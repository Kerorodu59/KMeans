import numbers
from sklearn.preprocessing import minmax_scale
from numpy.random import randint
import numpy as np
import pandas as pd
from math import *


def dissimilarity(pointA,pointB):
    """
    Renvoie l'indice de dissimilarité 
    Plus il est élevé et plus la distance entre les deux points est grande
    """
    
    res = sum([1 for k in pointA.index if pointA[k] != pointB[k]])
    
    return res



def new_centroid(cluster,numerical_keys,categorical_keys):
    """
    Renvoie le centre d'un cluster catégorical
    """
    # Calcul des coordonnées numériques du nouveau centre du cluster
    centroid = [cluster[k].mean() for k in numerical_keys]

    # Calcul des coordonnées catégoriques du nouveau centre du cluster    
    for k in categorical_keys :
        values = dict((i,cluster[cluster[k]==i].shape[0]) for i in set(cluster[k])) 
        # On prend la valeur qui apparait le plus souvent comme centre
        centroid.append(max(values,key=values.get))
        
    serie = pd.DataFrame(data=[centroid],columns=numerical_keys+categorical_keys)
        
    return serie



def rand_index(labelTrue,labelPredicted):
    """
    Calcule le taux de représentativité de labelPredicted par rapport à labelTrue
    Plus la valeur est proche de 1 et plus les labels sont bien prédits
    """
    cluster = list(set(labelTrue))
    clusterTrue = [set() for i in range(len(cluster))]
    clusterPredicted = [set() for i in range(len(cluster))]
    
    for i in range(len(labelTrue)):
        clusterTrue[cluster.index(labelTrue[i])].add(i)
        clusterPredicted[cluster.index(labelPredicted[i])].add(i)
        
    rand_index = 0
    for i in clusterPredicted:
        max_similitude = 0
        index = 0
        for j in range(len(clusterTrue)):
            similitude = len(i.intersection(clusterTrue[j]))
            if max_similitude < similitude:
                max_similitude = similitude
                index = j
        rand_index+=max_similitude
        del(clusterTrue[index])
    
    return rand_index/len(labelTrue)



def Adjusted_Rand_Index(labelPredicted,labelTrue):
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



def Accuracy(predictedLabel,trueLabel):
    """
    Calcul le taux de labels étant vrais
    """
    return sum([1 for i in range(len(predictedLabel)) if predictedLabel[i]==trueLabel[i]])/len(predictedLabel)



def KPrototype(K,data,max_iter,gamma):
    """
    
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
        new_centers= [new_centroid(cluster[i],number_keys,categorical_keys) for i in range(K)]
        new_centers = pd.concat(new_centers,axis=0)
        
        if (np.array_equal(new_centers,centers)): 
            not_centers_same = False
        else:
            centers=new_centers
        
    return (new_centers,cluster,label) 





if __name__ == '__main__':
    Df = pd.read_csv("Données/bank.csv",sep=";")
    ncenters,cluster,label = KPrototype(2,Df,1000,1)