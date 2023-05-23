import pandas as pd
import numpy as np

### Fonctions utiles pour KMeans
def random_centers(K,data):
    return [np.random.uniform(data.min(axis=0),data.max(axis=0)) for i in range(K)]

def mean_centroid(K,data):
    centroid = []
    for i in range(K):
        centroid.append(data[i*data.shape[0]//K:(i+1)*data.shape[0]//K].mean(axis=0))
    return centroid

#### Fonctions utiles pour KModes
def k_modes_center(K,data):
    values = []
    
    # On trie pour tous les attributs, les valeurs les plus courantes à celles qui se font plus rares
    for k in data :
        possibility = dict((i,len(data[data[k]==i])) for i in list(set(data[k])))
        values.append([k for k,v in sorted(possibility.items(),key=lambda x : x[1], reverse =True)])
    
    # On regarde les centres à prendre en fonction du tableau obtenue précédemment
    centers = []
    length_table = [len(liste) for liste in values]
    center = [i if i<length_table[i] else 0 for i in range(len(values))] 
    
    # Trouver les K points qui rassemblent tous les attributs apparaissant le plus souvent
    for i in range(K) :
        centers.append([values[att][center[att]] for att in range(len(values))])
        center = [center[k]+1 if center[k]+1 < length_table[k] else 0 for k in range(len(center)) ]

    centers = pd.DataFrame(data=centers,columns=data.columns)
        
    return centers    


def new_centroid_KMode(cluster):
    """
    Renvoie le centre d'un cluster catégorical
    """
    centroid = []
    
    for k in cluster.columns :
        values = dict((i,cluster[cluster[k]==i].shape[0]) for i in set(cluster[k])) 
        
        centroid.append(max(values,key=values.get))
        
    df = pd.DataFrame(data=np.array(centroid)[:,np.newaxis].T,columns=cluster.columns,index=[0])
        
    return df

def dissimilarity(pointA,pointB):
    """
    Renvoie l'indice de dissimilarité 
    Plus il est élevé et plus la distance entre les deux points est grande
    """
    
    res = sum([1 for k in pointA.index if pointA[k] != pointB[k]])
    
    return res


#### Fonction utiles pour KPrototypes

def new_centroid_KProt(cluster,numerical_keys,categorical_keys):
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
