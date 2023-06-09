"""
Fonctions intermédiaires de KModes, KMeans et KPrototype
"""

__authors__ = ("Audrey")
__contact__ = ("audrey.bilon.etu@univ-lille.fr")
__copyright__ = "CRISTAL"
__date__ = "2023-05-02"

import pandas as pd
import numpy as np
from math import *
import numbers

### Fonctions utiles pour KMeans
def random_centers(K,data):
    """renvoie K centres choisis aléatoirement tel que chaque attribut soit dans 
    l'espace réel du champ choisi

    Args:
        K (int): Nombre de centres voulus
        data (list(list(float))): tableau contenant tous les échantillons (qui sont sous 
        forme de tableau de nombre)

    Returns:
        list(list(float)): liste des centres (sous forme de liste d'échantillons)
    """
    return [np.random.uniform(data.min(axis=0),data.max(axis=0)) for i in range(K)]

def mean_centroid(K,data):
    """renvoie K centres tels que chaque centre est la moyenne de parties composées d'échantillons
    choisis aléatoirement dans data

    Args:
        K (int): Nombre de centres voulus
        data (list(list(float))): tableau contenant tous les échantillons (qui sont sous 
        forme de tableau de nombre)

    Returns:
        list(list(float)): liste des centres (sous forme de liste d'échantillons)
    """
    centroid = []
    for i in range(K):
        centroid.append(data[i*data.shape[0]//K:(i+1)*data.shape[0]//K].mean(axis=0))
    return centroid

#### Fonctions utiles pour KModes
def k_modes_center(K,data):
    """Renvoie les K centres initiaux de K_Modes

    Args:
        K (int): nombre de centres souhaités
        data (DataFrame): dataframe contenant les échantillons

    Returns:
        DataFrame: dataframe contenant les centres calculés
    """
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
    """Calcule les nouveaux centres à partir des clusters
    
    Args:
        cluster (dict(int,DataFrame)): dictionnaire représentant le cluster par son label(int) 
        et les éléments qui compose le cluster (sa valeur associée)

    Returns:
        DataFrame: DataFrame des nouveaux centres 
    """
    centroid = []
    
    for k in cluster.columns :
        values = dict((i,cluster[cluster[k]==i].shape[0]) for i in set(cluster[k])) 
        
        centroid.append(max(values,key=values.get))
        
    df = pd.DataFrame(data=np.array(centroid)[:,np.newaxis].T,columns=cluster.columns,index=[0])
        
    return df

def dissimilarity(pointA,pointB):
    """Renvoie l'indice de dissimilarité 
    Plus il est élevé et plus la distance entre les deux points est grande
    
    Args:
        pointA (Serie): Un échantillon à comparer
        pointB (Serie): l'autre échantillon à comparer

    Returns:
        int : distance entre les deux échantillons
    """
    
    res = sum([1 for k in pointA.index if pointA[k] != pointB[k]])
    
    return res


#### Fonction utiles pour KPrototypes

def preparation_donnees(Df):
    """ Permet la préparation des données d'un DataFrame numériques et catégoriques
    La fonction enlève les NaN présents dans le tableau pour les valeurs numériques
    
    Args:
        Df (DataFrame) : DataFrame à modifier
    
    Returns:
        DataFrame : DataFrame modifié
    """
    numerical = []
    
    for k in Df :
        if isinstance(Df.iloc[0][k],numbers.Real) :
            numerical.append(k)

    for i in range(len(Df)):
        if (Df.iloc[i].isna().any()):
            for k in numerical :
                if pd.isna(Df.iloc[i][k]):
                    Df.loc[i,k] = Df[k].mean()
    return Df

def new_centroid_KProt(cluster,numerical_keys,categorical_keys):
    """Renvoie les nouveaux centres calculé en fonction d'un cluster mixte
    
    Args:
        cluster (dict(int,DataFrame)): dictionnaire représentant le cluster par son label(int) 
        et les éléments qui compose le cluster (sa valeur associée)
        numerical_keys (list(str)) : liste des noms de champs de données numériques
        categorical_keys (list(str)) : liste des noms de champs de données catégoriques

    Returns:
        DataFrame: DataFrame des nouveaux centres 
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

def gamma_distance(pointA,pointB,gamma,numericalKeys,categoricalKeys) :
    """Calcule la distance entre deux points qui sont numériques et catégoriques

    Args:
        pointA (Series): premier point à comparer
        pointB (Series): second point à comparer
        gamma (float): coefficient des points catégoriques
        numericalKeys (list(str)): liste des noms des attributs numériques
        categoricalKeys (list(str)): liste des noms des attributs catégoriques

    Returns:
        float : distance entre les points A et B
    """
    res = np.linalg.norm(np.array(pointA[numericalKeys])-np.array(pointB[numericalKeys]))+gamma*dissimilarity(pointB[categoricalKeys],pointA[categoricalKeys])
    return res

