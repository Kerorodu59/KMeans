import pandas as pd
import numpy as np

np.random.seed(0)

def dissimilarity(pointA,pointB):
    """
    Renvoie l'indice de dissimilarité 
    Plus il est élevé et plus la distance entre les deux points est grande
    """
    
    res = sum([1 for k in pointA.index if pointA[k] != pointB[k]])
    
    return res


def new_centroid(cluster):
    """
    Renvoie le centre d'un cluster catégorical
    """
    centroid = []
    
    for k in cluster.columns :
        values = dict((i,cluster[cluster[k]==i].shape[0]) for i in set(cluster[k])) 
        
        centroid.append(max(values,key=values.get))
        
    serie = pd.Series(data=centroid,index=cluster.columns)
        
    return serie

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
    print("dissimilarity : ",dissimilarity(Df.iloc[0],Df.iloc[1]))
    print("New centroid : ",new_centroid(Df.iloc[0:10]))
    print("Test de la fonction K-Mode : ")
    center, cluster, label = k_modes(2,Df,100)
    print(center)