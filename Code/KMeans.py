def k_means(n_clusters,data,max_iter):
    """
    Renvoie un tuple contenant:
    - la liste des centres de chaque cluster
    - un dictionnaire avec un ensemble des points contenus dans chaque cluster
    chaque liste est associé à la valeur du numéro du cluster auquel il correspond 
    """
    centroid = []
    cluster = {}
    cluster_not_the_same = True
    c = 0
       
    for i in range(n_clusters):
        centroid.append(np.random.uniform(data.min(axis=0),data.max(axis=0)))    
    
    # Tant que la condition d'arrêt n'est pas respecté 
    while(c!=max_iter and cluster_not_the_same):
        c += 1
        new_centroid=[]
        
        for i in range(n_clusters):
            cluster[i]=[]
        
        #Assignation des points en fonction de leurs clusters
        for point in data:
            # Aucun cluster n'est assigné au point
            clust = (None,float('inf')) # (numéro du cluster,distance au cluster)
            
            for i in range(n_clusters):
                dist = np.linalg.norm(point-centroid[i])
                #Si le centre du cluster est plus proche que ceux étudiés avant nous le remplaçons
                if dist<clust[1]:
                    clust = (i,dist)
            
            cluster[clust[0]].append(point)
        
        # Recalcule les centres
        for i in range(n_clusters):
            new_centroid.append(np.mean(cluster[i],axis=0))
        
        # On regarde si les centres sont toujours les mêmes
        if (np.array_equal(new_centroid,centroid)): 
            cluster_not_the_same= False
        else:
            centroid=new_centroid
    
    return (centroid,cluster)