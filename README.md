Les dossiers de santé électroniques sont importants à étudier en milieu hospitalier pour différencier les patients en fonction de leurs attributs,
les regrouper en cluster et adapter les traitements de chaque patient en fonction du cluster où ils sont. 
 
Pour cela, nous utilisons des algorithmes de clustering. Le problème que nous rencontrons est que les données médicales sont mixtes et que parmi les algorithmes officiels, 
il n'en existe qu'un traitant de clustering. 
Ce dernier ne s'occupe que de données numériques. 
Notre objectif ici est de coder un algorithme capable de s'occuper aussi bien des données catégorielles que numériques et de pouvoir l'utiliser pour prédire dans quel 
cluster sera chaque patient. 

Pour cela, nous coderons $K$-Means qui gère les données numériques, $K$-Modes les données catégorielles et $K$-Prototype qui s'occupera des deux. 

Pour utiliser ces derniers il faut les importer.
Les fichiers contenant les codes se trouvent dans le dossier Code.
Les fonctions sont implémentés dans les fichiers contenant leurs noms.
Les métriques sont situés dans le fichier Metriques.py