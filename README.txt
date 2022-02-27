Les programmes à exécuter sont dans le repertoire scripts.
Le job de l'entrainement : training.py
Le job de sauvegarde : sauvegarde.py (un repertoire models doit être créé à la racine)
le job de prédiction : testing.py

Le fichier utils.py contient les fonctions dont nous avons besoin pour le prepocessing.
Pour gérer les valeurs manquantes, nous avons supprimé les colonnes avec énormement de valeurs manquantes, 
et supprimé les lignes contenants les Na restants. (On aurait pu penser à une méthode d'imputation tel que le remplacement par le mode 
par exemple pour les variables qualitatives, et la moyenne pour les variables quantitatives, mais reflechir à la meilleur méthode d'imputation nous prendrait trop de temps, car cela demande plus de connaissances métiers.

Nous avons choisi l'encodage ordinal, qui n'est pas forcément optimal, mais vu que l'encodage one hot augmenterait énormement la dimension de notre jeu, nous avons préféré celui qui optimiserait notre temps de calcul.


Le problème consistait à faire une classification supervisée multiclasse de la variable cible correct_fedas_code dont les données sont très déséquilibrées.

Le fichier training2 ne doit pas être exécuté, c'est juste une ouverture sur ce qui pourrait être fait pour améliorer le score.
Pour des contraintes liées à l'installation de certains packages, au temps de calcul(qui deviendra évidemment long après l'oversampling), et au deadline, Je n'ai pas poussé plus loin pour améliorer l'accuracy. 
J'ai obtenu 67,04% avec la méthode de boosting utilisée, à l'issue de ma division de mon jeu d'apprentissage en apprentissage-test.
