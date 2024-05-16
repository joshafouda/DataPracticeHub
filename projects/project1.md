# Projet : Service Machine Learning de prediction du defaut de paiement

## Description
Ce projet a pour objectif de construire une application web avec Streamlit pour prédire le défaut de paiement des clients de cartes de crédit. Nous utiliserons le jeu de données "Default of Credit Card Clients" disponible sur le site UCI Machine Learning Repository. Ce jeu de données contient des informations démographiques et de paiement pour 30 000 clients de cartes de crédit à Taïwan. Notre tâche sera de développer un modèle de machine learning capable de prédire si un client fera défaut sur son paiement le mois suivant.

## Image
imgs/project1/project1.png


## Instructions
1. **Chargement des données** : Téléchargez le jeu de données depuis [cette page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) et chargez-le dans votre environnement de travail.
2. **Préparation des données** : Effectuez un nettoyage des données, gérez les valeurs manquantes, et effectuez toute transformation nécessaire.
3. **Exploration des données** : Réalisez une analyse exploratoire des données pour comprendre les relations entre les différentes variables.
4. **Construction du modèle** : Séparez les données en ensembles d'entraînement et de test, puis entraînez plusieurs modèles de machine learning (par exemple, régression logistique, arbres de décision, etc.).
5. **Évaluation du modèle** : Évaluez les performances des modèles à l'aide de métriques appropriées comme l'exactitude, le rappel et la précision.
6. **Déploiement de l'application** : Utilisez Streamlit pour créer une application web interactive permettant de prédire le défaut de paiement en fonction des informations fournies par l'utilisateur.
7. **Documentation et partage** : Documentez le processus de développement et partagez votre application.

## Resources
- [Jeu de données "Default of Credit Card Clients"](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Tutoriels de machine learning](https://www.kaggle.com/learn/overview)
- [Documentation Scikit-Learn](https://scikit-learn.org/stable/documentation.html)

## Execution du Projet

Pour ce projet, vous pouvez travailler dans l'environnement de developpement Python de votre choix. Nous recommandons l'utilisation de Visual Studio Code (VSC).

L'exploration des donnees ainsi que les experiences ML peuvent etre realisees dans un notebook (fichier *credit_card_default.ipynb* par exemple).

- **Importation des Librairies** :

```python
# Librairies
import pandas as pd
import numpy as np
import requests
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, cross_val_score,
    RandomizedSearchCV, cross_validate,
    StratifiedKFold
)
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

from credit_card_default_utils import * # Le contenu du module credit_card_default_utils.py sera devoiler un peu plus bas
```


- **Introduction** :

L'apprentissage automatique (machine learning) peut être utilisé pour identifier les défauts de crédit, ce qui est crucial pour les institutions financières, les prêteurs et les établissements de crédit afin de minimiser les risques et les pertes associés aux prêts. Voici les étapes générales pour appliquer l'apprentissage automatique à la détection des défauts de crédit :

1. Collecte de données :

Rassemblez des données historiques sur les prêts, y compris les caractéristiques du demandeur, les antécédents de crédit, les renseignements financiers, les détails du prêt et les résultats (par exemple, si le prêt a été remboursé ou en défaut).

2. Prétraitement des données :

Nettoyez les données en traitant les valeurs manquantes, les valeurs aberrantes et en normalisant les caractéristiques.
Encodez les variables catégorielles en utilisant des techniques comme l'encodage one-hot ou la représentation vectorielle.

3. Séparation des données :

Divisez les données en ensembles d'entraînement, de validation et de test pour évaluer la performance du modèle.

4. Sélection du modèle :

Choisissez un algorithme d'apprentissage automatique adapté à la tâche. Les méthodes couramment utilisées pour la détection des défauts de crédit incluent la régression logistique, les arbres de décision, les forêts aléatoires, les machines à vecteurs de support (SVM) et les réseaux de neurones.

5. Entraînement du modèle :

Entraînez le modèle sur l'ensemble d'entraînement en utilisant les données historiques pour qu'il puisse apprendre à distinguer les emprunteurs à risque de ceux à faible risque.

6. Évaluation du modèle :

Utilisez l'ensemble de validation pour ajuster les hyperparamètres du modèle et évaluer ses performances à l'aide de mesures telles que la précision, le rappel, la F1-score et la courbe ROC-AUC.

7. Optimisation du modèle :

Optimisez le modèle en ajustant ses hyperparamètres, en appliquant des techniques de régularisation et en évaluant différentes stratégies de gestion de déséquilibre de classe (s'il y a un déséquilibre significatif entre les défauts de crédit et les remboursements).

8. Test du modèle :

Évaluez finalement la performance du modèle sur l'ensemble de test pour obtenir une estimation de sa capacité à généraliser sur de nouvelles données.

9. Déploiement du modèle :

Une fois que le modèle atteint des performances satisfaisantes, il peut être déployé pour automatiser le processus de décision de crédit ou servir de support à la prise de décision humaine.

10. Surveillance continue :

Surveillez en continu les performances du modèle après son déploiement, car les comportements des emprunteurs et les conditions économiques peuvent évoluer.

L'application de l'apprentissage automatique pour la détection des défauts de crédit peut permettre de prendre des décisions plus précises, de réduire les risques de crédit et d'optimiser le rendement du portefeuille de prêts. Il est important de noter que la gestion des risques et la conformité réglementaire doivent être prises en compte dans ce contexte, car la détection des défauts de crédit implique des considérations éthiques et légales.

À la fin de ce projet, vous serez familiarisé avec une approche réelle d'une tâche d'apprentissage automatique, depuis la collecte et le nettoyage des données jusqu'à la création et le réglage d'un classificateur. Un autre point à retenir est de comprendre l'approche générale des projets d'apprentissage automatique, qui peut ensuite être appliquée à de nombreuses tâches différentes, qu'il s'agisse de prédire le taux de désabonnement ou d'estimer le prix d'un nouveau bien immobilier dans un quartier.


- **Données** :

L’ensemble de données utilisé dans ce chapitre a été collecté dans une banque taïwanaise en octobre 2005. L’étude était motivée par le fait qu’à cette époque, de plus en plus de banques accordaient du crédit (en espèces ou par carte de crédit) à des clients consentants. En outre, de plus en plus de personnes, quelle que soit leur capacité de remboursement, ont accumulé des dettes importantes. Tout cela a conduit à des situations dans lesquelles certaines personnes n’ont pas pu rembourser leurs dettes impayées. En d’autres termes, ils n’ont pas remboursé leurs prêts.

L'objectif de l'étude est d'utiliser certaines informations de base sur les clients (telles que le sexe, l'âge et le niveau d'éducation), ainsi que leurs antécédents de remboursement, pour prédire lesquels d'entre eux étaient susceptibles de faire défaut. Le contexte peut être décrit comme suit : en utilisant les 6 mois précédents d'historique de remboursement (avril-septembre 2005), nous essayons de prédire si le client fera défaut en octobre 2005. Naturellement, une telle étude pourrait être généralisée pour prédire si un client sera en défaut de paiement le mois suivant, au cours du trimestre suivant, et ainsi de suite.

Source des données : https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

- **Importation des donnees** :

```python
# Les fonctions DownloadRawData et ReadRawData sont dans le module credit_card_default_utils.py
DownloadRawData()
raw_df = ReadRawData()
```

***Explication des fonctions DownloadRawData() et ReadRawData()***

Ces deux fonctions Python ont pour objectif de télécharger et importer des données brutes à partir d'une source en ligne, puis de les charger dans un DataFrame Pandas pour l'analyse ultérieure. Voici comment elles fonctionnent :

1. DownloadRawData(): Cette fonction effectue les étapes suivantes :

    - Définit l'URL du fichier ZIP que vous souhaitez télécharger.

    - Spécifie le nom du fichier ZIP local dans lequel les données seront stockées.

    - Utilise la bibliothèque requests pour télécharger le fichier ZIP depuis l'URL.

    - Enregistre le contenu téléchargé dans le fichier ZIP local.

    - Ensuite, la fonction extrait les fichiers du ZIP dans le répertoire actuel.

2. ReadRawData(): Cette fonction effectue les étapes suivantes :

    - Spécifie le chemin du fichier de données extrait à partir du ZIP. Dans ce cas, le fichier s'appelle "default of credit card clients.xls".

    - Utilise la bibliothèque Pandas (pd.read_excel()) pour charger le contenu du fichier Excel dans un DataFrame Pandas. Le paramètre header=1 indique que la première ligne du fichier Excel doit être ignorée et que la deuxième ligne (ligne d'en-tête) doit être utilisée comme nom de colonne.
    
    - La fonction retourne le DataFrame contenant les données brutes.

En utilisant ces deux fonctions, vous pouvez automatiser le téléchargement et l'importation de données brutes à partir de l'URL du fichier ZIP. Une fois les données chargées dans le DataFrame Pandas, vous pouvez les manipuler et les analyser selon vos besoins.