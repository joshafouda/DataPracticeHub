# Projet : Nettoyage de Données des Campagnes Marketing pour les Prêts Bancaires Personnels 

## Description

Ce projet de nettoyage de données se concentre sur le traitement et la structuration des informations collectées lors d'une récente campagne marketing d'une banque visant à encourager les clients à souscrire des prêts personnels. Étant donné que les prêts personnels représentent une source de revenus importante pour les banques, il est crucial d'assurer une gestion précise et efficace des données collectées.

## Image
imgs/project5/project5.jpg

## Instructions

**Étape 1 : Lecture et Division des Données**

    1. Charger le Fichier CSV :

        - Importez les [données](https://archive.ics.uci.edu/dataset/222/bank+marketing) dans votre environnement de travail. La [documentation](https://archive.ics.uci.edu/dataset/222/bank+marketing) des données vous indique comment vous pouvez importer ces données directement dans votre environnement de travail en utilisant le package *ucimlrepo*.

        - Utilisez une bibliothèque comme Pandas pour lire le fichier et créer un DataFrame principal.

    2. Créer Trois DataFrames :

        - Séparez les données en trois DataFrames distincts pour "clients.csv" (données démographiques des clients), "campagnes.csv" (données sur la campagne Marketing) et "economics.csv" (données économiques). Vous devez juger vous-mêmes de l'appartenance d'une colonne à telle ou telle autre catégorie de données. Les 3 dataframes doivent avoir la colonne "client_id" en commun.

**Étape 2 : Nettoyage des Données**

    1. Nettoyer les Colonnes et Modifier les Valeurs :

        - Remplacez les caractères spécifiques et gérez les valeurs manquantes selon les exigences de chaque colonne.
    
    2. Créer et Supprimer des Colonnes :

        - Ajoutez toute colonne nécessaire et supprimez celles qui ne sont plus utiles après transformation.

**Étape 3 : Sauvegarde des Données**

    1. Sauvegarder les Trois DataFrames en Fichiers CSV :

        - Enregistrez chaque DataFrame nettoyée dans un fichier CSV avec les noms spécifiés (clients.csv, campagnes.csv, economics.csv)

**Étape 4 : En vous basant sur votre script ou votre notebook de nettoyage des données, développez une application Streamlit où vous pourrez charger le fichier CSV, voir un extrait des données, exécuter le nettoyage et la division des données, et télécharger les fichiers CSV résultants**.

    Une telle application permettra à l'équipe Marketing d'être autonome sur cette phase de nettoyage des données pour les prochaines campagnes Marketing.


## Resources
- [Source des données](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [Apprendre à programmer avec Python](https://youtu.be/LwkWwxg10IU)
- [Playlist sur Streamlit](https://www.youtube.com/playlist?list=PLmJWMf9F8euQKADN-mSCpTlp7uYDyCQNF)
- [Comment déployer une web app Streamlit](https://youtu.be/wjRlWuXmlvw)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)


## Execution du Projet

**Contexte :**

Une banque renommée a récemment mené une campagne marketing ciblant ses clients dans le but de les inciter à contracter des prêts personnels. La campagne a permis de collecter une grande quantité de données, mais celles-ci sont actuellement brutes et non structurées. La banque prévoit de mener plusieurs autres campagnes similaires à l'avenir et souhaite établir un processus rigoureux pour assurer la qualité et la cohérence des données collectées.

**Mission :**

Vous êtes un Data Scientist mandaté par la banque pour nettoyer et structurer les données issues de cette campagne marketing. L'objectif est de garantir que les données sont conformes aux spécifications requises, afin de faciliter dans l'avenir leur importation dans une base de données PostgreSQL dédiée. Cette base de données servira non seulement à stocker les informations de la campagne actuelle, mais aussi à intégrer facilement les données des futures campagnes marketing.

**Impact :**

Grâce à votre travail, la banque pourra non seulement tirer parti des données actuelles pour analyser l'efficacité de leur campagne marketing, mais également disposer d'un cadre robuste pour gérer et intégrer les données des campagnes futures. Cela permettra d'améliorer la prise de décision basée sur les données et d'optimiser les efforts marketing pour atteindre un plus grand nombre de clients potentiels.


```python
# Importation des librairies nécessaires
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import os
```


```python
# Importation des données (Source : https://archive.ics.uci.edu/dataset/222/bank+marketing)
bank_marketing = fetch_ucirepo(id=222) 
type(bank_marketing)
```




    ucimlrepo.dotdict.dotdict




```python
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
print(type(X))
print(type(y))
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>



```python
X.head()
```


```python
X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45211 entries, 0 to 45210
    Data columns (total 16 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   age          45211 non-null  int64 
     1   job          44923 non-null  object
     2   marital      45211 non-null  object
     3   education    43354 non-null  object
     4   default      45211 non-null  object
     5   balance      45211 non-null  int64 
     6   housing      45211 non-null  object
     7   loan         45211 non-null  object
     8   contact      32191 non-null  object
     9   day_of_week  45211 non-null  int64 
     10  month        45211 non-null  object
     11  duration     45211 non-null  int64 
     12  campaign     45211 non-null  int64 
     13  pdays        45211 non-null  int64 
     14  previous     45211 non-null  int64 
     15  poutcome     8252 non-null   object
    dtypes: int64(7), object(9)
    memory usage: 5.5+ MB



```python
y.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45211 entries, 0 to 45210
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   y       45211 non-null  object
    dtypes: object(1)
    memory usage: 353.3+ KB



```python
# metadata 
print(bank_marketing.metadata) 
```

    {'uci_id': 222, 'name': 'Bank Marketing', 'repository_url': 'https://archive.ics.uci.edu/dataset/222/bank+marketing', 'data_url': 'https://archive.ics.uci.edu/static/public/222/data.csv', 'abstract': 'The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).', 'area': 'Business', 'tasks': ['Classification'], 'characteristics': ['Multivariate'], 'num_instances': 45211, 'num_features': 16, 'feature_types': ['Categorical', 'Integer'], 'demographics': ['Age', 'Occupation', 'Marital Status', 'Education Level'], 'target_col': ['y'], 'index_col': None, 'has_missing_values': 'yes', 'missing_values_symbol': 'NaN', 'year_of_dataset_creation': 2014, 'last_updated': 'Fri Aug 18 2023', 'dataset_doi': '10.24432/C5K306', 'creators': ['S. Moro', 'P. Rita', 'P. Cortez'], 'intro_paper': {'title': 'A data-driven approach to predict the success of bank telemarketing', 'authors': 'Sérgio Moro, P. Cortez, P. Rita', 'published_in': 'Decision Support Systems', 'year': 2014, 'url': 'https://www.semanticscholar.org/paper/cab86052882d126d43f72108c6cb41b295cc8a9e', 'doi': '10.1016/j.dss.2014.03.001'}, 'additional_info': {'summary': "The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. \n\nThere are four datasets: \n1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]\n2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.\n3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs). \n4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs). \nThe smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM). \n\nThe classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).", 'purpose': None, 'funded_by': None, 'instances_represent': None, 'recommended_data_splits': None, 'sensitive_data': None, 'preprocessing_description': None, 'variable_info': 'Input variables:\n   # bank client data:\n   1 - age (numeric)\n   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",\n                                       "blue-collar","self-employed","retired","technician","services") \n   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)\n   4 - education (categorical: "unknown","secondary","primary","tertiary")\n   5 - default: has credit in default? (binary: "yes","no")\n   6 - balance: average yearly balance, in euros (numeric) \n   7 - housing: has housing loan? (binary: "yes","no")\n   8 - loan: has personal loan? (binary: "yes","no")\n   # related with the last contact of the current campaign:\n   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") \n  10 - day: last contact day of the month (numeric)\n  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")\n  12 - duration: last contact duration, in seconds (numeric)\n   # other attributes:\n  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)\n  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)\n  15 - previous: number of contacts performed before this campaign and for this client (numeric)\n  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")\n\n  Output variable (desired target):\n  17 - y - has the client subscribed a term deposit? (binary: "yes","no")\n', 'citation': None}}



```python
# variable information 
bank_marketing.variables
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>role</th>
      <th>type</th>
      <th>demographic</th>
      <th>description</th>
      <th>units</th>
      <th>missing_values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>Age</td>
      <td>None</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>job</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>Occupation</td>
      <td>type of job (categorical: 'admin.','blue-colla...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>marital</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>Marital Status</td>
      <td>marital status (categorical: 'divorced','marri...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>education</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>Education Level</td>
      <td>(categorical: 'basic.4y','basic.6y','basic.9y'...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>default</td>
      <td>Feature</td>
      <td>Binary</td>
      <td>None</td>
      <td>has credit in default?</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>balance</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>average yearly balance</td>
      <td>euros</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>housing</td>
      <td>Feature</td>
      <td>Binary</td>
      <td>None</td>
      <td>has housing loan?</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>loan</td>
      <td>Feature</td>
      <td>Binary</td>
      <td>None</td>
      <td>has personal loan?</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>contact</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>contact communication type (categorical: 'cell...</td>
      <td>None</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>day_of_week</td>
      <td>Feature</td>
      <td>Date</td>
      <td>None</td>
      <td>last contact day of the week</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>10</th>
      <td>month</td>
      <td>Feature</td>
      <td>Date</td>
      <td>None</td>
      <td>last contact month of year (categorical: 'jan'...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>11</th>
      <td>duration</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>last contact duration, in seconds (numeric). ...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>12</th>
      <td>campaign</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>number of contacts performed during this campa...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>13</th>
      <td>pdays</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>number of days that passed by after the client...</td>
      <td>None</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>14</th>
      <td>previous</td>
      <td>Feature</td>
      <td>Integer</td>
      <td>None</td>
      <td>number of contacts performed before this campa...</td>
      <td>None</td>
      <td>no</td>
    </tr>
    <tr>
      <th>15</th>
      <td>poutcome</td>
      <td>Feature</td>
      <td>Categorical</td>
      <td>None</td>
      <td>outcome of the previous marketing campaign (ca...</td>
      <td>None</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>16</th>
      <td>y</td>
      <td>Target</td>
      <td>Binary</td>
      <td>None</td>
      <td>has the client subscribed a term deposit?</td>
      <td>None</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>



Voici la description de chaque variable basée sur le résultat du code précédent :

1. **age**
   - **Role**: Feature
   - **Type**: Integer
   - **Demographic**: Age
   - **Description**: Âge du client
   - **Units**: 
   - **Missing Values**: no

2. **job**
   - **Role**: Feature
   - **Type**: Categorical
   - **Demographic**: Occupation
   - **Description**: Type d'emploi (catégories : 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
   - **Units**: 
   - **Missing Values**: yes

3. **marital**
   - **Role**: Feature
   - **Type**: Categorical
   - **Demographic**: Marital Status
   - **Description**: État civil (catégories : 'divorced','married','single','unknown'; note: 'divorced' inclut les divorcés et les veufs)
   - **Units**: 
   - **Missing Values**: no

4. **education**
   - **Role**: Feature
   - **Type**: Categorical
   - **Demographic**: Education Level
   - **Description**: Niveau d'éducation (catégories : 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
   - **Units**: 
   - **Missing Values**: yes

5. **default**
   - **Role**: Feature
   - **Type**: Binary
   - **Demographic**: 
   - **Description**: Le client a-t-il un crédit en défaut ?
   - **Units**: 
   - **Missing Values**: no

6. **balance**
   - **Role**: Feature
   - **Type**: Integer
   - **Demographic**: 
   - **Description**: Solde moyen annuel
   - **Units**: euros
   - **Missing Values**: no

7. **housing**
   - **Role**: Feature
   - **Type**: Binary
   - **Demographic**: 
   - **Description**: Le client a-t-il un prêt immobilier ?
   - **Units**: 
   - **Missing Values**: no

8. **loan**
   - **Role**: Feature
   - **Type**: Binary
   - **Demographic**: 
   - **Description**: Le client a-t-il un prêt personnel ?
   - **Units**: 
   - **Missing Values**: no

9. **contact**
   - **Role**: Feature
   - **Type**: Categorical
   - **Demographic**: 
   - **Description**: Type de communication de contact (catégories : 'cellular','telephone')
   - **Units**: 
   - **Missing Values**: yes

10. **day_of_week**
    - **Role**: Feature
    - **Type**: Date
    - **Demographic**: 
    - **Description**: Jour de la semaine du dernier contact
    - **Units**: 
    - **Missing Values**: no

11. **month**
    - **Role**: Feature
    - **Type**: Date
    - **Demographic**: 
    - **Description**: Mois du dernier contact (catégories : 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    - **Units**: 
    - **Missing Values**: no

12. **duration**
    - **Role**: Feature
    - **Type**: Integer
    - **Demographic**: 
    - **Description**: Durée du dernier contact en secondes. Note importante : cette variable affecte fortement la cible. Par exemple, si duration=0 alors y='no'. Cependant, la durée n'est pas connue avant l'appel. Après l'appel, y est évidemment connu. Donc, cette variable doit être incluse uniquement à des fins de benchmark et doit être exclue pour un modèle prédictif réaliste.
    - **Units**: seconds
    - **Missing Values**: no

13. **campaign**
    - **Role**: Feature
    - **Type**: Integer
    - **Demographic**: 
    - **Description**: Nombre de contacts effectués durant cette campagne pour ce client (inclut le dernier contact)
    - **Units**: 
    - **Missing Values**: no

14. **pdays**
    - **Role**: Feature
    - **Type**: Integer
    - **Demographic**: 
    - **Description**: Nombre de jours écoulés depuis le dernier contact avec le client dans une campagne précédente (-1 signifie que le client n'a pas été contacté auparavant)
    - **Units**: 
    - **Missing Values**: yes

15. **previous**
    - **Role**: Feature
    - **Type**: Integer
    - **Demographic**: 
    - **Description**: Nombre de contacts effectués avant cette campagne pour ce client
    - **Units**: 
    - **Missing Values**: no

16. **poutcome**
    - **Role**: Feature
    - **Type**: Categorical
    - **Demographic**: 
    - **Description**: Résultat de la campagne marketing précédente (catégories : 'failure','nonexistent','success')
    - **Units**: 
    - **Missing Values**: yes

17. **y**
    - **Role**: Target
    - **Type**: Binary
    - **Demographic**: 
    - **Description**: Le client a-t-il souscrit un dépôt à terme ?
    - **Units**: 
    - **Missing Values**: no

Dans le cadre d'un projet de Machine Learning, la variable cible serait la variable y et l'objectif serait de prédire si le client souscrira un dépôt à terme.

Mais notre objectif ici est de procéder au nettoyage de ces données.


```python
# Combiner X et y en une seule dataframe
df = pd.concat([X, y], axis=1)

# Obtenir la longueur de la dataframe
longueur = len(df)

# Diviser la dataframe en deux moitiés égales
df_premiere_moitie = df.iloc[:longueur // 2]
df_deuxieme_moitie = df.iloc[longueur // 2:]

# Affichage des dimensions
print(df.shape)
print(df_premiere_moitie.shape)
print(df_deuxieme_moitie.shape)

# Creation du dossier des données brutes
input_dir = "input_raw_data"
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

# Sauvegarder la première moitié dans un fichier CSV
input_premiere_moitie_path = os.path.join(input_dir, 'data_bank_marketing.csv')
print(input_premiere_moitie_path)
df_premiere_moitie.to_csv(input_premiere_moitie_path, index=False)

# Sauvegarder la deuxième moitié dans un fichier CSV
input_deuxieme_moitie_path = os.path.join(input_dir, 'new_data_bank_marketing.csv')
print(input_deuxieme_moitie_path)
df_deuxieme_moitie.to_csv(input_deuxieme_moitie_path, index=False)

print("Les données ont été sauvegardées avec succès !")
```

    (45211, 17)
    (22605, 17)
    (22606, 17)
    input_raw_data/data_bank_marketing.csv


    input_raw_data/new_data_bank_marketing.csv
    Les données ont été sauvegardées avec succès !


Ce code effectue plusieurs opérations sur des DataFrames pandas et gère des fichiers. Voici une explication détaillée de chaque partie :

1. Combiner X et y en une seule DataFrame
```python
df = pd.concat([X, y], axis=1)
```
- **Description** : Cette ligne combine les DataFrames `X` et `y` en les concaténant horizontalement (c'est-à-dire en ajoutant `y` comme colonnes supplémentaires à `X`).
- **`axis=1`** : Spécifie que la concaténation doit se faire au niveau des colonnes.

2. Obtenir la longueur de la DataFrame
```python
longueur = len(df)
```
- **Description** : Cette ligne obtient le nombre de lignes (la longueur) de la DataFrame `df`.

3. Diviser la DataFrame en deux moitiés égales
```python
df_premiere_moitie = df.iloc[:longueur // 2]
df_deuxieme_moitie = df.iloc[longueur // 2:]
```
- **`df.iloc[:longueur // 2]`** : Sélectionne la première moitié des lignes de `df`.
- **`df.iloc[longueur // 2:]`** : Sélectionne la deuxième moitié des lignes de `df`.

4. Affichage des dimensions
```python
print(df.shape)
print(df_premiere_moitie.shape)
print(df_deuxieme_moitie.shape)
```
- **Description** : Ces lignes impriment les dimensions (nombre de lignes et de colonnes) de `df`, `df_premiere_moitie` et `df_deuxieme_moitie`.

5. Création du dossier des données brutes
```python
input_dir = "input_raw_data"
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
```
- **Description** : Cette partie vérifie si le dossier `input_raw_data` existe. Si ce n'est pas le cas, il le crée.
- **`os.path.exists(input_dir)`** : Vérifie l'existence du dossier.
- **`os.makedirs(input_dir)`** : Crée le dossier s'il n'existe pas.

6. Sauvegarder la première moitié dans un fichier CSV
```python
input_premiere_moitie_path = os.path.join(input_dir, 'data_bank_marketing.csv')
print(input_premiere_moitie_path)
df_premiere_moitie.to_csv(input_premiere_moitie_path, index=False)
```
- **`input_premiere_moitie_path`** : Définit le chemin du fichier CSV pour la première moitié des données.
- **`print(input_premiere_moitie_path)`** : Imprime le chemin du fichier.
- **`df_premiere_moitie.to_csv(input_premiere_moitie_path, index=False)`** : Sauvegarde `df_premiere_moitie` dans un fichier CSV sans inclure l'index des lignes.

7. Sauvegarder la deuxième moitié dans un fichier CSV
```python
input_deuxieme_moitie_path = os.path.join(input_dir, 'new_data_bank_marketing.csv')
print(input_deuxieme_moitie_path)
df_deuxieme_moitie.to_csv(input_deuxieme_moitie_path, index=False)
```
- **`input_deuxieme_moitie_path`** : Définit le chemin du fichier CSV pour la deuxième moitié des données.
- **`print(input_deuxieme_moitie_path)`** : Imprime le chemin du fichier.
- **`df_deuxieme_moitie.to_csv(input_deuxieme_moitie_path, index=False)`** : Sauvegarde `df_deuxieme_moitie` dans un fichier CSV sans inclure l'index des lignes.

8. Confirmation de la sauvegarde
```python
print("Les données ont été sauvegardées avec succès !")
```
- **Description** : Imprime un message confirmant que les données ont été sauvegardées avec succès.

La dataframe df_deuxieme_moitie sera utilisée pour simuler une nouvelle campagne Marketing afin de tester l'application Streamlit qui sera construite pour nettoyer les données des futures camapagnes Marketing.


```python
df_premiere_moitie.head()
```

Paasons maintenant à la phase de Nettoyage.


```python
# On suppose que nous venons de recevoir le fichier CSV des données de la campagne actuelle

marketing = pd.read_csv(input_premiere_moitie_path)
marketing.head()
```


```python
marketing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22605 entries, 0 to 22604
    Data columns (total 17 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   age          22605 non-null  int64  
     1   job          22430 non-null  object 
     2   marital      22605 non-null  object 
     3   education    21629 non-null  object 
     4   default      22605 non-null  object 
     5   balance      22605 non-null  int64  
     6   housing      22605 non-null  object 
     7   loan         22605 non-null  object 
     8   contact      9918 non-null   object 
     9   day_of_week  22605 non-null  int64  
     10  month        22605 non-null  object 
     11  duration     22605 non-null  int64  
     12  campaign     22605 non-null  int64  
     13  pdays        22605 non-null  int64  
     14  previous     22605 non-null  int64  
     15  poutcome     0 non-null      float64
     16  y            22605 non-null  object 
    dtypes: float64(1), int64(7), object(9)
    memory usage: 2.9+ MB



```python
marketing.isna().sum()
```




    age                0
    job              175
    marital            0
    education        976
    default            0
    balance            0
    housing            0
    loan               0
    contact        12687
    day_of_week        0
    month              0
    duration           0
    campaign           0
    pdays              0
    previous           0
    poutcome       22605
    y                  0
    dtype: int64



Toutes les valeurs de la colonne 'poutcome' sont manquantes donc nous allons la supprimer car elle ne nous apporte aucune information.


```python
def drop_poutcome_column(df):
    """
    Supprime la colonne 'poutcome' de la DataFrame.

    Args:
        df (pandas.DataFrame): La DataFrame initiale.

    Returns:
        pandas.DataFrame: Une copie de la DataFrame sans la colonne 'poutcome'.

    Raises:
        KeyError: Si la colonne 'poutcome' n'existe pas dans la DataFrame.
        Exception: Si une autre erreur se produit lors de la suppression de la colonne.
    
    Example:
        >>> new_df = drop_poutcome_column(clients)
    """
    try:
        # Création d'une copie de la DataFrame
        df_copy = df.copy()
        
        # Suppression de la colonne 'poutcome'
        df_copy = df_copy.drop('poutcome', axis=1)
        
        return df_copy
    except KeyError as e:
        raise KeyError("La colonne 'poutcome' est absente de la DataFrame :", e)
    except Exception as e:
        raise Exception("Une erreur s'est produite lors de la suppression de la colonne 'poutcome' :", e)
```


```python
marketing_clean1 = drop_poutcome_column(marketing)
marketing_clean1.columns
```




    Index(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
           'loan', 'contact', 'day_of_week', 'month', 'duration', 'campaign',
           'pdays', 'previous', 'y'],
          dtype='object')



Investigation sur les colonnes 'job', 'education' et 'contact' pour connaitre leurs modalités :


```python
def afficher_modalites_variables_categorielles(dataframe, colonnes_categorielles):

    # Parcourez chaque colonne catégorielle et affichez ses modalités
    for colonne in colonnes_categorielles:
        print("Modalités de la colonne '{}' :".format(colonne))
        print(dataframe[colonne].unique())
        print()

# Exemple d'utilisation de la fonction avec une DataFrame nommée 'clients'
afficher_modalites_variables_categorielles(marketing, ['job', 'education', 'contact'])
```

    Modalités de la colonne 'job' :
    ['management' 'technician' 'entrepreneur' 'blue-collar' nan 'retired'
     'admin.' 'services' 'self-employed' 'unemployed' 'housemaid' 'student']
    
    Modalités de la colonne 'education' :
    ['tertiary' 'secondary' nan 'primary']
    
    Modalités de la colonne 'contact' :
    [nan 'cellular' 'telephone']
    


Nous allons remplacer les valeurs manquantes dans ces colonnes 'job', 'education' et 'contact' par "unknown".


```python
def imputer_valeurs_manquantes(dataframe, variables_categorielles):
    """
    Impute les valeurs manquantes dans les variables catégorielles d'une DataFrame.

    Args:
        dataframe (pandas.DataFrame): La DataFrame contenant les données.
        variables_categorielles (list): Liste des noms des variables catégorielles où faire l'imputation.

    Returns:
        pandas.DataFrame: La DataFrame avec les valeurs manquantes imputées.

    Raises:
        Exception: Si une erreur se produit lors de l'imputation.

    Example:
        >>> variables_categorielles = ['job', 'education']
        >>> clean_df = imputer_valeurs_manquantes(clients, variables_catégorielles)
    """
    try:
        # Creer une copie de dataframe
        dataframe_copy = dataframe.copy()
        
        # Remplacer les valeurs manquantes dans les variables catégorielles par "unknown"
        for variable in variables_categorielles:
            dataframe_copy[variable] = dataframe_copy[variable].fillna('unknown')
        return dataframe_copy
    except Exception as e:
        raise Exception("Une erreur s'est produite lors de l'imputation des valeurs manquantes :", e)
```


```python
marketing_clean2 = imputer_valeurs_manquantes(marketing_clean1, ['job', 'education', 'contact'])
marketing_clean2.isna().sum()
```




    age            0
    job            0
    marital        0
    education      0
    default        0
    balance        0
    housing        0
    loan           0
    contact        0
    day_of_week    0
    month          0
    duration       0
    campaign       0
    pdays          0
    previous       0
    y              0
    dtype: int64




```python
marketing_clean2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22605 entries, 0 to 22604
    Data columns (total 16 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   age          22605 non-null  int64 
     1   job          22605 non-null  object
     2   marital      22605 non-null  object
     3   education    22605 non-null  object
     4   default      22605 non-null  object
     5   balance      22605 non-null  int64 
     6   housing      22605 non-null  object
     7   loan         22605 non-null  object
     8   contact      22605 non-null  object
     9   day_of_week  22605 non-null  int64 
     10  month        22605 non-null  object
     11  duration     22605 non-null  int64 
     12  campaign     22605 non-null  int64 
     13  pdays        22605 non-null  int64 
     14  previous     22605 non-null  int64 
     15  y            22605 non-null  object
    dtypes: int64(7), object(9)
    memory usage: 2.8+ MB



```python
marketing_clean2.head(10)
```


Nous avons maintenant une dataframe nettoyée. Passons à la répartition des données.

Pour identifier les colonnes qui représentent uniquement les données démographiques des clients, nous devons nous référer à la description des variables que nous avons. Les colonnes démographiques typiques dans ce contexte incluent les informations personnelles et statiques des clients telles que l'âge, l'emploi, l'état civil, et le niveau d'éducation. 

Voici les colonnes identifiées comme étant les données démographiques des clients :
1. `age`
2. `job`
3. `marital`
4. `education`

Nous allons créer une nouvelle dataframe `clients` en sélectionnant uniquement ces colonnes de la dataframe `marketing`.


```python
# Sélection des colonnes démographiques
demographic_columns = ['age', 'job', 'marital', 'education']

# Création de la dataframe clients
clients = marketing_clean2[demographic_columns]

# Affichage de la dataframe clients pour vérification
clients.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
    </tr>
  </tbody>
</table>
</div>




```python
clients.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22605 entries, 0 to 22604
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   age        22605 non-null  int64 
     1   job        22605 non-null  object
     2   marital    22605 non-null  object
     3   education  22605 non-null  object
    dtypes: int64(1), object(3)
    memory usage: 706.5+ KB


Pour identifier les colonnes qui représentent uniquement les données de la campagne marketing, nous devons rechercher celles qui ont des informations spécifiques à la campagne, telles que le nombre de contacts effectués, le résultat de la campagne précédente, etc.

En se basant sur les informations fournies, voici les colonnes qui semblent représenter les données de la campagne marketing :

- **contact**
- **day_of_week**
- **month**
- **duration**
- **campaign**
- **pdays**
- **previous**
- **y** (c'est finalement le résultat ultime de la campagne Marketing)

Créons maintenant la DataFrame "campagnes" à partir de ces colonnes :


```python
# Création de la DataFrame campagnes
colonnes_campagnes = ['contact', 'day_of_week', 'month', 
                      'duration', 'campaign', 'pdays', 
                      'previous', 'y']
campagnes = marketing_clean2[colonnes_campagnes]

# Affichage des informations sur la DataFrame campagnes
print("DataFrame campagnes :")
campagnes.head()
```

    DataFrame campagnes :





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>contact</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>



Maintenant, la DataFrame "campagnes" contient uniquement les colonnes liées aux données de la campagne marketing.


```python
campagnes.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22605 entries, 0 to 22604
    Data columns (total 8 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   contact      22605 non-null  object
     1   day_of_week  22605 non-null  int64 
     2   month        22605 non-null  object
     3   duration     22605 non-null  int64 
     4   campaign     22605 non-null  int64 
     5   pdays        22605 non-null  int64 
     6   previous     22605 non-null  int64 
     7   y            22605 non-null  object
    dtypes: int64(5), object(3)
    memory usage: 1.4+ MB


Les données économiques concernent le reste des variables de marketing_clean2 :


```python
colonnes_economics = [
    c for c in marketing_clean2.columns
    if c not in clients.columns
    and c not in campagnes.columns
]
colonnes_economics
```




    ['default', 'balance', 'housing', 'loan']



Ce code crée une liste de colonnes nommée `colonnes_economics` en sélectionnant les colonnes qui sont présentes dans la DataFrame `marketing_clean2` mais qui ne sont pas présentes dans les DataFrames `clients` et `campagnes`. Voici une explication détaillée de chaque partie du code :

1. **Liste en compréhension** :
    ```python
    colonnes_economics = [
    ```
    Cette syntaxe est utilisée pour créer une nouvelle liste basée sur une autre séquence ou une autre condition.

2. **Boucle sur les colonnes de `marketing_clean2`** :
    ```python
    c for c in marketing_clean2.columns
    ```
    Cela signifie que pour chaque colonne `c` dans la liste des colonnes de la DataFrame `marketing_clean2`, l'élément `c` sera inclus dans la nouvelle liste si certaines conditions sont remplies.

3. **Conditions pour inclure une colonne** :
    ```python
    if c not in clients.columns and c not in campagnes.columns
    ```
    La colonne `c` est incluse dans la liste `colonnes_economics` si elle n'est pas présente dans les colonnes de la DataFrame `clients` **et** qu'elle n'est pas présente dans les colonnes de la DataFrame `campagnes`.


```python
economics = marketing_clean2[colonnes_economics]
economics.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
economics.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22605 entries, 0 to 22604
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   default  22605 non-null  object
     1   balance  22605 non-null  int64 
     2   housing  22605 non-null  object
     3   loan     22605 non-null  object
    dtypes: int64(1), object(3)
    memory usage: 706.5+ KB



```python
# Sauvegarde des 3 dataframes au format CSV
import os

output_dir = "output_clean_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_clients_path = os.path.join(output_dir, "clients.csv")
output_campagnes_path = os.path.join(output_dir, "campagnes.csv")
output_economics_path = os.path.join(output_dir, "economics.csv")

clients.to_csv(output_clients_path, index=False)
campagnes.to_csv(output_campagnes_path, index=False)
economics.to_csv(output_economics_path, index=False)
```

Passons maintenant à la constrution de l'application web pour nettoyer automatiquement les données des futures camapagnes Marketing.

Ci-dessous, le code de l'application Streamlit :

```python
import streamlit as st
import pandas as pd
import os
import io
from utils import load_to_csv

# Titre de l'application
st.title("Application de Nettoyage et de Division des Données Marketing")

# Téléchargement du fichier des données marketing
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Lecture du fichier en DataFrame
    marketing_data = pd.read_csv(uploaded_file)

    # Affichage d'un extrait de la DataFrame et de sa structure côte à côte
    st.write("Aperçu des données et de la structure de marketing_data :")
    col1, col2 = st.columns(2)
    with col1:
        st.write(marketing_data.head())
        st.write(f"Dimensions des données : {marketing_data.shape}")
    with col2:
        buffer = io.StringIO()
        marketing_data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    

    # Affichage des dimensions de la DataFrame
    st.write(f"Dimensions des données : {marketing_data.shape}")
    
    # Bouton pour exécuter le processus de nettoyage et de division
    run_button_clicked = st.button("Run")
    
    if run_button_clicked:
        try:
            load_to_csv(marketing_data)
            
            output_dir = "output_clean_data"
            output_clients_path = os.path.join(output_dir, "clients.csv")
            output_campagnes_path = os.path.join(output_dir, "campagnes.csv")
            output_economics_path = os.path.join(output_dir, "economics.csv")
            
            # Fonction pour afficher DataFrame et info côte à côte
            def display_dataframe_info(df, title):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(title)
                    st.write(df.head())
                    st.write(f"Dimensions : {df.shape}")
                with col2:
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text(s)
            
            # Affichage des trois DataFrames résultantes
            st.write("DataFrame Clients et sa structure :")
            display_dataframe_info(pd.read_csv(output_clients_path), "DataFrame Clients :")
            
            st.write("DataFrame Campagnes et sa structure :")
            display_dataframe_info(pd.read_csv(output_campagnes_path), "DataFrame Campagnes :")
            
            st.write("DataFrame Economics et sa structure :")
            display_dataframe_info(pd.read_csv(output_economics_path), "DataFrame Economics :")
            
            st.success("Les fichiers CSV ont été sauvegardés avec succès!")
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")

```

Voici une explication détaillée du code de l'application :

```python
import streamlit as st
import pandas as pd
import os
import io
from utils import load_to_csv
```
Cette section importe les bibliothèques nécessaires. `streamlit` est utilisé pour créer l'interface utilisateur, `pandas` pour manipuler les données, `os` pour interagir avec le système de fichiers, et `io` pour gérer les flux d'entrée/sortie. `load_to_csv` est une fonction que nous avons définie dans le module `utils` pour nettoyer et diviser les données, puis les sauvegarder en fichiers CSV.

```python
# Titre de l'application
st.title("Application de Nettoyage et de Division des Données Marketing")
```
Cette ligne définit le titre de l'application Streamlit.

```python
# Téléchargement du fichier des données marketing
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
```
Cette ligne crée un composant de téléchargement de fichier. Les utilisateurs peuvent utiliser ce composant pour télécharger un fichier CSV.

```python
if uploaded_file is not None:
    # Lecture du fichier en DataFrame
    marketing_data = pd.read_csv(uploaded_file)
```
Si un fichier est téléchargé, il est lu en tant que DataFrame `pandas` et stocké dans la variable `marketing_data`.

```python
    # Affichage d'un extrait de la DataFrame et de sa structure côte à côte
    st.write("Aperçu des données et de la structure de marketing_data :")
    col1, col2 = st.columns(2)
    with col1:
        st.write(marketing_data.head())
        st.write(f"Dimensions des données : {marketing_data.shape}")
    with col2:
        buffer = io.StringIO()
        marketing_data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
```
Cette section affiche un aperçu des données et leur structure côte à côte. `st.columns(2)` crée deux colonnes. Dans la première colonne, nous affichons les premières lignes de la DataFrame et ses dimensions. Dans la deuxième colonne, nous affichons la structure des données (`info()`), y compris les types de données et les valeurs manquantes. Nous utilisons un buffer `io.StringIO()` pour capturer la sortie de `marketing_data.info()`.

```python
    # Bouton pour exécuter le processus de nettoyage et de division
    run_button_clicked = st.button("Run")
    
    if run_button_clicked:
        try:
            load_to_csv(marketing_data)
            
            output_dir = "output_clean_data"
            output_clients_path = os.path.join(output_dir, "clients.csv")
            output_campagnes_path = os.path.join(output_dir, "campagnes.csv")
            output_economics_path = os.path.join(output_dir, "economics.csv")
```
Cette section crée un bouton "Run". Lorsque le bouton est cliqué, il appelle la fonction `load_to_csv` pour nettoyer et diviser les données, puis les sauvegarde en trois fichiers CSV dans le dossier `output_clean_data`. Les chemins vers ces fichiers sont stockés dans des variables.

```python
            # Fonction pour afficher DataFrame et info côte à côte
            def display_dataframe_info(df, title):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(title)
                    st.write(df.head())
                    st.write(f"Dimensions : {df.shape}")
                with col2:
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text(s)
            
            # Affichage des trois DataFrames résultantes
            st.write("DataFrame Clients et sa structure :")
            display_dataframe_info(pd.read_csv(output_clients_path), "DataFrame Clients :")
            
            st.write("DataFrame Campagnes et sa structure :")
            display_dataframe_info(pd.read_csv(output_campagnes_path), "DataFrame Campagnes :")
            
            st.write("DataFrame Economics et sa structure :")
            display_dataframe_info(pd.read_csv(output_economics_path), "DataFrame Economics :")
            
            st.success("Les fichiers CSV ont été sauvegardés avec succès!")
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
```
Nous définissons une fonction `display_dataframe_info` pour afficher une DataFrame et sa structure côte à côte. Cette fonction est ensuite utilisée pour afficher les trois DataFrames résultantes (`clients`, `campagnes`, `economics`) et leur structure.

- `display_dataframe_info` utilise `st.columns(2)` pour créer deux colonnes : l'une affiche les premières lignes de la DataFrame et ses dimensions, et l'autre affiche la structure de la DataFrame.
- Les trois DataFrames sont chargées à partir des fichiers CSV et affichées à l'aide de `display_dataframe_info`.
- Si tout se passe bien, un message de succès est affiché. Si une erreur se produit, un message d'erreur est affiché.

Ce code permet aux utilisateurs de télécharger un fichier CSV, de voir un aperçu des données, de nettoyer et de diviser les données en trois DataFrames, puis de les sauvegarder tout en évitant les doublons.

Voici une explication concise des fonctions du script `utils.py` :

- Importations et préparation
```python
import pandas as pd
import numpy as np
import os
```
- **`pandas`** : utilisé pour manipuler les données sous forme de DataFrame.
- **`numpy`** : utilisé pour des opérations mathématiques et manipulation des tableaux.
- **`os`** : utilisé pour interagir avec le système de fichiers, comme créer des dossiers.

- Fonction `drop_poutcome_column`
```python
def drop_poutcome_column(df):
    """
    Supprime la colonne 'poutcome' de la DataFrame.
    """
    try:
        df_copy = df.copy()
        df_copy = df_copy.drop('poutcome', axis=1)
        return df_copy
    except KeyError as e:
        raise KeyError("La colonne 'poutcome' est absente de la DataFrame :", e)
    except Exception as e:
        raise Exception("Une erreur s'est produite lors de la suppression de la colonne 'poutcome' :", e)
```
- **Objectif** : Supprimer la colonne `'poutcome'` de la DataFrame.
- **Retourne** : Une copie de la DataFrame sans la colonne `'poutcome'`.
- **Gestion des erreurs** : 
  - **`KeyError`** : si la colonne `'poutcome'` n'existe pas.
  - **`Exception`** : pour toute autre erreur lors de la suppression.

- Fonction `imputer_valeurs_manquantes`
```python
def imputer_valeurs_manquantes(dataframe, variables_categorielles):
    """
    Impute les valeurs manquantes dans les variables catégorielles d'une DataFrame.
    """
    try:
        dataframe_copy = dataframe.copy()
        for variable in variables_categorielles:
            dataframe_copy[variable] = dataframe_copy[variable].fillna('unknown')
        return dataframe_copy
    except Exception as e:
        raise Exception("Une erreur s'est produite lors de l'imputation des valeurs manquantes :", e)
```
- **Objectif** : Remplacer les valeurs manquantes dans les colonnes catégorielles par `'unknown'`.
- **Retourne** : Une copie de la DataFrame avec les valeurs manquantes imputées.
- **Gestion des erreurs** : 
  - **`Exception`** : pour toute erreur lors de l'imputation.

- Fonction `clean_and_split_data`
```python
def clean_and_split_data(marketing_data):
    """
    Nettoie et divise les données marketing en trois DataFrames distinctes : clients, campagnes et economics.
    """
    try:
        marketing_data_clean1 = drop_poutcome_column(marketing_data)
        marketing_data_clean2 = imputer_valeurs_manquantes(
            marketing_data_clean1, 
            ['job', 'education', 'contact']
        )

        demographic_columns = ['age', 'job', 'marital', 'education']
        clients = marketing_data_clean2[demographic_columns]

        colonnes_campagnes = ['contact', 'day_of_week', 'month', 
                              'duration', 'campaign', 'pdays', 
                              'previous', 'y']
        campagnes = marketing_data_clean2[colonnes_campagnes]

        colonnes_economics = [
            c for c in marketing_data_clean2.columns
            if c not in clients.columns
            and c not in campagnes.columns
        ]
        economics = marketing_data_clean2[colonnes_economics]

        return clients, campagnes, economics

    except KeyError as e:
        raise KeyError("Certaines colonnes nécessaires sont absentes de la DataFrame :", e)
    except Exception as e:
        raise Exception("Une erreur s'est produite lors du nettoyage et de la division des données :", e)
```
- **Objectif** : Nettoyer les données et les diviser en trois DataFrames : `clients`, `campagnes`, et `economics`.
- **Étapes** :
  - Supprimer la colonne `'poutcome'`.
  - Imputer les valeurs manquantes pour les colonnes catégorielles spécifiées.
  - Créer trois DataFrames distinctes pour les données démographiques, les campagnes, et les données économiques.
- **Retourne** : Un tuple contenant les trois DataFrames (`clients`, `campagnes`, `economics`).
- **Gestion des erreurs** : 
  - **`KeyError`** : si des colonnes nécessaires sont absentes.
  - **`Exception`** : pour toute autre erreur lors du nettoyage et de la division.

- Fonction `load_to_csv`
```python
def load_to_csv(marketing_data):
    clients_df, campagnes_df, economics_df = clean_and_split_data(marketing_data)
    output_dir = "output_clean_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_clients_path = os.path.join(output_dir, "clients.csv")
    output_campagnes_path = os.path.join(output_dir, "campagnes.csv")
    output_economics_path = os.path.join(output_dir, "economics.csv")

    # On écrase les anciens fichiers et on les remplace par les nouvelles données
    clients_df.to_csv(output_clients_path, index=False)
    campagnes_df.to_csv(output_campagnes_path, index=False)
    economics_df.to_csv(output_economics_path, index=False)
```
- **Objectif** : Nettoyer et diviser les données, puis sauvegarder les DataFrames résultantes en fichiers CSV.
- **Étapes** :
  - Nettoyer et diviser les données en appelant `clean_and_split_data`.
  - Créer un dossier `output_clean_data` si celui-ci n'existe pas.
  - Définir les chemins pour les fichiers CSV.
  - Sauvegarder les DataFrames `clients`, `campagnes`, et `economics` en écrasant les anciens fichiers CSV.

