# Projet : Prédire le prix d'un bien immobilier grâce au Machine Learning

## Description

Dans ce projet, nous cherchons à créer un modèle de machine learning capable de prédire le prix des biens immobiliers en France. Nous disposons d'un ensemble de données réelles comprenant diverses caractéristiques des propriétés telles que la taille, le nombre de chambres, l'emplacement géographique, etc. L'objectif est d'exploiter ces informations pour entraîner un modèle précis et fiable qui peut estimer le prix d'une propriété sur la base de ses caractéristiques.


## Image
imgs/project13/project13.png

## Instructions

1. **Exploration des Données (EDA):**
   - Charger les données dans un notebook (p. ex. Jupyter).
   - Effectuer une exploration des données pour comprendre la structure, les types de données, les valeurs manquantes, etc.
   - Visualiser les distributions, les corrélations entre les variables et explorer les relations potentielles avec le prix des biens immobiliers.

2. **Feature Engineering:**
   - Identifier les variables pertinentes pour la prédiction du prix immobilier.
   - Créer de nouvelles caractéristiques potentielles à partir des données existantes, telles que la surface habitable par chambre, la distance par rapport aux points d'intérêt, etc.
   - Traiter les valeurs manquantes et les valeurs aberrantes.

3. **Prétraitement des Données:**
   - Effectuer une normalisation ou une standardisation des données si nécessaire.
   - Encoder les variables catégorielles.
   - Diviser les données en ensembles d'entraînement et de test.

4. **Modélisation:**
   - Choisir un ou plusieurs algorithmes de machine learning appropriés (par exemple, régression linéaire, random forest, XGBoost, etc.).
   - Entraîner les modèles sur les données d'entraînement.
   - Évaluer les performances des modèles à l'aide de métriques telles que l'erreur quadratique moyenne (RMSE), le coefficient de détermination (R²), etc.

5. **Optimisation et Validation:**
   - Optimiser les hyperparamètres des modèles pour améliorer les performances.
   - Valider les modèles sur l'ensemble de test pour s'assurer de leur généralisation.

6. **Sauvegarde du Modèle:**
   - Sauvegarder le modèle entraîné pour une utilisation future.
   - Documenter les étapes et les décisions prises tout au long du processus.

7. **Rapport Final:**
   - Présenter les résultats obtenus, y compris les performances du modèle, les caractéristiques les plus importantes, les défis rencontrés, etc.
   - Proposer des recommandations pour l'amélioration future du modèle ou de la collecte de données.

En suivant ces étapes, l'utilisateur pourra créer un modèle de prédiction des prix immobiliers en France utilisant des données réelles et des techniques avancées de machine learning.


## Resources

- [Ensemble de données des Caractéristiques](https://drive.google.com/file/d/1aE7TPBJqRgLALYjYoqJaVQfifYF_9GY4/view?usp=sharing)
- [Ensemble de données des Étiquettes](https://drive.google.com/file/d/1dkjerJ7u5Tq8XCyTH_6glAxSYVdHpu1w/view?usp=sharing)
- [Regardez le projet en vidéo](https://youtu.be/3hyxEtBdC_w)
- [Machine Learning par la pratique avec Python](https://www.amazon.fr/dp/B08DV8X9D2?ref_=ast_author_ofdp)
- [Playlist de Vidéos sur le Machine Learning avec Python](https://www.youtube.com/playlist?list=PLmJWMf9F8euTuNEnfnV-qdaVOOL8cIY9Q)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)


## Execution du Projet

Pour ce projet, vous pouvez utiliser les packages suivants :

joblib==1.4.2

numpy==1.26.4

pandas==2.2.2

scikit-learn==1.5.0

seaborn==0.13.2

xgboost==2.0.3


🖥 **Notebook pour la réalisation du projet**

Créez un notebook dans votre environnement python. Exemple : *french_real_estate_prediction.ipynb*




```python
import pandas as pd
features = pd.read_csv('X_train_J01Z4CN.csv')
features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37368 entries, 0 to 37367
    Data columns (total 27 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   id_annonce                   37368 non-null  int64  
     1   property_type                37368 non-null  object 
     2   approximate_latitude         37368 non-null  float64
     3   approximate_longitude        37368 non-null  float64
     4   city                         37368 non-null  object 
     5   postal_code                  37368 non-null  int64  
     6   size                         36856 non-null  float64
     7   floor                        9743 non-null   float64
     8   land_size                    15581 non-null  float64
     9   energy_performance_value     19068 non-null  float64
     10  energy_performance_category  19068 non-null  object 
     11  ghg_value                    18530 non-null  float64
     12  ghg_category                 18530 non-null  object 
     13  exposition                   9094 non-null   object 
     14  nb_rooms                     35802 non-null  float64
     15  nb_bedrooms                  34635 non-null  float64
     16  nb_bathrooms                 24095 non-null  float64
     17  nb_parking_places            37368 non-null  float64
     18  nb_boxes                     37368 non-null  float64
     19  nb_photos                    37368 non-null  float64
     20  has_a_balcony                37368 non-null  float64
     21  nb_terraces                  37368 non-null  float64
     22  has_a_cellar                 37368 non-null  float64
     23  has_a_garage                 37368 non-null  float64
     24  has_air_conditioning         37368 non-null  float64
     25  last_floor                   37368 non-null  float64
     26  upper_floors                 37368 non-null  float64
    dtypes: float64(20), int64(2), object(5)
    memory usage: 7.7+ MB



```python
features.head()
```


```python
# Target
labels = pd.read_csv('y_train_OXxrJt1.csv')
labels.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37368 entries, 0 to 37367
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   id_annonce  37368 non-null  int64  
     1   price       37368 non-null  float64
    dtypes: float64(1), int64(1)
    memory usage: 584.0 KB



```python
labels.head()
```


```python
df = features.merge(labels, on='id_annonce')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37368 entries, 0 to 37367
    Data columns (total 28 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   id_annonce                   37368 non-null  int64  
     1   property_type                37368 non-null  object 
     2   approximate_latitude         37368 non-null  float64
     3   approximate_longitude        37368 non-null  float64
     4   city                         37368 non-null  object 
     5   postal_code                  37368 non-null  int64  
     6   size                         36856 non-null  float64
     7   floor                        9743 non-null   float64
     8   land_size                    15581 non-null  float64
     9   energy_performance_value     19068 non-null  float64
     10  energy_performance_category  19068 non-null  object 
     11  ghg_value                    18530 non-null  float64
     12  ghg_category                 18530 non-null  object 
     13  exposition                   9094 non-null   object 
     14  nb_rooms                     35802 non-null  float64
     15  nb_bedrooms                  34635 non-null  float64
     16  nb_bathrooms                 24095 non-null  float64
     17  nb_parking_places            37368 non-null  float64
     18  nb_boxes                     37368 non-null  float64
     19  nb_photos                    37368 non-null  float64
     20  has_a_balcony                37368 non-null  float64
     21  nb_terraces                  37368 non-null  float64
     22  has_a_cellar                 37368 non-null  float64
     23  has_a_garage                 37368 non-null  float64
     24  has_air_conditioning         37368 non-null  float64
     25  last_floor                   37368 non-null  float64
     26  upper_floors                 37368 non-null  float64
     27  price                        37368 non-null  float64
    dtypes: float64(21), int64(2), object(5)
    memory usage: 8.0+ MB


Ce code réalise une opération de fusion (merge) entre deux DataFrames nommés `features` et `labels` en utilisant une colonne commune appelée `id_annonce`. Voici une explication détaillée de chaque élément du code :

1. **`df =`** :
   - Ceci indique que le résultat de l'opération de fusion sera assigné à une nouvelle variable nommée `df`.

2. **`features`** :
   - Il s'agit d'un DataFrame, qui est une structure de données en pandas utilisée pour stocker des données tabulaires (similaire à une table dans une base de données ou une feuille de calcul Excel). Ce DataFrame contient les caractéristiques (features) des annonces immobilières.

3. **`merge`** :
   - Il s'agit d'une méthode pandas utilisée pour combiner deux DataFrames en utilisant une clé commune. C'est similaire à une jointure (JOIN) en SQL.

4. **`labels`** :
   - Il s'agit d'un autre DataFrame contenant probablement les étiquettes (labels) ou les valeurs cibles associées aux annonces immobilières, comme les prix des propriétés.

5. **`on='id_annonce'`** :
   - Cela spécifie la colonne commune sur laquelle la fusion doit être effectuée. Dans ce cas, la colonne `id_annonce` est présente dans les deux DataFrames (`features` et `labels`), et elle est utilisée comme clé pour aligner les lignes correspondantes des deux DataFrames.

**Contexte et Fonctionnement :**

- **But de la Fusion :** La fusion est utilisée pour combiner les données des deux DataFrames de manière à ce que chaque ligne de `features` soit alignée avec la ligne correspondante de `labels` basée sur la colonne `id_annonce`.
- **Résultat de la Fusion :** Le DataFrame résultant `df` contiendra toutes les colonnes de `features` et de `labels`, avec les lignes combinées là où les valeurs de `id_annonce` correspondent. Si une annonce (une valeur de `id_annonce`) est présente dans les deux DataFrames, ses caractéristiques et son étiquette seront combinées dans une seule ligne de `df`.

**Exemple Visuel :**

Supposons que nous ayons les deux DataFrames suivants :

`features` :
| id_annonce | size | nb_rooms |
|------------|------|----------|
| 1          | 50   | 3        |
| 2          | 75   | 4        |
| 3          | 60   | 2        |

`labels` :
| id_annonce | price |
|------------|-------|
| 1          | 250000|
| 2          | 320000|
| 3          | 210000|

Après la fusion (`merge`), nous obtenons :

`df` :
| id_annonce | size | nb_rooms | price  |
|------------|------|----------|--------|
| 1          | 50   | 3        | 250000 |
| 2          | 75   | 4        | 320000 |
| 3          | 60   | 2        | 210000 |

Ainsi, `df` contient à la fois les caractéristiques et les étiquettes pour chaque annonce immobilière.

En résumé, ce code permet de combiner les données de caractéristiques et d'étiquettes en une seule table cohérente, facilitant ainsi les étapes ultérieures d'analyse et de modélisation des données.


```python
df.head()
```


```python
df.isna().sum()
```




    id_annonce                         0
    property_type                      0
    approximate_latitude               0
    approximate_longitude              0
    city                               0
    postal_code                        0
    size                             512
    floor                          27625
    land_size                      21787
    energy_performance_value       18300
    energy_performance_category    18300
    ghg_value                      18838
    ghg_category                   18838
    exposition                     28274
    nb_rooms                        1566
    nb_bedrooms                     2733
    nb_bathrooms                   13273
    nb_parking_places                  0
    nb_boxes                           0
    nb_photos                          0
    has_a_balcony                      0
    nb_terraces                        0
    has_a_cellar                       0
    has_a_garage                       0
    has_air_conditioning               0
    last_floor                         0
    upper_floors                       0
    price                              0
    dtype: int64




```python
df.duplicated().sum()
```




    0



Le code `df.duplicated().sum()` est utilisé pour identifier et compter le nombre de lignes dupliquées dans le DataFrame `df`.

Voici une explication détaillée du fonctionnement de ce code :

1. **`df`** :
   - C'est le DataFrame sur lequel l'opération est effectuée. Il contient les données combinées de caractéristiques et d'étiquettes suite à la fusion des DataFrames `features` et `labels`.

2. **`duplicated()`** :
   - C'est une méthode pandas qui renvoie une série booléenne de la même longueur que le DataFrame, où chaque valeur indique si la ligne correspondante est une duplication d'une ligne précédente. Une ligne est considérée comme dupliquée si toutes ses valeurs sont identiques à une autre ligne précédemment rencontrée dans le DataFrame.
   - Par défaut, `duplicated()` marque les duplications après la première occurrence (c'est-à-dire que la première occurrence de la duplication n'est pas marquée comme dupliquée).

3. **`sum()`** :
   - Appliqué à une série booléenne, `sum()` traite chaque valeur `True` comme 1 et chaque valeur `False` comme 0. En additionnant ces valeurs, `sum()` donne le nombre total de duplications (c'est-à-dire le nombre total de `True` dans la série résultante de `duplicated()`).

**Contexte et Fonctionnement :**

- **But de l'Opération :** L'objectif est de détecter la présence de lignes dupliquées dans le DataFrame `df`, ce qui pourrait indiquer des problèmes dans les données, tels que des entrées redondantes ou des erreurs dans le processus de collecte des données.
- **Résultat de l'Opération :** Le nombre retourné par `df.duplicated().sum()` est le total des lignes dupliquées dans le DataFrame, ce qui aide à évaluer la nécessité de nettoyer ces duplications pour éviter de biaiser les analyses ou les modèles de machine learning.

**Exemple Visuel :**

Supposons que nous avons le DataFrame suivant :

`df` :
| id_annonce | size | nb_rooms | price  |
|------------|------|----------|--------|
| 1          | 50   | 3        | 250000 |
| 2          | 75   | 4        | 320000 |
| 3          | 60   | 2        | 210000 |
| 2          | 75   | 4        | 320000 |  (Duplication de la 2ème ligne)

Lorsque nous exécutons `df.duplicated()`, nous obtenons :

| id_annonce | size | nb_rooms | price  | duplicated |
|------------|------|----------|--------|-------------|
| 1          | 50   | 3        | 250000 | False       |
| 2          | 75   | 4        | 320000 | False       |
| 3          | 60   | 2        | 210000 | False       |
| 2          | 75   | 4        | 320000 | True        |

Ensuite, en exécutant `df.duplicated().sum()`, nous obtenons `1`, car il y a une ligne dupliquée.

En résumé, `df.duplicated().sum()` est un outil pratique pour détecter et compter les duplications dans un DataFrame, facilitant ainsi la gestion de la qualité des données avant de procéder à des analyses ou des modèles prédictifs.


```python
from sklearn.model_selection import train_test_split
seed=123
train_set, test_set = train_test_split(
    df.drop('id_annonce', axis=1), test_size=0.2,
    random_state=seed
)
print("train shape:", train_set.shape, "test shape:", test_set.shape)
```

    train shape: (29894, 27) test shape: (7474, 27)

Ce code utilise la bibliothèque scikit-learn pour diviser un DataFrame en ensembles d'entraînement et de test, ce qui est une étape courante dans les projets de machine learning. Voici une explication détaillée de chaque élément du code :

1. **Importation de la Fonction :**
   ```python
   from sklearn.model_selection import train_test_split
   ```
   - Cela importe la fonction `train_test_split` depuis le module `model_selection` de scikit-learn. Cette fonction est utilisée pour diviser les données en ensembles d'entraînement et de test de manière aléatoire.

2. **Définition de la graine aléatoire :**
   ```python
   seed = 123
   ```
   - La variable `seed` est définie pour stocker la valeur 123. Cette graine est utilisée pour assurer la reproductibilité des résultats en fixant l'état aléatoire de la fonction de division.

3. **Division des Données :**
   ```python
   train_set, test_set = train_test_split(
       df.drop('id_annonce', axis=1), test_size=0.2,
       random_state=seed
   )
   ```
   - **`df.drop('id_annonce', axis=1)` :**
     - Cette partie du code supprime la colonne `id_annonce` du DataFrame `df` avant la division. La colonne `id_annonce` est probablement un identifiant unique qui n'est pas pertinent pour l'entraînement du modèle.
     - `axis=1` indique que nous supprimons une colonne (et non une ligne).

   - **`train_test_split(..., test_size=0.2, random_state=seed)` :**
     - **`test_size=0.2` :** Cela signifie que 20% des données seront utilisées pour l'ensemble de test et 80% pour l'ensemble d'entraînement.
     - **`random_state=seed` :** Cette option fixe la graine aléatoire pour que la division soit reproductible. Chaque exécution du code avec la même graine donnera les mêmes ensembles d'entraînement et de test.

   - **`train_set, test_set` :** Les ensembles d'entraînement et de test résultants sont stockés dans les variables `train_set` et `test_set`, respectivement.

4. **Affichage de la Taille des Ensembles :**
   ```python
   print("train shape:", train_set.shape, "test shape:", test_set.shape)
   ```
   - Cette ligne affiche la forme (le nombre de lignes et de colonnes) des ensembles d'entraînement et de test. Cela permet de vérifier que la division s'est faite correctement et que les proportions spécifiées sont respectées.

**Contexte et Fonctionnement :**

- **But de la Division :** Diviser les données en ensembles d'entraînement et de test est essentiel pour évaluer les performances d'un modèle de machine learning. L'ensemble d'entraînement est utilisé pour ajuster le modèle, tandis que l'ensemble de test est utilisé pour évaluer sa performance sur des données non vues.
- **Reproductibilité :** Utiliser une graine aléatoire (`random_state=seed`) permet de garantir que la division des données est la même à chaque exécution, ce qui est important pour comparer les résultats de différentes expérimentations de manière cohérente.

**Exemple Visuel :**

Supposons que `df` contient 1000 lignes et 10 colonnes (en incluant `id_annonce`). Après exécution de ce code :

- `train_set` contiendra 800 lignes et 9 colonnes (car `id_annonce` a été supprimée).
- `test_set` contiendra 200 lignes et 9 colonnes.

Le résultat de l'affichage sera quelque chose comme :

```
train shape: (800, 9) test shape: (200, 9)
```

En résumé, ce code divise les données en ensembles d'entraînement et de test de manière aléatoire et reproductible, après avoir supprimé une colonne non pertinente (`id_annonce`).



🖥 Analyse exploratoire


```python
real_estate = train_set.copy()
```

    - **Feature Engineering**


      Le **Feature Engineering** est le processus de création de nouvelles caractéristiques (features) ou de transformation des caractéristiques existantes pour améliorer les performances des modèles de machine learning. Il s'agit d'une étape cruciale dans le pipeline de data science car les caractéristiques de haute qualité peuvent rendre les modèles plus précis et robustes. Voici une explication détaillée des principaux aspects du Feature Engineering :

      - Objectifs du Feature Engineering

      1. **Améliorer les Performances du Modèle :**
        - En créant ou en transformant les caractéristiques, on peut fournir des informations plus pertinentes et utiles au modèle, ce qui peut améliorer sa capacité à apprendre et à faire des prédictions précises.

      2. **Réduire la Dimensionalité :**
        - Le Feature Engineering peut inclure la sélection de caractéristiques importantes ou la combinaison de caractéristiques redondantes, ce qui peut réduire la complexité du modèle et améliorer ses performances.

      3. **Gérer les Données Manquantes :**
        - Créer des caractéristiques qui indiquent la présence de données manquantes ou utiliser des techniques pour imputer ces valeurs peut aider à maintenir l'intégrité des données.

      - Techniques Courantes de Feature Engineering

      1. **Création de Nouvelles Caractéristiques :**
        - **Combinaisons de Caractéristiques :** Créer de nouvelles caractéristiques en combinant les existantes, par exemple, en multipliant la taille d'une maison par le nombre de chambres pour obtenir une mesure de densité.
        - **Transformation des Caractéristiques :** Appliquer des transformations mathématiques (logarithmique, racine carrée, etc.) pour modifier la distribution des données.

      2. **Encodage des Variables Catégorielles :**
        - **One-Hot Encoding :** Convertir des variables catégorielles en plusieurs colonnes binaires (0 ou 1).
        - **Label Encoding :** Convertir des catégories en valeurs numériques.

      3. **Gestion des Données Manquantes :**
        - **Imputation :** Remplir les valeurs manquantes avec des moyennes, des médianes, des modes ou des valeurs calculées à partir d'autres caractéristiques.
        - **Indicateurs de Valeurs Manquantes :** Ajouter des colonnes pour indiquer quelles valeurs sont manquantes.

      4. **Normalisation et Standardisation :**
        - **Normalisation :** Mettre les valeurs des caractéristiques sur une échelle commune (par exemple, entre 0 et 1).
        - **Standardisation :** Transformer les caractéristiques pour qu'elles aient une moyenne de 0 et un écart-type de 1.

      5. **Extraction de Caractéristiques :**
        - **Date et Heure :** Extraire des informations comme le jour de la semaine, le mois ou l'heure à partir de dates.
        - **Texte :** Utiliser des techniques de traitement du langage naturel (NLP) pour extraire des caractéristiques des données textuelles.

      6. **Réduction de Dimensionalité :**
        - Utiliser des techniques telles que l'Analyse en Composantes Principales (PCA) pour réduire le nombre de caractéristiques tout en conservant l'essentiel de l'information.

      - Exemple Pratique

      Supposons que nous ayons un ensemble de données sur des transactions immobilières avec les caractéristiques suivantes :

      - `size` (taille de la maison en m²)
      - `nb_rooms` (nombre de pièces)
      - `age` (âge de la maison)
      - `city` (ville)

      Voici quelques exemples de Feature Engineering :

      1. **Création d'une Nouvelle Caractéristique :**
        - `size_per_room` = `size` / `nb_rooms`

      2. **Encodage de la Variable Catégorielle `city` :**
        - Utilisation de One-Hot Encoding pour convertir `city` en plusieurs colonnes binaires.

      3. **Imputation des Données Manquantes :**
        - Remplir les valeurs manquantes de `age` avec la moyenne des âges des maisons dans les mêmes `city`.

      4. **Transformation de la Caractéristique `age` :**
        - Appliquer une transformation logarithmique pour réduire l'effet des valeurs extrêmes :
          - `log_age` = log(`age` + 1)

      En résumé, le Feature Engineering est une étape clé qui consiste à créer et transformer des caractéristiques afin de fournir des données optimisées et pertinentes aux modèles de machine learning, ce qui peut considérablement améliorer leurs performances et leur capacité de généralisation.


```python
real_estate['living_area_per_total_land'] = real_estate['size']/real_estate['land_size']
real_estate['living_area_per_total_land']
```




    10813           NaN
    28163    262.088235
    2732       0.015584
    11636           NaN
    34955           NaN
                ...    
    7763       0.034836
    15377      0.030000
    17730      0.719424
    28030           NaN
    15725      0.226777
    Name: living_area_per_total_land, Length: 29894, dtype: float64




```python
real_estate['total_number_of_rooms'] = real_estate['nb_rooms'] + real_estate['nb_bedrooms'] + real_estate['nb_bathrooms']
real_estate['total_number_of_rooms']
```




    10813    11.0
    28163     8.0
    2732     11.0
    11636     NaN
    34955     NaN
             ... 
    7763      8.0
    15377     NaN
    17730    15.0
    28030     NaN
    15725     8.0
    Name: total_number_of_rooms, Length: 29894, dtype: float64




```python
real_estate['bedrooms_per_room'] = real_estate['nb_bedrooms']/real_estate['total_number_of_rooms']
real_estate['bedrooms_per_room']
```




    10813    0.454545
    28163    0.375000
    2732     0.272727
    11636         NaN
    34955         NaN
               ...   
    7763     0.375000
    15377         NaN
    17730    0.400000
    28030         NaN
    15725    0.375000
    Name: bedrooms_per_room, Length: 29894, dtype: float64




```python
real_estate['total_parking_capacity'] = real_estate['nb_parking_places'] + real_estate['nb_boxes']
real_estate['total_parking_capacity']
```




    10813    1.0
    28163    0.0
    2732     0.0
    11636    0.0
    34955    0.0
            ... 
    7763     0.0
    15377    1.0
    17730    0.0
    28030    1.0
    15725    1.0
    Name: total_parking_capacity, Length: 29894, dtype: float64




```python
real_estate['num_ameneties'] = real_estate['has_a_balcony'] + real_estate['has_a_cellar'] + real_estate['has_a_garage'] + real_estate['has_air_conditioning']
real_estate['num_ameneties']
```




    10813    1.0
    28163    0.0
    2732     0.0
    11636    0.0
    34955    0.0
            ... 
    7763     0.0
    15377    0.0
    17730    1.0
    28030    1.0
    15725    1.0
    Name: num_ameneties, Length: 29894, dtype: float64


Les codes ci-dessus effectuent plusieurs opérations de Feature Engineering sur un DataFrame appelé `real_estate`. De nouvelles caractéristiques dont crées à partir des caractéristiques existantes afin d'améliorer les informations disponibles pour un éventuel modèle de machine learning. Voici une explication détaillée de chaque opération :

1. **Création de la Caractéristique `living_area_per_total_land` :**
   ```python
   real_estate['living_area_per_total_land'] = real_estate['size'] / real_estate['land_size']
   ```
   - **But :** Calculer la proportion de la surface habitable (`size`) par rapport à la taille totale du terrain (`land_size`).
   - **Explication :** Cette nouvelle caractéristique peut donner une idée de l'utilisation du terrain. Une valeur élevée peut indiquer que la maison occupe une grande partie du terrain.

2. **Création de la Caractéristique `total_number_of_rooms` :**
   ```python
   real_estate['total_number_of_rooms'] = real_estate['nb_rooms'] + real_estate['nb_bedrooms'] + real_estate['nb_bathrooms']
   ```
   - **But :** Calculer le nombre total de pièces en additionnant les pièces de vie (`nb_rooms`), les chambres (`nb_bedrooms`) et les salles de bain (`nb_bathrooms`).
   - **Explication :** Cette caractéristique donne une vue d'ensemble du nombre total de pièces dans la maison, ce qui peut être pertinent pour évaluer la taille et la fonctionnalité de la propriété.

3. **Création de la Caractéristique `bedrooms_per_room` :**
   ```python
   real_estate['bedrooms_per_room'] = real_estate['nb_bedrooms'] / real_estate['total_number_of_rooms']
   ```
   - **But :** Calculer la proportion de chambres par rapport au nombre total de pièces.
   - **Explication :** Cette caractéristique peut aider à comprendre la proportion de chambres dans la maison par rapport à toutes les autres pièces, ce qui peut être un indicateur du type de propriété (par exemple, une maison plus familiale ou non).

4. **Création de la Caractéristique `total_parking_capacity` :**
   ```python
   real_estate['total_parking_capacity'] = real_estate['nb_parking_places'] + real_estate['nb_boxes']
   ```
   - **But :** Calculer la capacité totale de stationnement en additionnant les places de parking (`nb_parking_places`) et les box/garages (`nb_boxes`).
   - **Explication :** Cette caractéristique donne une vue d'ensemble de la capacité de stationnement, ce qui peut être un facteur important pour les acheteurs potentiels.

5. **Création de la Caractéristique `num_ameneties` :**
   ```python
   real_estate['num_ameneties'] = real_estate['has_a_balcony'] + real_estate['has_a_cellar'] + real_estate['has_a_garage'] + real_estate['has_air_conditioning']
   ```
   - **But :** Calculer le nombre total de commodités (balcon, cave, garage, climatisation) présentes dans la propriété.
   - **Explication :** Cette caractéristique indique combien de commodités la maison offre, ce qui peut être un indicateur de confort et de qualité de vie.

**Résumé des Objectifs :**

- **`living_area_per_total_land`** : Comprendre l'utilisation du terrain par rapport à la surface habitable.
- **`total_number_of_rooms`** : Obtenir le nombre total de pièces dans la maison.
- **`bedrooms_per_room`** : Évaluer la proportion de chambres par rapport aux autres pièces.
- **`total_parking_capacity`** : Calculer la capacité totale de stationnement.
- **`num_ameneties`** : Quantifier le nombre de commodités disponibles dans la propriété.

En créant ces nouvelles caractéristiques, le DataFrame `real_estate` devient plus riche en informations, ce qui peut aider les modèles de machine learning à mieux comprendre les différentes facettes des propriétés et, potentiellement, à faire des prédictions plus précises sur des aspects comme le prix des biens immobiliers.


```python
from geopy.distance import geodesic

# Paris coordinates (Tour Eiffeil)
paris_latitude = 48.858370
paris_longitude = 2.294481

# Function to calculate distance between Paris and a data point
def calculate_distance(row):
    point_latitude = row['approximate_latitude']
    point_longitude = row['approximate_longitude']
    point_coordinates = (point_latitude, point_longitude)
    paris_coordinates = (paris_latitude, paris_longitude)
    distance = geodesic(point_coordinates, paris_coordinates).km
    return distance

# Apply the function to your dataframe
real_estate['distance_to_paris'] = real_estate.apply(calculate_distance, axis=1)
real_estate['distance_to_paris']
```




    10813     10.223255
    28163    438.637920
    2732     349.514431
    11636    308.683589
    34955    694.013322
                ...    
    7763     230.060205
    15377    192.529959
    17730    343.553036
    28030    305.093606
    15725    421.132323
    Name: distance_to_paris, Length: 29894, dtype: float64



Ce code utilise la bibliothèque `geopy` pour calculer la distance géodésique (c'est-à-dire la distance en ligne droite suivant la courbure de la Terre) entre chaque point de données dans un DataFrame et un point de référence, ici la Tour Eiffel à Paris. Voici une explication détaillée de chaque partie du code :

 Importation de la Bibliothèque

```python
from geopy.distance import geodesic
```
- Cela importe la fonction `geodesic` de la bibliothèque `geopy`, qui est utilisée pour calculer la distance géodésique entre deux points spécifiés par leurs coordonnées (latitude et longitude).

 Définition des Coordonnées de Référence (Tour Eiffel)

```python
# Paris coordinates (Tour Eiffeil)
paris_latitude = 48.858370
paris_longitude = 2.294481
```
- Ces variables stockent les coordonnées géographiques de la Tour Eiffel, qui serviront de point de référence pour les calculs de distance.

 Définition de la Fonction de Calcul de la Distance

```python
# Function to calculate distance between Paris and a data point
def calculate_distance(row):
    point_latitude = row['approximate_latitude']
    point_longitude = row['approximate_longitude']
    point_coordinates = (point_latitude, point_longitude)
    paris_coordinates = (paris_latitude, paris_longitude)
    distance = geodesic(point_coordinates, paris_coordinates).km
    return distance
```
- **`calculate_distance(row)` :** Cette fonction prend une ligne (row) du DataFrame comme argument.
  - **`point_latitude` et `point_longitude` :** Ces variables extraient les coordonnées de latitude et de longitude de la ligne actuelle.
  - **`point_coordinates` :** Tuple contenant les coordonnées de la ligne actuelle.
  - **`paris_coordinates` :** Tuple contenant les coordonnées de la Tour Eiffel.
  - **`distance` :** La fonction `geodesic` calcule la distance en kilomètres entre `point_coordinates` et `paris_coordinates`.
  - **`return distance` :** La fonction retourne la distance calculée.

 Application de la Fonction au DataFrame

```python
# Apply the function to your dataframe
real_estate['distance_to_paris'] = real_estate.apply(calculate_distance, axis=1)
```
- **`real_estate.apply(calculate_distance, axis=1)` :** La méthode `apply` de pandas est utilisée pour appliquer la fonction `calculate_distance` à chaque ligne du DataFrame `real_estate`.
  - **`axis=1` :** Indique que la fonction doit être appliquée sur les lignes.
- **`real_estate['distance_to_paris']` :** Une nouvelle colonne `distance_to_paris` est ajoutée au DataFrame, contenant les distances calculées pour chaque point de données par rapport à la Tour Eiffel.

 Contexte et Utilité

- **But de cette Opération :** Calculer la distance de chaque propriété par rapport à un point de référence (la Tour Eiffel) pour éventuellement utiliser cette information comme caractéristique dans un modèle de machine learning. La distance à une grande ville ou un point de repère peut être un indicateur important pour le prix d'un bien immobilier.
- **Utilisation Potentielle :** Cette nouvelle caractéristique `distance_to_paris` peut aider à comprendre l'influence de la proximité à Paris sur le prix des biens immobiliers.

 Exemple Pratique

Supposons que le DataFrame `real_estate` ait les colonnes `approximate_latitude` et `approximate_longitude` pour chaque propriété :

| approximate_latitude | approximate_longitude |
|----------------------|-----------------------|
| 48.8566              | 2.3522                |
| 48.8530              | 2.3499                |
| ...                  | ...                   |

Après application de la fonction, une nouvelle colonne `distance_to_paris` sera ajoutée :

| approximate_latitude | approximate_longitude | distance_to_paris |
|----------------------|-----------------------|-------------------|
| 48.8566              | 2.3522                | 4.5               |
| 48.8530              | 2.3499                | 5.0               |
| ...                  | ...                   | ...               |

En résumé, ce code ajoute une nouvelle caractéristique `distance_to_paris` au DataFrame `real_estate`, qui contient la distance en kilomètres de chaque propriété à la Tour Eiffel, apportant ainsi une information géographique potentiellement utile pour l'analyse ou la modélisation des données.


```python
real_estate[['city', 'distance_to_paris']].sample(5)
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
      <th>city</th>
      <th>distance_to_paris</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5577</th>
      <td>paris-9eme</td>
      <td>3.999960</td>
    </tr>
    <tr>
      <th>10521</th>
      <td>gien</td>
      <td>132.357926</td>
    </tr>
    <tr>
      <th>18339</th>
      <td>l'hay-les-roses</td>
      <td>8.212950</td>
    </tr>
    <tr>
      <th>26581</th>
      <td>montreuil</td>
      <td>10.707766</td>
    </tr>
    <tr>
      <th>5741</th>
      <td>lacaune</td>
      <td>573.011402</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Résumé statistique
real_estate.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>approximate_latitude</th>
      <td>29894.0</td>
      <td>46.554705</td>
      <td>2.354314</td>
      <td>41.374436</td>
      <td>43.930923</td>
      <td>46.965949</td>
      <td>48.842526</td>
      <td>5.104601e+01</td>
    </tr>
    <tr>
      <th>approximate_longitude</th>
      <td>29894.0</td>
      <td>2.607762</td>
      <td>2.592422</td>
      <td>-4.733545</td>
      <td>1.089791</td>
      <td>2.378397</td>
      <td>4.567152</td>
      <td>9.483665e+00</td>
    </tr>
    <tr>
      <th>postal_code</th>
      <td>29894.0</td>
      <td>53712.291898</td>
      <td>28786.435880</td>
      <td>1000.000000</td>
      <td>30210.000000</td>
      <td>59000.000000</td>
      <td>78230.000000</td>
      <td>9.588000e+04</td>
    </tr>
    <tr>
      <th>size</th>
      <td>29477.0</td>
      <td>1095.707060</td>
      <td>5629.776972</td>
      <td>1.000000</td>
      <td>73.000000</td>
      <td>115.000000</td>
      <td>239.000000</td>
      <td>4.113110e+05</td>
    </tr>
    <tr>
      <th>floor</th>
      <td>7813.0</td>
      <td>3.457827</td>
      <td>6.628996</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>5.500000e+01</td>
    </tr>
    <tr>
      <th>land_size</th>
      <td>12424.0</td>
      <td>4005.178686</td>
      <td>59071.859008</td>
      <td>1.000000</td>
      <td>363.000000</td>
      <td>797.000000</td>
      <td>1836.250000</td>
      <td>6.203700e+06</td>
    </tr>
    <tr>
      <th>energy_performance_value</th>
      <td>15210.0</td>
      <td>206.996121</td>
      <td>872.853538</td>
      <td>0.000000</td>
      <td>125.000000</td>
      <td>180.000000</td>
      <td>240.000000</td>
      <td>1.000000e+05</td>
    </tr>
    <tr>
      <th>ghg_value</th>
      <td>14791.0</td>
      <td>32.323372</td>
      <td>322.041271</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>16.000000</td>
      <td>36.000000</td>
      <td>1.702400e+04</td>
    </tr>
    <tr>
      <th>nb_rooms</th>
      <td>28634.0</td>
      <td>4.234826</td>
      <td>2.935642</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>1.250000e+02</td>
    </tr>
    <tr>
      <th>nb_bedrooms</th>
      <td>27707.0</td>
      <td>2.854946</td>
      <td>2.147069</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>1.180000e+02</td>
    </tr>
    <tr>
      <th>nb_bathrooms</th>
      <td>19277.0</td>
      <td>0.919956</td>
      <td>0.271941</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <th>nb_parking_places</th>
      <td>29894.0</td>
      <td>0.293504</td>
      <td>0.455375</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>nb_boxes</th>
      <td>29894.0</td>
      <td>0.178999</td>
      <td>0.383358</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>nb_photos</th>
      <td>29894.0</td>
      <td>7.972637</td>
      <td>4.649109</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>5.000000e+01</td>
    </tr>
    <tr>
      <th>has_a_balcony</th>
      <td>29894.0</td>
      <td>0.149194</td>
      <td>0.356285</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>nb_terraces</th>
      <td>29894.0</td>
      <td>0.305914</td>
      <td>0.460801</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>has_a_cellar</th>
      <td>29894.0</td>
      <td>0.201177</td>
      <td>0.400887</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>has_a_garage</th>
      <td>29894.0</td>
      <td>0.053857</td>
      <td>0.225739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>has_air_conditioning</th>
      <td>29894.0</td>
      <td>0.039306</td>
      <td>0.194324</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>last_floor</th>
      <td>29894.0</td>
      <td>0.003680</td>
      <td>0.060550</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>upper_floors</th>
      <td>29894.0</td>
      <td>0.000268</td>
      <td>0.016357</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>price</th>
      <td>29894.0</td>
      <td>341586.581388</td>
      <td>308223.958671</td>
      <td>24465.000000</td>
      <td>155000.000000</td>
      <td>255000.000000</td>
      <td>412375.000000</td>
      <td>2.299000e+06</td>
    </tr>
    <tr>
      <th>living_area_per_total_land</th>
      <td>12420.0</td>
      <td>3.566579</td>
      <td>68.298408</td>
      <td>0.000015</td>
      <td>0.090015</td>
      <td>0.196907</td>
      <td>0.490631</td>
      <td>5.039000e+03</td>
    </tr>
    <tr>
      <th>total_number_of_rooms</th>
      <td>18736.0</td>
      <td>8.011422</td>
      <td>4.857124</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>2.370000e+02</td>
    </tr>
    <tr>
      <th>bedrooms_per_room</th>
      <td>17208.0</td>
      <td>0.335426</td>
      <td>0.057611</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>0.333333</td>
      <td>0.375000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>total_parking_capacity</th>
      <td>29894.0</td>
      <td>0.472503</td>
      <td>0.598309</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <th>num_ameneties</th>
      <td>29894.0</td>
      <td>0.443534</td>
      <td>0.658244</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <th>distance_to_paris</th>
      <td>29894.0</td>
      <td>336.492172</td>
      <td>244.442169</td>
      <td>0.449444</td>
      <td>86.167824</td>
      <td>362.702396</td>
      <td>575.148604</td>
      <td>9.907175e+02</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Table de fréquence des variables qualitatives
for col in list(real_estate.select_dtypes(include='object')):
  print('-----------Colonne :', col, 'avec ', real_estate[col].nunique(), ' modalités-------------')
  print(real_estate[col].value_counts(normalize=True))
  print('\n')
```

    -----------Colonne : property_type avec  22  modalités-------------
    property_type
    appartement          0.421757
    maison               0.419315
    divers               0.057403
    terrain              0.041614
    villa                0.020272
    propriété            0.010470
    terrain à bâtir      0.007794
    duplex               0.005921
    viager               0.004616
    ferme                0.003512
    parking              0.002710
    loft                 0.001338
    chalet               0.001338
    château              0.000602
    moulin               0.000368
    manoir               0.000335
    péniche              0.000201
    hôtel particulier    0.000167
    chambre              0.000134
    gîte                 0.000067
    hôtel                0.000033
    atelier              0.000033
    Name: proportion, dtype: float64
    
    
    -----------Colonne : city avec  7702  modalités-------------
    city
    toulouse                0.012544
    montpellier             0.010035
    paris-16eme             0.006590
    paris-17eme             0.006322
    paris-18eme             0.006055
                              ...   
    villelaure              0.000033
    la-forest-landerneau    0.000033
    ners                    0.000033
    briquenay               0.000033
    saulgond                0.000033
    Name: proportion, Length: 7702, dtype: float64
    
    
    -----------Colonne : energy_performance_category avec  7  modalités-------------
    energy_performance_category
    D    0.366601
    C    0.215713
    E    0.201118
    B    0.112755
    F    0.054635
    A    0.032479
    G    0.016700
    Name: proportion, dtype: float64
    
    
    -----------Colonne : ghg_category avec  7  modalités-------------
    ghg_category
    B    0.242648
    C    0.184910
    D    0.176594
    E    0.160773
    A    0.137313
    F    0.069705
    G    0.028058
    Name: proportion, dtype: float64
    
    
    -----------Colonne : exposition avec  12  modalités-------------
    exposition
    Sud           0.355760
    Sud-Ouest     0.149252
    Sud-Est       0.104902
    Est-Ouest     0.090897
    Ouest         0.090485
    Est           0.076617
    Sud-Nord      0.038446
    Nord-Ouest    0.037073
    Nord          0.025127
    Nord-Est      0.024715
    Ouest-Est     0.003845
    Nord-Sud      0.002883
    Name: proportion, dtype: float64
    
    

Ce code crée et affiche des tableaux de fréquence pour chaque variable qualitative (ou catégorielle) dans le DataFrame `real_estate`. Voici une explication détaillée de chaque partie du code :

 Sélection des Colonnes Qualitatives

```python
real_estate.select_dtypes(include='object')
```
- **`real_estate.select_dtypes(include='object')` :** Cette méthode de pandas sélectionne toutes les colonnes du DataFrame `real_estate` qui ont un type de données 'object'. En pandas, les colonnes de type 'object' sont généralement utilisées pour représenter des variables qualitatives ou catégorielles.

 Boucle sur les Colonnes Qualitatives

```python
for col in list(real_estate.select_dtypes(include='object')):
```
- **`for col in list(...)` :** La boucle `for` itère sur la liste des noms des colonnes qualitatives sélectionnées.
- **`col` :** À chaque itération, `col` représente le nom d'une colonne qualitative dans le DataFrame `real_estate`.

 Affichage des Informations pour Chaque Colonne

   Impression de l'En-tête de la Colonne

  ```python
  print('-----------Colonne :', col, 'avec ', real_estate[col].nunique(), ' modalités-------------')
  ```
  - **`print(...)` :** Affiche le nom de la colonne actuelle (`col`), ainsi que le nombre de modalités (ou valeurs distinctes) qu'elle contient.
  - **`real_estate[col].nunique()` :** Retourne le nombre de valeurs uniques dans la colonne `col`.

   Affichage des Fréquences des Modalités

  ```python
  print(real_estate[col].value_counts(normalize=True))
  ```
  - **`real_estate[col].value_counts(normalize=True)` :** Cette méthode compte le nombre d'occurrences de chaque valeur dans la colonne `col` et normalise les résultats pour obtenir des proportions (fréquences relatives) au lieu de simples comptages.
  - **`normalize=True` :** Indique que les fréquences relatives (proportions) doivent être calculées.

   Ajout d'une Ligne Blanche pour la Lisibilité

  ```python
  print('\n')
  ```
  - **`print('\n')` :** Ajoute une ligne blanche après l'affichage des fréquences pour améliorer la lisibilité des résultats.

 Exécution du Code

Pour illustrer avec un exemple pratique, supposons que le DataFrame `real_estate` contient deux colonnes qualitatives : `city` et `property_type`.

1. **Sélection des colonnes qualitatives :**
   ```python
   real_estate.select_dtypes(include='object')
   ```
   Imaginons que cela retourne les colonnes `['city', 'property_type']`.

2. **Boucle sur les colonnes :**
   ```python
   for col in list(real_estate.select_dtypes(include='object')):
   ```

3. **Première itération (`col = 'city'`) :**
   - **Affichage de l'en-tête :**
     ```python
     print('-----------Colonne :', col, 'avec ', real_estate[col].nunique(), ' modalités-------------')
     ```
     Supposons que `real_estate['city'].nunique()` retourne `3`.

     **Sortie :**
     ```
     -----------Colonne : city avec  3  modalités-------------
     ```

   - **Affichage des fréquences :**
     ```python
     print(real_estate[col].value_counts(normalize=True))
     ```
     Supposons que `real_estate['city'].value_counts(normalize=True)` retourne :
     ```
     Paris     0.5
     Lyon      0.3
     Marseille 0.2
     ```

     **Sortie :**
     ```
     Paris        0.5
     Lyon         0.3
     Marseille    0.2
     ```

   - **Ligne blanche :**
     ```python
     print('\n')
     ```

4. **Deuxième itération (`col = 'property_type'`) :**
   - **Affichage de l'en-tête :**
     ```python
     print('-----------Colonne :', col, 'avec ', real_estate[col].nunique(), ' modalités-------------')
     ```
     Supposons que `real_estate['property_type'].nunique()` retourne `2`.

     **Sortie :**
     ```
     -----------Colonne : property_type avec  2  modalités-------------
     ```

   - **Affichage des fréquences :**
     ```python
     print(real_estate[col].value_counts(normalize=True))
     ```
     Supposons que `real_estate['property_type'].value_counts(normalize=True)` retourne :
     ```
     Apartment    0.7
     House        0.3
     ```

     **Sortie :**
     ```
     Apartment    0.7
     House        0.3
     ```

   - **Ligne blanche :**
     ```python
     print('\n')
     ```

En résumé, ce code affiche pour chaque variable qualitative dans le DataFrame `real_estate` le nombre de modalités et les fréquences relatives de chaque modalité, ce qui est utile pour comprendre la distribution des valeurs catégorielles et identifier les catégories dominantes.



```python
# Histogrammes
real_estate.hist(bins=50, figsize=(18, 13));
```


    
![png](french_real_estate_prediction_files/french_real_estate_prediction_22_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='approximate_longitude', y='approximate_latitude', data=real_estate)
plt.xlabel('approximate_longitude')
plt.ylabel('approximate_latitude')
plt.title('Scatter plot of real estate data')
plt.show()

```


    
![png](french_real_estate_prediction_files/french_real_estate_prediction_23_0.png)
    



```python
sns.scatterplot(x='approximate_longitude', 
                y='approximate_latitude', 
                size='price', 
                hue='price', 
                data=real_estate)
plt.xlabel('approximate_longitude')
plt.ylabel('approximate_latitude')
plt.title('Scatter plot of real estate data')
plt.legend(title='Price')
plt.show()
```


    
![png](french_real_estate_prediction_files/french_real_estate_prediction_24_0.png)
    



```python
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='approximate_longitude', y='approximate_latitude', 
                size='price', hue='property_type', 
                data=real_estate, palette='tab10')
plt.xlabel('approximate_longitude')
plt.ylabel('approximate_latitude')
plt.title('Scatter plot of real estate data')
plt.legend(title='Property Type')
plt.show()

```


    
![png](french_real_estate_prediction_files/french_real_estate_prediction_25_0.png)
    



```python

filtered_data = real_estate[(real_estate['property_type'] == 'appartement') | (real_estate['property_type'] == 'maison')]
plt.figure(figsize=(10, 6))  # Définition de la taille de la figure
sns.scatterplot(x='approximate_longitude', y='approximate_latitude', 
                size='price', hue='property_type', 
                data=filtered_data, palette='tab10')
plt.xlabel('approximate_longitude')
plt.ylabel('approximate_latitude')
plt.title('Scatter plot of real estate data (Apartments and Houses)')
plt.legend(title='Property Type')
plt.show()
```


Ce code effectue les opérations suivantes pour créer un graphique de dispersion (scatter plot) qui visualise la distribution géographique des biens immobiliers, en mettant en évidence la différence entre les appartements et les maisons. Voici une explication détaillée de chaque étape :

 Filtrage des Données

```python
filtered_data = real_estate[(real_estate['property_type'] == 'appartement') | (real_estate['property_type'] == 'maison')]
```
- **`real_estate[...]`** : Cette expression sélectionne les lignes du DataFrame `real_estate` où la colonne `property_type` est soit "appartement" soit "maison".
- **`filtered_data`** : Le résultat de cette sélection est stocké dans une nouvelle variable `filtered_data`, qui contient uniquement les biens immobiliers de type appartement et maison.

 Configuration de la Taille de la Figure

```python
plt.figure(figsize=(10, 6))  # Définition de la taille de la figure
```
- **`plt.figure(figsize=(10, 6))`** : Cette fonction de Matplotlib définit la taille de la figure du graphique, ici de 10 pouces de large et 6 pouces de haut.

 Création du Scatter Plot

```python
sns.scatterplot(x='approximate_longitude', y='approximate_latitude', 
                size='price', hue='property_type', 
                data=filtered_data, palette='tab10')
```
- **`sns.scatterplot(...)`** : Cette fonction de Seaborn crée un graphique de dispersion.
  - **`x='approximate_longitude'`** : Les coordonnées longitudinales des propriétés sont utilisées pour l'axe des x.
  - **`y='approximate_latitude'`** : Les coordonnées latitudinales des propriétés sont utilisées pour l'axe des y.
  - **`size='price'`** : La taille des points sur le graphique est proportionnelle au prix des propriétés.
  - **`hue='property_type'`** : Les points sont colorés selon le type de propriété (appartement ou maison).
  - **`data=filtered_data`** : Le DataFrame filtré est utilisé comme source de données.
  - **`palette='tab10'`** : Une palette de couleurs spécifique est utilisée pour différencier les types de propriétés.

 Ajout des Étiquettes et du Titre

```python
plt.xlabel('approximate_longitude')
plt.ylabel('approximate_latitude')
plt.title('Scatter plot of real estate data (Apartments and Houses)')
```
- **`plt.xlabel('approximate_longitude')`** : Définit l'étiquette de l'axe des x comme "approximate_longitude".
- **`plt.ylabel('approximate_latitude')`** : Définit l'étiquette de l'axe des y comme "approximate_latitude".
- **`plt.title('Scatter plot of real estate data (Apartments and Houses)')`** : Définit le titre du graphique.

 Affichage de la Légende

```python
plt.legend(title='Property Type')
```
- **`plt.legend(title='Property Type')`** : Ajoute une légende au graphique avec le titre "Property Type" pour expliquer les couleurs correspondant aux types de propriétés.

 Affichage du Graphique

```python
plt.show()
```
- **`plt.show()`** : Affiche le graphique.

 Contexte et Utilité

- **But du Graphique :** Visualiser la répartition géographique des appartements et des maisons, avec une indication visuelle des prix des propriétés.
- **Utilisation Potentielle :** Ce graphique peut aider à identifier des tendances géographiques, comme des concentrations de propriétés de certains types ou des zones où les prix sont plus élevés.

En résumé, ce code crée un graphique de dispersion pour visualiser la localisation géographique des biens immobiliers en France, en mettant en évidence les différences entre les appartements et les maisons et en représentant le prix des propriétés par la taille des points.


```python
import numpy as np
plt.figure(figsize=(18,10))

# Exclure les colonnes non numériques
numeric_real_estate = real_estate.select_dtypes(include=[np.number])

real_estate_corr = numeric_real_estate.corr()
mask = np.triu(np.ones_like(real_estate_corr, dtype=bool))
sns.heatmap(
    real_estate_corr, mask=mask, center=0,
    cmap='RdBu', linewidths=1, annot=True,
    fmt=".2f", vmin=-1, vmax=1
)
plt.title("Carte des corrélations")
plt.show()
```

Ce code utilise la bibliothèque NumPy et la bibliothèque de visualisation Matplotlib avec son module pyplot pour créer une carte de chaleur (heatmap) des corrélations entre les différentes variables numériques d'un ensemble de données sur l'immobilier.

Voici une explication ligne par ligne :

1. `import numpy as np`: Importe la bibliothèque NumPy sous l'alias np.
2. `import matplotlib.pyplot as plt`: Importe la bibliothèque Matplotlib avec son module pyplot sous l'alias plt.
3. `plt.figure(figsize=(18,10))`: Crée une nouvelle figure avec une taille de 18 pouces de largeur et 10 pouces de hauteur.
4. `numeric_real_estate = real_estate.select_dtypes(include=[np.number])`: Sélectionne uniquement les colonnes numériques de l'ensemble de données `real_estate` (qui est supposé être défini ailleurs dans le code mais n'est pas présent dans l'extrait que vous avez fourni).
5. `real_estate_corr = numeric_real_estate.corr()`: Calcule la matrice des corrélations entre les variables numériques de l'ensemble de données.
6. `mask = np.triu(np.ones_like(real_estate_corr, dtype=bool))`: Crée un masque pour masquer la moitié supérieure de la carte de chaleur, car la matrice des corrélations est symétrique par rapport à sa diagonale.
7. `sns.heatmap(...)` : Trace une carte de chaleur des corrélations en utilisant la fonction heatmap de la bibliothèque Seaborn (qui n'a pas été importée dans le code fourni mais qui est nécessaire pour cette ligne). Les arguments spécifiés incluent la matrice de corrélation, le masque pour la moitié supérieure, le centre de la colormap, la colormap utilisée (RdBu), l'épaisseur des lignes, l'annotation des valeurs sur la carte, le format des annotations, les valeurs minimales et maximales pour la colormap.
8. `plt.title("Carte des corrélations")`: Définit le titre de la carte de chaleur.
9. `plt.show()`: Affiche la carte de chaleur.

Ce code est utile pour visualiser les relations linéaires entre les variables numériques d'un ensemble de données et peut aider à identifier des modèles ou des tendances.
    

```python
for col in list(real_estate.select_dtypes(include='object').drop('city', axis=1)):
  print("-----------------price VS", col, "------------------------")
  print(real_estate.groupby(col)['price'].mean().sort_values())
  sns.boxplot(data=real_estate, x='price', y=col)
  plt.show()
  plt.close()
```

Ce code parcourt les colonnes de type "objet" (c'est-à-dire les colonnes catégoriques) d'un ensemble de données `real_estate` à l'exception de la colonne 'city'. Pour chaque colonne, il affiche les statistiques de prix (en moyenne) par catégorie de cette colonne et trace également une boîte à moustaches (boxplot) pour visualiser la distribution des prix par catégorie.

Voici une explication ligne par ligne :

1. `for col in list(real_estate.select_dtypes(include='object').drop('city', axis=1)):`: Parcourt chaque colonne de l'ensemble de données `real_estate` qui est de type "objet" (c'est-à-dire catégorique), à l'exception de la colonne 'city'.
   
2. `print("-----------------price VS", col, "------------------------")`: Affiche une ligne de séparation et le nom de la colonne actuellement analysée.

3. `print(real_estate.groupby(col)['price'].mean().sort_values())`: Calcule la moyenne des prix (`price`) pour chaque catégorie de la colonne actuellement analysée (`col`) et trie les résultats par ordre croissant. Cela affiche les moyennes des prix pour chaque catégorie de la colonne.

4. `sns.boxplot(data=real_estate, x='price', y=col)`: Trace un boxplot avec les prix (`price`) en tant que variable indépendante et la colonne catégorique actuellement analysée (`col`) en tant que variable dépendante.

5. `plt.show()`: Affiche le boxplot.

6. `plt.close()`: Ferme la figure après l'avoir affichée. Cela est nécessaire car sans cette instruction, la boucle continuerait à tracer sur la même figure, superposant les graphiques précédents.

Ce code est utile pour visualiser la relation entre les variables catégoriques et les prix dans un ensemble de données, ce qui peut aider à identifier des tendances ou des différences significatives dans les prix en fonction des catégories.



    -----------------price VS property_type ------------------------
    property_type
    parking              5.801333e+04
    chambre              1.063750e+05
    terrain à bâtir      1.083508e+05
    terrain              1.218439e+05
    viager               1.785017e+05
    divers               3.059696e+05
    ferme                3.214845e+05
    maison               3.255814e+05
    appartement          3.720782e+05
    chalet               3.986384e+05
    duplex               4.086984e+05
    moulin               4.706545e+05
    hôtel                4.910000e+05
    loft                 5.049762e+05
    villa                5.257610e+05
    gîte                 5.550000e+05
    propriété            6.504257e+05
    péniche              6.925000e+05
    hôtel particulier    7.083800e+05
    manoir               7.822050e+05
    château              1.078403e+06
    atelier              1.300000e+06
    Name: price, dtype: float64



    
![png](french_real_estate_prediction_files/french_real_estate_prediction_28_1.png)
    


    -----------------price VS energy_performance_category ------------------------
    energy_performance_category
    G    216562.129921
    F    246950.354994
    E    303041.066362
    D    352285.848458
    A    384611.190283
    C    434601.926242
    B    439656.787755
    Name: price, dtype: float64



    
![png](french_real_estate_prediction_files/french_real_estate_prediction_28_3.png)
    


    -----------------price VS ghg_category ------------------------
    ghg_category
    G    279869.831325
    F    339485.526673
    C    355154.803291
    B    355794.675676
    E    377837.122372
    A    379216.728705
    D    387317.205590
    Name: price, dtype: float64



    
![png](french_real_estate_prediction_files/french_real_estate_prediction_28_5.png)
    


    -----------------price VS exposition ------------------------
    exposition
    Nord          306179.398907
    Nord-Ouest    321697.211111
    Sud-Nord      358697.864286
    Ouest         366705.218513
    Est           367180.910394
    Sud           392291.823234
    Sud-Est       402456.744764
    Est-Ouest     405793.345921
    Nord-Est      435204.138889
    Sud-Ouest     441905.737810
    Nord-Sud      496062.142857
    Ouest-Est     512836.821429
    Name: price, dtype: float64



    
![png](french_real_estate_prediction_files/french_real_estate_prediction_28_7.png)
    



🖥 **Data preprocessing : Prétraitement des données**


```python
trainset = train_set.copy()
```


```python
X_train = trainset.drop("price", axis=1)
y_train = trainset['price']
```

    - Pipeline de prétraitement des variables numériques


```python
# variables numériques
vars_num = list(trainset.drop('price', axis=1).select_dtypes(exclude='object'))
vars_num
```




    ['approximate_latitude',
     'approximate_longitude',
     'postal_code',
     'size',
     'floor',
     'land_size',
     'energy_performance_value',
     'ghg_value',
     'nb_rooms',
     'nb_bedrooms',
     'nb_bathrooms',
     'nb_parking_places',
     'nb_boxes',
     'nb_photos',
     'has_a_balcony',
     'nb_terraces',
     'has_a_cellar',
     'has_a_garage',
     'has_air_conditioning',
     'last_floor',
     'upper_floors']




```python
# Fonction d'ajout de nouvelles variables
def add_features(Z):
  Z['living_area_per_total_land'] = Z['size']/Z['land_size']
  Z['total_number_of_rooms'] = Z['nb_rooms'] + Z['nb_bedrooms'] + Z['nb_bathrooms']
  Z['bedrooms_per_room'] = Z['nb_bedrooms']/Z['total_number_of_rooms']
  Z['total_parking_capacity'] = Z['nb_parking_places'] + Z['nb_boxes']
  Z['num_ameneties'] = Z['has_a_balcony'] + Z['has_a_cellar'] + Z['has_a_garage'] + Z['has_air_conditioning']
  Z['distance_to_paris'] = Z.apply(calculate_distance, axis=1)
  return Z.values

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline([
    ('feats_adder', FunctionTransformer(add_features)),
    ('num_impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

num_pipeline
```


Ce code définit un pipeline de transformation pour prétraiter les caractéristiques numériques d'un ensemble de données. Voici une explication ligne par ligne :

1. **Définition de la fonction d'ajout de nouvelles variables** (`add_features`):
   - Cette fonction prend un DataFrame `Z` en entrée et ajoute plusieurs nouvelles variables basées sur les caractéristiques existantes du DataFrame.
   - Les nouvelles variables ajoutées sont :
     - `living_area_per_total_land` : la surface habitable par rapport à la taille totale du terrain.
     - `total_number_of_rooms` : le nombre total de pièces, qui est la somme des chambres, des salles de bains et des salons.
     - `bedrooms_per_room` : le ratio de chambres par rapport au nombre total de pièces.
     - `total_parking_capacity` : la capacité totale de stationnement, qui est la somme des places de parking et des garages.
     - `num_ameneties` : le nombre total d'équipements disponibles, tels que le balcon, la cave, le garage et la climatisation.
     - `distance_to_paris` : la distance à Paris, calculée à partir d'une fonction `calculate_distance` appliquée à chaque ligne du DataFrame.
   - La fonction retourne les valeurs du DataFrame transformé.

2. **Importation des modules nécessaires** :
   - `from sklearn.pipeline import Pipeline`: Importe la classe `Pipeline` de scikit-learn pour créer un pipeline de transformation.
   - `from sklearn.preprocessing import FunctionTransformer, StandardScaler`: Importe des transformateurs de scikit-learn pour appliquer des transformations personnalisées et standardiser les caractéristiques numériques.
   - `from sklearn.impute import SimpleImputer`: Importe la classe `SimpleImputer` pour gérer les valeurs manquantes dans les données.

3. **Création du pipeline** (`num_pipeline`):
   - `Pipeline([...])`: Crée un pipeline de transformation pour les caractéristiques numériques.
   - Les étapes du pipeline sont :
     - `('feats_adder', FunctionTransformer(add_features))`: Applique la fonction `add_features` pour ajouter de nouvelles variables aux données.
     - `('num_impute', SimpleImputer(strategy='median'))`: Impute les valeurs manquantes en utilisant la médiane des valeurs existantes.
     - `('scaler', StandardScaler())`: Standardise les caractéristiques en soustrayant la moyenne et en divisant par l'écart-type pour centrer les données autour de zéro et mettre à l'échelle pour avoir une variance unitaire.

4. **Affichage du pipeline** :
   - `num_pipeline`: Affiche le pipeline créé.

Ce pipeline peut être utilisé pour prétraiter les caractéristiques numériques d'un ensemble de données, en ajoutant de nouvelles variables, en imputant les valeurs manquantes et en standardisant les données.


    - Pipeline de prétraitement des variables catégorielles


```python
# Variables catégorielles
vars_cat = list(trainset.select_dtypes(include='object').drop('city', axis=1))
vars_cat
```




    ['property_type', 'energy_performance_category', 'ghg_category', 'exposition']




```python
cat_pipeline = Pipeline([
    ('cat_imputer', SimpleImputer(strategy='constant', fill_value="UNKNOWN")),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=0.002))
])
cat_pipeline
```

Ce code crée un pipeline de transformation pour prétraiter les caractéristiques catégoriques d'un ensemble de données. Voici une explication ligne par ligne :

1. **Sélection des caractéristiques catégoriques** :
   - `vars_cat = list(trainset.select_dtypes(include='object').drop('city', axis=1))`: Sélectionne les noms des colonnes catégoriques de l'ensemble de données `trainset`, à l'exception de la colonne 'city'.

2. **Création du pipeline de transformation** (`cat_pipeline`) :
   - `Pipeline([...])`: Crée un pipeline de transformation pour les caractéristiques catégoriques.
   - Les étapes du pipeline sont :
     - `('cat_imputer', SimpleImputer(strategy='constant', fill_value="UNKNOWN"))`: Utilise l'imputation constante pour remplacer les valeurs manquantes par la chaîne de caractères "UNKNOWN". Cela signifie que toute valeur manquante dans les caractéristiques catégoriques sera remplacée par "UNKNOWN".
     - `('encoder', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=0.002))`: Utilise un encodeur one-hot pour transformer les caractéristiques catégoriques en variables binaires. Les arguments spécifiés sont :
        - `sparse_output=False` : pour obtenir une sortie dense plutôt que sparse.
        - `handle_unknown='infrequent_if_exist'` : spécifie comment gérer les catégories inconnues lors de la transformation. Si une catégorie inconnue est rencontrée pendant la transformation, elle sera traitée comme une catégorie rare.
        - `min_frequency=0.002` : spécifie la fréquence minimale d'apparition d'une catégorie pour qu'elle soit considérée comme fréquente. Les catégories dont la fréquence est inférieure à cette valeur seront traitées comme des catégories rares.

Ce pipeline peut être utilisé pour prétraiter les caractéristiques catégoriques d'un ensemble de données en remplaçant les valeurs manquantes par "UNKNOWN" et en les encodant en variables binaires à l'aide d'un encodage one-hot.


    - Combinaison des deux pipelines en un seul pipeline


```python
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, vars_num),
    ("cat", cat_pipeline, vars_cat)
])
preprocessor
```

Ce code utilise la classe `ColumnTransformer` de scikit-learn pour créer un préprocesseur qui applique des transformations spécifiques aux colonnes numériques et catégoriques d'un ensemble de données. Voici une explication ligne par ligne :

1. **Importation du module nécessaire** :
   - `from sklearn.compose import ColumnTransformer`: Importe la classe `ColumnTransformer` de scikit-learn, qui permet de transformer différentes colonnes de l'ensemble de données de manière spécifique.

2. **Création du préprocesseur** (`preprocessor`) :
   - `preprocessor = ColumnTransformer([...])`: Crée un préprocesseur qui applique des transformations spécifiques aux différentes colonnes de l'ensemble de données.
   - Les étapes du préprocesseur sont définies dans une liste, où chaque élément est un tuple contenant trois éléments :
     - Le premier élément du tuple est une chaîne de caractères qui identifie la transformation.
     - Le deuxième élément est le transformateur à appliquer sur les colonnes sélectionnées.
     - Le troisième élément est une liste des noms des colonnes sur lesquelles appliquer la transformation.
   
3. **Transformation des colonnes numériques** :
   - `("num", num_pipeline, vars_num)`: Applique le pipeline de transformation `num_pipeline` (défini précédemment) sur les colonnes numériques. `vars_num` est une liste des noms des colonnes numériques sélectionnées.
   
4. **Transformation des colonnes catégoriques** :
   - `("cat", cat_pipeline, vars_cat)`: Applique le pipeline de transformation `cat_pipeline` (également défini précédemment) sur les colonnes catégoriques. `vars_cat` est une liste des noms des colonnes catégoriques sélectionnées.

En combinant ces transformations dans un seul `ColumnTransformer`, le préprocesseur peut être utilisé pour appliquer toutes les étapes de prétraitement nécessaires sur les colonnes numériques et catégoriques de l'ensemble de données en une seule opération.


```python
# Application aux données d'entrainement
X_train_prepared = preprocessor.fit_transform(X_train)
print(X_train_prepared.shape)
X_train_prepared
```

    (29894, 67)





    array([[ 0.98030347, -0.06715911,  1.36829606, ...,  0.        ,
             1.        ,  0.        ],
           [-0.39282477, -1.40638835, -1.26840651, ...,  0.        ,
             1.        ,  0.        ],
           [-0.24320179, -0.84242234, -1.2979348 , ...,  0.        ,
             1.        ,  0.        ],
           ...,
           [-0.2605051 , -0.70053739, -1.29550306, ...,  0.        ,
             1.        ,  0.        ],
           [ 0.90957735, -1.71950283, -0.6410172 , ...,  0.        ,
             1.        ,  0.        ],
           [-0.09496148,  1.48887805,  0.71172586, ...,  0.        ,
             1.        ,  0.        ]])



    - Expérimentations ML pour la sélection de modèle


```python
# Régression linéaire
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)
```




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" checked><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>


Ce code accomplit plusieurs tâches en utilisant un modèle de régression linéaire pour effectuer une prédiction :

1. **Transformation des données d'entraînement** :
   - `X_train_prepared = preprocessor.fit_transform(X_train)`: Les données d'entraînement `X_train` sont prétraitées en utilisant le préprocesseur `preprocessor`. Cela implique l'application des transformations spécifiques définies pour les colonnes numériques et catégoriques. Les colonnes numériques sont transformées à l'aide du pipeline `num_pipeline`, tandis que les colonnes catégoriques sont transformées à l'aide du pipeline `cat_pipeline`. Cette opération prépare les données pour être utilisées dans le modèle de régression linéaire.

2. **Entraînement du modèle de régression linéaire** :
   - `from sklearn.linear_model import LinearRegression`: Importe la classe `LinearRegression` de scikit-learn pour créer un modèle de régression linéaire.
   - `lin_reg = LinearRegression()`: Initialise un objet de modèle de régression linéaire.
   - `lin_reg.fit(X_train_prepared, y_train)`: Entraîne le modèle de régression linéaire en utilisant les données prétraitées `X_train_prepared` en tant que variables explicatives et les étiquettes `y_train` correspondantes comme valeurs cibles. Le modèle ajuste les coefficients de régression pour minimiser l'erreur quadratique moyenne entre les prédictions et les valeurs réelles.

En résumé, ce code prépare les données d'entraînement en les prétraitant à l'aide d'un `preprocessor` qui applique des transformations spécifiques aux colonnes numériques et catégoriques, puis entraîne un modèle de régression linéaire à l'aide de ces données prétraitées.



```python
# Prédictions
y_train_preds = lin_reg.predict(X_train_prepared)

# RMSE
from sklearn.metrics import root_mean_squared_error
lin_rmse = root_mean_squared_error(y_train, y_train_preds)
lin_rmse
```




    268519.6895255891


Ce code effectue une prédiction sur les données d'entraînement en utilisant le modèle de régression linéaire entraîné précédemment, puis calcule la racine de l'erreur quadratique moyenne (RMSE) entre les prédictions et les valeurs cibles réelles. Voici une explication ligne par ligne :

1. **Prédiction sur les données d'entraînement** :
   - `y_train_preds = lin_reg.predict(X_train_prepared)`: Le modèle de régression linéaire (`lin_reg`) est utilisé pour prédire les valeurs cibles (`y_train_preds`) en utilisant les données d'entraînement prétraitées (`X_train_prepared`). Cela donne les prédictions du modèle sur les données d'entraînement.

2. **Importation de la métrique de performance** :
   - `from sklearn.metrics import root_mean_squared_error`: Importe la fonction `root_mean_squared_error` de scikit-learn, qui calcule la racine de l'erreur quadratique moyenne (RMSE). Cette métrique est utilisée pour évaluer les performances du modèle de régression.

3. **Calcul de la RMSE** :
   - `lin_rmse = root_mean_squared_error(y_train, y_train_preds)`: Calcule la RMSE entre les valeurs cibles réelles (`y_train`) et les prédictions du modèle (`y_train_preds`). La RMSE mesure l'écart moyen entre les prédictions du modèle et les valeurs réelles, en tenant compte de la variance des erreurs. Cette valeur est stockée dans la variable `lin_rmse`.

En résumé, ce code évalue les performances du modèle de régression linéaire sur les données d'entraînement en calculant la RMSE entre les prédictions du modèle et les valeurs cibles réelles.


```python
# Validation croisée
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, X_train_prepared, y_train, 
                         scoring="neg_mean_squared_error", cv=3)
lin_rmse_scores = np.sqrt(-scores)
lin_rmse_scores
```




    array([272673.19731209, 272029.12417556, 269864.04921785])




```python
def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())
```


```python
display_scores(lin_rmse_scores)
```

    Scores: [272673.19731209 272029.12417556 269864.04921785]
    Mean: 271522.1235685009
    Standard deviation: 1201.5588390367857


Ce code utilise la validation croisée pour évaluer les performances du modèle de régression linéaire sur les données d'entraînement. Voici une explication ligne par ligne :

1. **Importation du module nécessaire** :
   - `from sklearn.model_selection import cross_val_score`: Importe la fonction `cross_val_score` de scikit-learn, qui permet de réaliser la validation croisée.

2. **Évaluation du modèle par validation croisée** :
   - `scores = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=3)`: Utilise la fonction `cross_val_score` pour évaluer le modèle de régression linéaire (`lin_reg`) en utilisant une validation croisée à 3 plis (`cv=3`). Les données d'entraînement prétraitées (`X_train_prepared`) et les valeurs cibles (`y_train`) sont utilisées. La métrique de performance utilisée est l'opposé de l'erreur quadratique moyenne (`scoring="neg_mean_squared_error"`), car `cross_val_score` maximise les valeurs de score, mais dans ce cas, nous voulons minimiser l'erreur quadratique moyenne.

3. **Conversion des scores en RMSE** :
   - `lin_rmse_scores = np.sqrt(-scores)`: Prend l'opposé des scores de l'erreur quadratique moyenne (qui ont été négativés pour des raisons de convention dans scikit-learn) et calcule la racine carrée pour obtenir les scores de la racine de l'erreur quadratique moyenne (RMSE). Ces valeurs sont stockées dans `lin_rmse_scores`.

4. **Définition d'une fonction pour afficher les scores** :
   - `def display_scores(scores)`: Définit une fonction `display_scores` qui prend en entrée les scores et affiche les scores, la moyenne des scores et l'écart type des scores.
   - `print("Scores:", scores)`: Affiche les scores de la validation croisée.
   - `print("Mean:", scores.mean())`: Affiche la moyenne des scores.
   - `print("Standard deviation:", scores.std())`: Affiche l'écart type des scores.

5. **Affichage des scores de validation croisée et de leurs statistiques** :
   - `display_scores(lin_rmse_scores)`: Appelle la fonction `display_scores` pour afficher les scores de la RMSE et leurs statistiques.

En résumé, ce code évalue les performances du modèle de régression linéaire sur les données d'entraînement en utilisant la validation croisée et affiche les scores de la RMSE ainsi que leurs statistiques.


En machine learning, la validation croisée est une technique essentielle pour évaluer la performance des modèles prédictifs et estimer leur capacité à généraliser à de nouvelles données non vues. Elle consiste à diviser l'ensemble de données en sous-ensembles plus petits appelés "plis", puis à entraîner et évaluer le modèle plusieurs fois en utilisant différents plis comme ensemble de validation, tandis que les autres plis sont utilisés comme ensemble d'entraînement.

La validation croisée est importante pour plusieurs raisons :

1. **Estimation fiable de la performance du modèle** : La validation croisée fournit une estimation plus fiable de la performance du modèle que la simple division de l'ensemble de données en ensembles d'entraînement et de test. En moyennant les performances sur plusieurs itérations de validation, elle réduit le risque de biais lié à la sélection aléatoire des ensembles d'entraînement et de test.

2. **Utilisation efficace des données** : En utilisant chaque observation à la fois comme données d'entraînement et de test à différentes itérations, la validation croisée permet de maximiser l'utilisation des données disponibles pour l'entraînement et l'évaluation du modèle.

3. **Évaluation de la capacité de généralisation** : La validation croisée fournit une estimation de la capacité du modèle à généraliser à de nouvelles données non vues. En testant le modèle sur des ensembles de données différents de ceux utilisés pour l'entraînement, elle évalue sa capacité à faire des prédictions précises sur des données inconnues.

4. **Détection du surajustement (overfitting)** : La validation croisée permet de détecter le surajustement en fournissant une estimation plus fiable des performances du modèle sur des données réelles. Si le modèle présente des performances élevées sur les ensembles d'entraînement mais des performances médiocres sur les ensembles de validation, cela peut indiquer un surajustement.

En résumé, la validation croisée est une technique essentielle en machine learning pour évaluer la performance des modèles, estimer leur capacité à généraliser à de nouvelles données et détecter le surajustement. Elle fournit une évaluation plus fiable et robuste des modèles prédictifs, ce qui est crucial pour la prise de décision dans de nombreuses applications.



```python
def train_and_evaluate(ml_algo):
  reg = ml_algo
  reg.fit(X_train_prepared, y_train)

  y_train_predictions = reg.predict(X_train_prepared)
  reg_rmse = root_mean_squared_error(y_train, y_train_predictions)
  print("RMSE:", reg_rmse)

  reg_scores = cross_val_score(reg, X_train_prepared, y_train, 
                               scoring="neg_mean_squared_error", cv=3)
  reg_rmse_scores = np.sqrt(-reg_scores)
  display_scores(reg_rmse_scores)
  return reg
```


```python
all_models = []
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.neural_network import MLPRegressor
for ml_algo in [RandomForestRegressor(random_state=seed), 
                XGBRegressor(random_state=seed)]:
  print(ml_algo)
  model = train_and_evaluate(ml_algo)
  all_models.append(model)
  print('\n')
```

    RandomForestRegressor(random_state=123)
    RMSE: 56579.313618180415
    Scores: [158875.65614356 157124.34426933 158029.27379189]
    Mean: 158009.7580682598
    Standard deviation: 715.1032409389485
    
    
    XGBRegressor(base_score=None, booster=None, callbacks=None,
                 colsample_bylevel=None, colsample_bynode=None,
                 colsample_bytree=None, device=None, early_stopping_rounds=None,
                 enable_categorical=False, eval_metric=None, feature_types=None,
                 gamma=None, grow_policy=None, importance_type=None,
                 interaction_constraints=None, learning_rate=None, max_bin=None,
                 max_cat_threshold=None, max_cat_to_onehot=None,
                 max_delta_step=None, max_depth=None, max_leaves=None,
                 min_child_weight=None, missing=nan, monotone_constraints=None,
                 multi_strategy=None, n_estimators=None, n_jobs=None,
                 num_parallel_tree=None, random_state=123, ...)
    RMSE: 102855.8063586502
    Scores: [155600.20801531 153316.95177794 153926.69687876]
    Mean: 154281.28555733606
    Standard deviation: 965.268394318689
    
    
Ce code définit une fonction `train_and_evaluate` qui prend un algorithme d'apprentissage automatique en entrée, entraîne ce modèle sur les données d'entraînement, évalue sa performance et retourne le modèle entraîné. Ensuite, il boucle sur une liste d'algorithmes d'apprentissage automatique (Random Forest et XGBoost) et utilise cette fonction pour entraîner et évaluer chaque modèle.

Explications détaillées du code :

1. **Définition de la fonction `train_and_evaluate`** :
   - Cette fonction prend un algorithme d'apprentissage automatique (`ml_algo`) en entrée.
   - Elle entraîne ce modèle sur les données d'entraînement préparées (`X_train_prepared`, `y_train`).
   - Elle effectue des prédictions sur les données d'entraînement pour calculer la racine de l'erreur quadratique moyenne (RMSE) entre les valeurs cibles réelles et les prédictions du modèle.
   - Elle utilise la validation croisée à 3 plis pour évaluer la performance du modèle en utilisant l'opposé de l'erreur quadratique moyenne comme métrique de performance.
   - Elle affiche la RMSE sur les données d'entraînement et les scores de RMSE moyens et écart-type obtenus par la validation croisée.
   - Enfin, elle retourne le modèle entraîné.

2. **Initialisation de la liste `all_models`** :
   - `all_models = []` : Initialise une liste vide `all_models` qui stockera tous les modèles entraînés.

3. **Importation des modules nécessaires** :
   - `from sklearn.ensemble import RandomForestRegressor` : Importe la classe `RandomForestRegressor` de scikit-learn pour entraîner un modèle de forêt aléatoire.
   - `from xgboost.sklearn import XGBRegressor` : Importe la classe `XGBRegressor` de la bibliothèque XGBoost pour entraîner un modèle de boosting de gradient.
   - `from sklearn.neural_network import MLPRegressor` : Importe la classe `MLPRegressor` de scikit-learn pour entraîner un modèle de réseau de neurones.

4. **Boucle sur les algorithmes d'apprentissage automatique** :
   - La boucle itère sur une liste contenant des instances d'algorithmes d'apprentissage automatique : `RandomForestRegressor` et `XGBRegressor`.
   - Pour chaque algorithme, il affiche l'algorithme utilisé, puis appelle la fonction `train_and_evaluate` pour entraîner et évaluer le modèle.
   - Le modèle entraîné est ensuite ajouté à la liste `all_models`.

En résumé, ce code entraîne et évalue plusieurs modèles d'apprentissage automatique (Random Forest et XGBoost) en utilisant une fonction commune pour entraîner et évaluer chaque modèle. Il fournit également des informations sur la performance de chaque modèle sur les données d'entraînement et via la validation croisée.


- Réglage des hyperparamètres


```python
num_pipeline = Pipeline([
    ('feats_adder', FunctionTransformer(add_features)),
    ('num_impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('cat_imputer', SimpleImputer(strategy='constant', fill_value="UNKNOWN")),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=0.002))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, vars_num),
    ("cat", cat_pipeline, vars_cat)
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ('model', XGBRegressor(random_state=seed))
])

full_pipeline
```


```python
full_pipeline[1]
```

Ce code définit plusieurs pipelines pour prétraiter les caractéristiques numériques et catégoriques, puis les fusionne dans un pipeline complet qui prétraite les données et entraîne un modèle de régression XGBoost. Voici une explication ligne par ligne :

1. **Pipeline pour les caractéristiques numériques (`num_pipeline`)** :
   - Ce pipeline contient trois étapes :
     - `('feats_adder', FunctionTransformer(add_features))`: Utilise la fonction `add_features` pour ajouter de nouvelles variables aux caractéristiques numériques.
     - `('num_impute', SimpleImputer(strategy='median'))`: Impute les valeurs manquantes en utilisant la médiane des valeurs existantes.
     - `('scaler', StandardScaler())`: Standardise les caractéristiques en centrant les données autour de zéro et en les mettant à l'échelle pour avoir une variance unitaire.

2. **Pipeline pour les caractéristiques catégoriques (`cat_pipeline`)** :
   - Ce pipeline contient deux étapes :
     - `('cat_imputer', SimpleImputer(strategy='constant', fill_value="UNKNOWN"))`: Impute les valeurs manquantes en remplaçant par "UNKNOWN".
     - `('encoder', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=0.002))`: Encode les caractéristiques catégoriques en utilisant un encodage one-hot, avec des options pour gérer les catégories inconnues et minimiser la fréquence des catégories rares.

3. **ColumnTransformer (`preprocessor`)** :
   - Utilise le `num_pipeline` pour prétraiter les caractéristiques numériques (`vars_num`) et le `cat_pipeline` pour prétraiter les caractéristiques catégoriques (`vars_cat`).

4. **Pipeline complet (`full_pipeline`)** :
   - Ce pipeline contient deux étapes :
     - `("preprocess", preprocessor)`: Utilise le `preprocessor` pour prétraiter les caractéristiques numériques et catégoriques.
     - `('model', XGBRegressor(random_state=seed))`: Utilise un modèle de régression XGBoost comme modèle final pour l'apprentissage.
  
5. **Affichage du pipeline complet (`full_pipeline`)** :
   - `full_pipeline`: Affiche le pipeline complet, montrant toutes les étapes de prétraitement et de modélisation.

6. **Accès au modèle entraîné dans le pipeline complet** :
   - `full_pipeline[1]`: Permet d'accéder au deuxième élément du pipeline complet, qui est le modèle de régression XGBoost. Cela peut être utile pour accéder au modèle entraîné et utiliser ses méthodes ou attributs après l'entraînement.

En résumé, ce code définit un pipeline complet qui prétraite les caractéristiques numériques et catégoriques, puis entraîne un modèle de régression XGBoost. Ce pipeline peut être utilisé pour entraîner et évaluer le modèle de manière cohérente et répétable.



```python
X_train = train_set.drop('price', axis=1)
y_train = train_set['price']
```


```python
from scipy.stats import randint

param_dist = {
    'model__n_estimators': randint(low=150, high=200),
    'model__max_depth': randint(low=5, high=10),
    'model__learning_rate': np.arange(0.05, 1, 0.05),
    'model__colsample_bytree': np.arange(0.5, 1, 0.05)
}

from sklearn.model_selection import RandomizedSearchCV
rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_dist,
    n_iter=15, cv=3, scoring='neg_mean_squared_error',
    random_state=seed, n_jobs=-1
)

rnd_search.fit(X_train, y_train)

print("Meilleurs hyperparamètres :", rnd_search.best_params_)
print("Meilleur score :", np.sqrt(-rnd_search.best_score_))
```

    Meilleurs hyperparamètres : {'model__colsample_bytree': 0.9000000000000004, 'model__learning_rate': 0.2, 'model__max_depth': 7, 'model__n_estimators': 155}
    Meilleur score : 150460.43190867038



Ce code effectue une recherche aléatoire des hyperparamètres pour un modèle XGBoost en utilisant la validation croisée pour évaluer la performance des combinaisons d'hyperparamètres. Voici une explication ligne par ligne :

1. **Préparation des données d'entraînement** :
   - `X_train = train_set.drop('price', axis=1)`: Sépare les caractéristiques des étiquettes en enlevant la colonne 'price' des données d'entraînement.
   - `y_train = train_set['price']`: Extrait la colonne 'price' comme étiquettes des données d'entraînement.

2. **Définition de l'espace des hyperparamètres** (`param_dist`) :
   - `param_dist`: Un dictionnaire définissant les distributions de probabilité des hyperparamètres à explorer dans la recherche aléatoire. Les hyperparamètres spécifiés sont :
     - `'model__n_estimators'`: Le nombre d'estimateurs dans le modèle XGBoost.
     - `'model__max_depth'`: La profondeur maximale de chaque arbre dans le modèle XGBoost.
     - `'model__learning_rate'`: Le taux d'apprentissage du modèle XGBoost.
     - `'model__colsample_bytree'`: La proportion de caractéristiques à considérer pour chaque arbre dans le modèle XGBoost.

3. **Importation de la classe `RandomizedSearchCV`** :
   - `from sklearn.model_selection import RandomizedSearchCV`: Importe la classe `RandomizedSearchCV` de scikit-learn, qui effectue une recherche aléatoire des hyperparamètres avec validation croisée.

4. **Initialisation de l'objet `RandomizedSearchCV`** (`rnd_search`) :
   - `rnd_search = RandomizedSearchCV(...)` : Initialise un objet `RandomizedSearchCV` avec les paramètres suivants :
     - `estimator=full_pipeline`: Le pipeline complet contenant le modèle XGBoost et les prétraitements.
     - `param_distributions=param_dist`: Les distributions d'hyperparamètres à explorer.
     - `n_iter=15`: Le nombre d'itérations de la recherche aléatoire.
     - `cv=3`: Le nombre de plis pour la validation croisée.
     - `scoring='neg_mean_squared_error'`: La métrique de performance à optimiser, dans ce cas, l'opposé de l'erreur quadratique moyenne.
     - `random_state=seed`: La graine aléatoire pour la reproductibilité.
     - `n_jobs=-1`: Utilise tous les cœurs disponibles pour accélérer le processus.

5. **Exécution de la recherche aléatoire** :
   - `rnd_search.fit(X_train, y_train)`: Exécute la recherche aléatoire en ajustant le modèle à l'ensemble de données d'entraînement.

6. **Affichage des meilleurs hyperparamètres et du meilleur score** :
   - `print("Meilleurs hyperparamètres :", rnd_search.best_params_)`: Affiche les meilleurs hyperparamètres trouvés par la recherche aléatoire.
   - `print("Meilleur score :", np.sqrt(-rnd_search.best_score_))`: Affiche le meilleur score de performance, qui est la racine de l'opposé de l'erreur quadratique moyenne, évalué sur l'ensemble de validation croisée.

En résumé, ce code effectue une recherche aléatoire des hyperparamètres pour un modèle XGBoost en utilisant la validation croisée pour évaluer la performance des différentes combinaisons d'hyperparamètres. Il sélectionne les meilleurs hyperparamètres et affiche le meilleur score de performance obtenu.


En machine learning, un hyperparamètre est un paramètre dont la valeur est fixée avant le début du processus d'apprentissage. Contrairement aux paramètres du modèle, qui sont appris à partir des données d'entraînement, les hyperparamètres ne sont pas directement appris par le modèle, mais plutôt spécifiés par le praticien de l'apprentissage automatique avant le processus d'entraînement.

Voici quelques exemples courants d'hyperparamètres dans différents algorithmes d'apprentissage automatique :

1. **Dans les modèles de régression linéaire** :
   - L'hyperparamètre pourrait être le terme de régularisation (par exemple, le coefficient de pénalité dans la régression de Ridge ou Lasso).

2. **Dans les modèles d'arbre de décision** :
   - L'hyperparamètre pourrait être la profondeur maximale de l'arbre, le nombre minimal d'observations dans les feuilles, ou le nombre minimum d'observations nécessaires pour diviser un nœud.

3. **Dans les modèles de réseaux de neurones** :
   - Les hyperparamètres pourraient inclure le nombre de couches, le nombre de neurones dans chaque couche, le taux d'apprentissage, etc.

Le réglage des hyperparamètres, également appelé optimisation des hyperparamètres, fait référence au processus de sélection des valeurs optimales pour les hyperparamètres d'un modèle afin d'optimiser ses performances sur un ensemble de données donné. C'est un aspect crucial de la construction de modèles performants en machine learning.

Le réglage des hyperparamètres peut se faire de manière exhaustive en essayant différentes combinaisons d'hyperparamètres à l'aide d'une validation croisée pour évaluer la performance de chaque combinaison. Il peut également se faire de manière plus efficace à l'aide de techniques d'optimisation automatique telles que la recherche aléatoire, la recherche par grille, ou des algorithmes d'optimisation plus sophistiqués comme la recherche bayésienne.

En résumé, les hyperparamètres sont des paramètres définis avant l'entraînement du modèle, et le réglage des hyperparamètres consiste à sélectionner les valeurs optimales de ces paramètres pour maximiser les performances du modèle sur les données d'entraînement et de test.



```python
pd.DataFrame(rnd_search.cv_results_)
```


```python
final_model = rnd_search.best_estimator_
final_model
```



Ce code utilise les résultats de la recherche aléatoire des hyperparamètres (`rnd_search`) pour créer un DataFrame Pandas contenant ces résultats, puis sélectionne le meilleur modèle trouvé par la recherche aléatoire.

1. **Création d'un DataFrame à partir des résultats de la recherche aléatoire** :
   - `pd.DataFrame(rnd_search.cv_results_)`: Utilise la fonction `pd.DataFrame()` pour créer un DataFrame Pandas à partir des résultats de la recherche aléatoire (`rnd_search.cv_results_`). Ces résultats contiennent des informations sur les différents paramètres testés, les scores de validation croisée et d'autres métriques associées.

2. **Sélection du meilleur modèle** :
   - `final_model = rnd_search.best_estimator_`: Sélectionne le meilleur modèle trouvé par la recherche aléatoire en accédant à l'attribut `best_estimator_` de l'objet `rnd_search`. Cet attribut contient l'estimateur final (c'est-à-dire le modèle) qui a obtenu le meilleur score de performance lors de la recherche aléatoire.

En résumé, ce code extrait les résultats de la recherche aléatoire des hyperparamètres dans un DataFrame Pandas pour une analyse plus approfondie, puis sélectionne le meilleur modèle trouvé par la recherche aléatoire pour une utilisation ultérieure. Cela permet d'accéder facilement aux performances et aux détails des différents modèles testés lors de la recherche des hyperparamètres, ainsi qu'au modèle optimal pour la prédiction.


```python
# Données de test
X_test = test_set.drop("price", axis=1)
y_test = test_set['price']
```


```python
# Prédictions sur les données de test
final_preds = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_preds)
final_rmse
```




    145401.679231076




```python
# Importance des attributs
feature_importances = final_model[1].feature_importances_
feature_importances
```




    array([0.01415569, 0.01989408, 0.01661087, 0.0342943 , 0.00701508,
           0.0168217 , 0.01055061, 0.00716285, 0.1057459 , 0.09275771,
           0.0027499 , 0.00859006, 0.00503853, 0.00901752, 0.00492183,
           0.02368234, 0.00634518, 0.00993551, 0.02150142, 0.00709649,
           0.        , 0.01088047, 0.0185196 , 0.00741489, 0.00682468,
           0.00577252, 0.11137699, 0.05187562, 0.03596661, 0.00904498,
           0.0088913 , 0.01877839, 0.01491495, 0.03699737, 0.01593864,
           0.00605421, 0.02725793, 0.01397329, 0.019726  , 0.00120924,
           0.00424008, 0.00791157, 0.01056543, 0.00286114, 0.00053156,
           0.00232237, 0.01415903, 0.00394749, 0.00546943, 0.00478915,
           0.006143  , 0.00392316, 0.01119221, 0.        , 0.01214399,
           0.00237037, 0.00952335, 0.00128693, 0.00520556, 0.00428585,
           0.00442623, 0.00599695, 0.0123602 , 0.00719747, 0.0065831 ,
           0.00525913, 0.        ], dtype=float32)




```python
len(feature_importances)
```




    67




```python
extra_attribs = [
    'living_area_per_total_land',
    'total_number_of_rooms',
    'bedrooms_per_room',
    'total_parking_capacity',
    'num_ameneties',
    'distance_to_paris'
]

cat_ohe_attribs = list(final_model[0].named_transformers_['cat'].get_feature_names_out())

attributes = vars_num + extra_attribs + cat_ohe_attribs
print(len(attributes))
attributes
```

    67





    ['approximate_latitude',
     'approximate_longitude',
     'postal_code',
     'size',
     'floor',
     'land_size',
     'energy_performance_value',
     'ghg_value',
     'nb_rooms',
     'nb_bedrooms',
     'nb_bathrooms',
     'nb_parking_places',
     'nb_boxes',
     'nb_photos',
     'has_a_balcony',
     'nb_terraces',
     'has_a_cellar',
     'has_a_garage',
     'has_air_conditioning',
     'last_floor',
     'upper_floors',
     'living_area_per_total_land',
     'total_number_of_rooms',
     'bedrooms_per_room',
     'total_parking_capacity',
     'num_ameneties',
     'distance_to_paris',
     'property_type_appartement',
     'property_type_divers',
     'property_type_duplex',
     'property_type_ferme',
     'property_type_maison',
     'property_type_parking',
     'property_type_propriété',
     'property_type_terrain',
     'property_type_terrain à bâtir',
     'property_type_viager',
     'property_type_villa',
     'property_type_infrequent_sklearn',
     'energy_performance_category_A',
     'energy_performance_category_B',
     'energy_performance_category_C',
     'energy_performance_category_D',
     'energy_performance_category_E',
     'energy_performance_category_F',
     'energy_performance_category_G',
     'energy_performance_category_UNKNOWN',
     'ghg_category_A',
     'ghg_category_B',
     'ghg_category_C',
     'ghg_category_D',
     'ghg_category_E',
     'ghg_category_F',
     'ghg_category_G',
     'ghg_category_UNKNOWN',
     'exposition_Est',
     'exposition_Est-Ouest',
     'exposition_Nord',
     'exposition_Nord-Est',
     'exposition_Nord-Ouest',
     'exposition_Ouest',
     'exposition_Sud',
     'exposition_Sud-Est',
     'exposition_Sud-Nord',
     'exposition_Sud-Ouest',
     'exposition_UNKNOWN',
     'exposition_infrequent_sklearn']




```python
importance_result = sorted(zip(feature_importances, attributes), reverse=True)
imp_df = pd.DataFrame(importance_result, 
                      columns=["Score d'importance", "Variable"])
imp_df
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
      <th>Score d'importance</th>
      <th>Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.111377</td>
      <td>distance_to_paris</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.105746</td>
      <td>nb_rooms</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.092758</td>
      <td>nb_bedrooms</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.051876</td>
      <td>property_type_appartement</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.036997</td>
      <td>property_type_propriété</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.001209</td>
      <td>energy_performance_category_A</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.000532</td>
      <td>energy_performance_category_F</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.000000</td>
      <td>upper_floors</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0.000000</td>
      <td>ghg_category_G</td>
    </tr>
    <tr>
      <th>66</th>
      <td>0.000000</td>
      <td>exposition_infrequent_sklearn</td>
    </tr>
  </tbody>
</table>
<p>67 rows × 2 columns</p>
</div>


Ce code effectue plusieurs étapes pour évaluer les performances du modèle final sur l'ensemble de test et pour analyser l'importance des variables dans les prédictions du modèle. Voici une explication ligne par ligne :

1. **Préparation des données de test** :
   - `X_test = test_set.drop("price", axis=1)`: Sépare les caractéristiques des étiquettes en enlevant la colonne 'price' des données de test.
   - `y_test = test_set['price']`: Extrait la colonne 'price' comme étiquettes des données de test.

2. **Prédiction sur les données de test** :
   - `final_preds = final_model.predict(X_test)`: Utilise le modèle final (`final_model`) pour faire des prédictions sur les données de test.

3. **Calcul de la RMSE sur les données de test** :
   - `final_rmse = root_mean_squared_error(y_test, final_preds)`: Calcule la racine de l'erreur quadratique moyenne (RMSE) entre les valeurs cibles réelles et les prédictions du modèle sur les données de test.

4. **Extraction de l'importance des variables** :
   - `feature_importances = final_model[1].feature_importances_`: Extrait l'importance des variables à partir du modèle final. Pour le modèle XGBoost, cette information est généralement disponible dans l'attribut `feature_importances_`.

5. **Création de la liste des variables supplémentaires et des variables catégorielles encodées en one-hot** :
   - `extra_attribs`: Liste des variables supplémentaires ajoutées lors du prétraitement.
   - `cat_ohe_attribs = list(final_model[0].named_transformers_['cat'].get_feature_names_out())`: Obtient les noms des caractéristiques encodées en one-hot à partir du préprocesseur. Il utilise le préfixe `'cat'` pour identifier les caractéristiques catégorielles dans le préprocesseur.

6. **Combinaison de toutes les caractéristiques** :
   - `attributes = vars_num + extra_attribs + cat_ohe_attribs`: Combine les caractéristiques numériques originales (`vars_num`), les variables supplémentaires et les variables catégorielles encodées en one-hot pour former une liste complète de toutes les caractéristiques.

7. **Analyse de l'importance des variables** :
   - `importance_result = sorted(zip(feature_importances, attributes), reverse=True)`: Combine les importances des variables avec leurs noms et les trie par ordre décroissant d'importance.
   - `imp_df = pd.DataFrame(importance_result, columns=["Score d'importance", "Variable"])`: Crée un DataFrame Pandas à partir des résultats d'importance des variables.

En résumé, ce code évalue les performances du modèle final sur l'ensemble de test en calculant la RMSE, puis analyse l'importance des variables dans les prédictions du modèle en extrayant les coefficients d'importance des variables et en les combinant avec les noms des variables. Il fournit ainsi une analyse détaillée des facteurs qui influent le plus sur les prédictions du modèle.


```python
plt.figure(figsize=(10,16))
sns.barplot(data=imp_df, 
            x="Score d'importance", 
            y="Variable");
```


    
![png](french_real_estate_prediction_files/french_real_estate_prediction_62_0.png)
    



```python
# Enregistrement
import joblib
joblib.dump(final_model, "final_model.pkl")
```




    ['final_model.pkl']


Ce code réalise deux actions :

1. **Création d'un graphique à barres pour visualiser l'importance des variables** :
   - `plt.figure(figsize=(10,16))`: Crée une nouvelle figure de matplotlib avec une taille spécifiée (10 pouces de largeur par 16 pouces de hauteur).
   - `sns.barplot(data=imp_df, x="Score d'importance", y="Variable")`: Crée un graphique à barres en utilisant les données du DataFrame `imp_df`, où les variables sont affichées sur l'axe des ordonnées (`y`) et les scores d'importance sont affichés sur l'axe des abscisses (`x`). Cela permet de visualiser l'importance relative des différentes variables dans les prédictions du modèle.

2. **Sauvegarde du modèle final entraîné** :
   - `joblib.dump(final_model, "final_model.pkl")`: Utilise la fonction `dump` du module `joblib` pour sauvegarder le modèle final entraîné (`final_model`) dans un fichier binaire spécifié (`final_model.pkl`). Cette opération permet de conserver le modèle pour une utilisation future, par exemple pour effectuer des prédictions sur de nouvelles données sans avoir à ré-entraîner le modèle à chaque fois.

En résumé, ce code génère un graphique à barres pour visualiser l'importance des variables dans les prédictions du modèle, puis sauvegarde le modèle final entraîné dans un fichier pour une utilisation ultérieure. Cela permet d'analyser les résultats du modèle et de le déployer facilement dans des applications ou des environnements de production.


🖥 **Conclusion**

Après avoir construit et sauvegardé le modèle de prédiction d'un bien immobilier, plusieurs étapes peuvent suivre pour tirer le meilleur parti du modèle et l'intégrer dans des applications réelles :

1. **Évaluation continue du modèle** :
   - Il est important de suivre et d'évaluer régulièrement les performances du modèle pour s'assurer qu'il continue de fournir des prédictions précises. Cela peut impliquer la surveillance des métriques de performance sur des données de validation ou de test, ainsi que la mise à jour périodique du modèle si nécessaire.

2. **Déploiement dans un environnement de production** :
   - Une fois que le modèle est prêt, il peut être déployé dans un environnement de production où il peut être utilisé pour faire des prédictions sur de nouvelles données en temps réel. Cela peut se faire en créant une API web, en intégrant le modèle dans une application ou un service existant, ou en le déployant sur un serveur.

3. **Intégration dans des systèmes décisionnels** :
   - Le modèle peut être intégré dans des systèmes décisionnels ou des outils d'analyse pour aider à la prise de décision. Par exemple, il peut être utilisé pour estimer la valeur d'un bien immobilier dans le cadre d'une évaluation immobilière ou pour prendre des décisions d'investissement immobilier.

4. **Formation et support utilisateur** :
   - Il peut être nécessaire de former les utilisateurs finaux sur la façon d'utiliser le modèle et de fournir un support technique continu pour répondre à leurs questions et résoudre les problèmes éventuels.

5. **Gestion des mises à jour et de la maintenance** :
   - Comme les données et les besoins commerciaux évoluent, il peut être nécessaire de mettre à jour et de maintenir le modèle en continu. Cela peut impliquer l'ajout de nouvelles fonctionnalités, l'optimisation des hyperparamètres, ou la réévaluation de l'ensemble de données utilisé pour l'entraînement.

6. **Sécurité et confidentialité des données** :
   - Assurer la sécurité et la confidentialité des données est essentiel lors du déploiement d'un modèle dans un environnement de production. Cela peut impliquer la mise en place de mesures de sécurité telles que le chiffrement des données, le contrôle d'accès et la gestion des identités, ainsi que le respect des réglementations telles que le RGPD.

En résumé, une fois que le modèle de prédiction d'un bien immobilier est construit et sauvegardé, la suite des étapes implique principalement son déploiement, son évaluation continue, son intégration dans des systèmes existants, la formation des utilisateurs finaux, la gestion des mises à jour et la sécurité des données.

**Dans cette application, vous trouverez plusieurs projets d'intégration de Modèles Machine Learning dans des applications web. Amusez-vous bien :)**