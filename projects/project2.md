# Projet : Système de recommandations de films

## Description

L'objectif de ce projet est de construire une application de système de recommandations de films. L'utilisateur de cette application renseigne les informations relatives à un film donné et l'application lui retourne les n films les plus similaires à ce film donné.

## Image
imgs/project2/project2.png

## Instructions

1. **Chargement des données** : Téléchargez les jeux de données nécessaires [movies.csv](https://drive.google.com/file/d/1zhy3IQXm_dthKTjG4hDQFBbmsfNITxwL/view?usp=sharing) et chargez-les dans votre environnement de travail. La source des données est : [MovieLens](https://grouplens.org/datasets/movielens/)

2. **Préparation des données** : Nettoyez les données, gérez les valeurs manquantes, et effectuez toute transformation nécessaire y compris la phase de Feature Engineering

3. **Exploration des données** : Réalisez une analyse exploratoire des données pour comprendre les différentes variables. V

4. **Calculer les similarités Cosinus pour chaue paire de lignes et Sauvegarde de la matrice de similarité** : La similarité cosinus est une mesure de similarité entre deux vecteurs dans un espace vectoriel, souvent utilisée pour mesurer la similarité entre des documents dans le contexte de la vectorisation de texte.

5. **Construction dúne fonction ayant comme arguments les informations relatifs à un film et qui retourne les films les plus similaires** 

6. **Tester cette fonction sur un certain nombre de films et analyser les résultats** : Évaluez les performances du modèle à l'aide de métriques telles que le Root Mean Squared Error (RMSE) et Mean Absolute Error (MAE).

7. **Déploiement de l'application** : Utilisez Streamlit pour créer une application web interactive permettant de recommander des films en fonction des préférences de l'utilisateur. L'utilisateur pourra entrer ses préférences et obtenir une liste de films recommandés.

8. **Documentation et partage** : Documentez le processus de développement et partagez votre application avec les autres. Assurez-vous que votre application est bien présentée et facile à utiliser.


## Resources
- [Jeu de données sur les Films](https://drive.google.com/file/d/1zhy3IQXm_dthKTjG4hDQFBbmsfNITxwL/view?usp=sharing)
- [Source des données](https://grouplens.org/datasets/movielens/)
- [Comment construire une application de recommandation de films comme Netflix ?](https://youtu.be/bjis_LebM1w)
- [Comment déployer une web app Streamlit](https://youtu.be/wjRlWuXmlvw)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)


## Execution du Projet

Un système de recommandation en intelligence artificielle (IA) est une application logicielle qui utilise des algorithmes pour analyser les préférences, le comportement et les habitudes d'un utilisateur afin de recommander des éléments pertinents. Ces éléments peuvent inclure des produits, des services, des contenus numériques, des connexions sociales, et bien d'autres choses en fonction du contexte d'utilisation.

Les applications des systèmes de recommandation sont variées et s'étendent à divers domaines, notamment :

1. **Commerce électronique :** Les systèmes de recommandation sont largement utilisés par les plateformes de commerce électronique pour recommander des produits aux clients en fonction de leurs historiques d'achats, de leurs recherches précédentes et des tendances générales.

2. **Streaming de contenu :** Des plateformes de streaming vidéo, de musique et de lecture audio utilisent des systèmes de recommandation pour suggérer des films, des séries, des chansons ou des podcasts en fonction des préférences de l'utilisateur.

3. **Réseaux sociaux :** Les médias sociaux utilisent des systèmes de recommandation pour suggérer des amis, des groupes ou des contenus qui pourraient intéresser un utilisateur en fonction de ses interactions passées.

4. **Publicité en ligne :** Les annonces ciblées peuvent être optimisées en utilisant des systèmes de recommandation pour présenter des publicités qui correspondent aux intérêts et aux comportements en ligne d'un utilisateur.

5. **Services de streaming de jeux :** Les plateformes de jeux vidéo peuvent recommander de nouveaux jeux ou du contenu en fonction des préférences de jeu passées d'un utilisateur.

6. **Applications de voyage :** Les systèmes de recommandation peuvent être utilisés pour suggérer des destinations, des hébergements, des activités, etc., en fonction des préférences de voyage passées.

Dans le contexte des entreprises, les systèmes de recommandation offrent plusieurs avantages :

- **Personnalisation :** Les entreprises peuvent personnaliser l'expérience utilisateur en recommandant des produits ou services adaptés aux besoins spécifiques de chaque client.

- **Amélioration de la rétention client :** En recommandant des éléments pertinents, les entreprises peuvent accroître l'engagement et la satisfaction des clients, favorisant ainsi la fidélité.

- **Optimisation des ventes :** Les systèmes de recommandation peuvent contribuer à stimuler les ventes en présentant des offres attrayantes et en simplifiant le processus de décision d'achat.

- **Analyse des données :** Les données collectées par les systèmes de recommandation peuvent être utilisées pour comprendre les tendances du marché, les comportements des clients et améliorer les stratégies commerciales.

En résumé, les systèmes de recommandation en IA sont des outils puissants pour personnaliser l'expérience utilisateur, améliorer la satisfaction client et stimuler les activités commerciales dans divers secteurs.

🖥 **Exploration des données dans un notebook (movie_recommendation_system.ipynb) et Création de la matrice de similarités Cosinus**

- ***Méthode de Cosine Similarity***

La technique de la "cosine similarity" (similarité cosinus) est une méthode couramment utilisée dans le domaine de la recommandation de films et d'autres systèmes de recommandation. Elle est particulièrement populaire dans le contexte du filtrage collaboratif. La similarité cosinus mesure l'angle entre deux vecteurs dans un espace multidimensionnel, et elle est souvent utilisée pour évaluer la similarité entre des utilisateurs ou des articles.

Voici une explication simple de la "cosine similarity" :

1. Représentation vectorielle : Chaque utilisateur et chaque film sont représentés comme des vecteurs dans un espace multidimensionnel. Chaque dimension correspond à une caractéristique particulière (par exemple, le genre du film, l'année de sortie, etc.).

2. Calcul des similarités : La similarité cosinus mesure l'angle entre deux vecteurs. Plus l'angle est petit, plus les vecteurs sont similaires. 


Formule de calcul : https://en.wikipedia.org/wiki/Cosine_similarity


La similarité cosinus renvoie une valeur comprise entre -1 et 1, où 1 indique une similarité totale, 0 une absence de similarité, et -1 une dissimilarité totale.

3. Application à la recommandation de films : Une fois que les utilisateurs et les films sont représentés comme des vecteurs, la similarité cosinus peut être utilisée pour mesurer la similarité entre les préférences d'un utilisateur et les caractéristiques d'un film. Plus la similarité cosinus entre l'utilisateur et le film est élevée, plus il est probable que l'utilisateur apprécie ce film.

4. Recommandation : Pour recommander des films à un utilisateur donné, le système peut calculer la similarité cosinus entre le vecteur représentant l'utilisateur et les vecteurs représentant les films non visionnés. Les films avec les valeurs de similarité les plus élevées sont alors recommandés à l'utilisateur.

Cette approche simple de la "cosine similarity" offre une méthode intuitive pour évaluer la similarité entre utilisateurs et articles dans un espace multidimensionnel, facilitant ainsi la recommandation de films basée sur les préférences passées des utilisateurs.

- ***Implémentation technique du projet***


```python
import pandas as pd
```


```python
df = pd.read_csv('movies.csv')
df.head()
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
      <th>rating</th>
      <th>genre</th>
      <th>year</th>
      <th>released</th>
      <th>score</th>
      <th>votes</th>
      <th>director</th>
      <th>writer</th>
      <th>star</th>
      <th>country</th>
      <th>budget</th>
      <th>gross</th>
      <th>company</th>
      <th>runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Shining</td>
      <td>R</td>
      <td>Drama</td>
      <td>1980</td>
      <td>June 13, 1980 (United States)</td>
      <td>8.4</td>
      <td>927000.0</td>
      <td>Stanley Kubrick</td>
      <td>Stephen King</td>
      <td>Jack Nicholson</td>
      <td>United Kingdom</td>
      <td>19000000.0</td>
      <td>46998772.0</td>
      <td>Warner Bros.</td>
      <td>146.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Blue Lagoon</td>
      <td>R</td>
      <td>Adventure</td>
      <td>1980</td>
      <td>July 2, 1980 (United States)</td>
      <td>5.8</td>
      <td>65000.0</td>
      <td>Randal Kleiser</td>
      <td>Henry De Vere Stacpoole</td>
      <td>Brooke Shields</td>
      <td>United States</td>
      <td>4500000.0</td>
      <td>58853106.0</td>
      <td>Columbia Pictures</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Star Wars: Episode V - The Empire Strikes Back</td>
      <td>PG</td>
      <td>Action</td>
      <td>1980</td>
      <td>June 20, 1980 (United States)</td>
      <td>8.7</td>
      <td>1200000.0</td>
      <td>Irvin Kershner</td>
      <td>Leigh Brackett</td>
      <td>Mark Hamill</td>
      <td>United States</td>
      <td>18000000.0</td>
      <td>538375067.0</td>
      <td>Lucasfilm</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airplane!</td>
      <td>PG</td>
      <td>Comedy</td>
      <td>1980</td>
      <td>July 2, 1980 (United States)</td>
      <td>7.7</td>
      <td>221000.0</td>
      <td>Jim Abrahams</td>
      <td>Jim Abrahams</td>
      <td>Robert Hays</td>
      <td>United States</td>
      <td>3500000.0</td>
      <td>83453539.0</td>
      <td>Paramount Pictures</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Caddyshack</td>
      <td>R</td>
      <td>Comedy</td>
      <td>1980</td>
      <td>July 25, 1980 (United States)</td>
      <td>7.3</td>
      <td>108000.0</td>
      <td>Harold Ramis</td>
      <td>Brian Doyle-Murray</td>
      <td>Chevy Chase</td>
      <td>United States</td>
      <td>6000000.0</td>
      <td>39846344.0</td>
      <td>Orion Pictures</td>
      <td>98.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7668 entries, 0 to 7667
    Data columns (total 15 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   name      7668 non-null   object 
     1   rating    7591 non-null   object 
     2   genre     7668 non-null   object 
     3   year      7668 non-null   int64  
     4   released  7666 non-null   object 
     5   score     7665 non-null   float64
     6   votes     7665 non-null   float64
     7   director  7668 non-null   object 
     8   writer    7665 non-null   object 
     9   star      7667 non-null   object 
     10  country   7665 non-null   object 
     11  budget    5497 non-null   float64
     12  gross     7479 non-null   float64
     13  company   7651 non-null   object 
     14  runtime   7664 non-null   float64
    dtypes: float64(5), int64(1), object(9)
    memory usage: 898.7+ KB


- ***Nettoyage et Feature Engineering***


```python
df.isna().sum()
```




    name           0
    rating        77
    genre          0
    year           0
    released       2
    score          3
    votes          3
    director       0
    writer         3
    star           1
    country        3
    budget      2171
    gross        189
    company       17
    runtime        4
    dtype: int64



Supprimons simplement les lignes avec des valeurs manquantes :


```python
df.dropna(inplace=True)
```


```python
df.isna().sum()
```




    name        0
    rating      0
    genre       0
    year        0
    released    0
    score       0
    votes       0
    director    0
    writer      0
    star        0
    country     0
    budget      0
    gross       0
    company     0
    runtime     0
    dtype: int64




```python
df.shape
```




    (5421, 15)



Feature engineering de la colonne released :


```python
# Créer les nouvelles colonnes avec une expression régulière
df[['date_of_release', 'country_of_release']] = df['released'].str.extract(r'(\w+ \d+, \d+) \(([^)]+)\)')

# Afficher le résultat
df[['released', 'date_of_release', 'country_of_release']].head()

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
      <th>released</th>
      <th>date_of_release</th>
      <th>country_of_release</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>June 13, 1980 (United States)</td>
      <td>June 13, 1980</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>July 2, 1980 (United States)</td>
      <td>July 2, 1980</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>June 20, 1980 (United States)</td>
      <td>June 20, 1980</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>July 2, 1980 (United States)</td>
      <td>July 2, 1980</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>July 25, 1980 (United States)</td>
      <td>July 25, 1980</td>
      <td>United States</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['released', 'date_of_release', 'country_of_release']].dtypes
```




    released              object
    date_of_release       object
    country_of_release    object
    dtype: object




```python
# Supprimer les colonnes 'released' et 'date_of_release'
df.drop(['released', 'date_of_release'], axis=1, inplace=True)
```


```python
df.isna().sum()
```




    name                   0
    rating                 0
    genre                  0
    year                   0
    score                  0
    votes                  0
    director               0
    writer                 0
    star                   0
    country                0
    budget                 0
    gross                  0
    company                0
    runtime                0
    country_of_release    14
    dtype: int64




```python
df.dropna(inplace=True)
```


```python
df.shape
```




    (5407, 15)



Vous pouvez utiliser toutes les colonnes de la dataframe pour créer une application de système web de recommandation de films. Mais gardez à l'esprit que les utilisateurs de vette application n'auront pas forcément certaines informations comme le budget du film, le score, les votes, etc.

Dans ce projet, nous utiliserons donc uniquement les colonnes qui décrivent les films et qui peuvent être renseignées facilement par les utilsateurs de notre application.


```python
# Choix de colonnes 
df = df[[
    'name',
    'genre',
    'year',
    'director',
    'writer',
    'star',
    'company',
    'country_of_release',
]]
```


```python
df['year'] = df['year'].astype('str')
df.dtypes
```




    name                  object
    genre                 object
    year                  object
    director              object
    writer                object
    star                  object
    company               object
    country_of_release    object
    dtype: object



Création de la colonne "cat_features" qui combine tous les mots de toutes les autres colonnes :


```python
'''
df['cat_features'] = df['name'] + ' ' + df['genre'] + ' ' + df['year'] + \
                     ' ' + df['director'] + ' ' + df['writer'] + ' ' + \
                     df['star'] + ' ' + df['company'] + ' ' + df['country_of_release']
'''

df['cat_features'] = df[df.columns].apply(lambda x: ' '.join(x), axis=1)

df.head()
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
      <th>genre</th>
      <th>year</th>
      <th>director</th>
      <th>writer</th>
      <th>star</th>
      <th>company</th>
      <th>country_of_release</th>
      <th>cat_features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Shining</td>
      <td>Drama</td>
      <td>1980</td>
      <td>Stanley Kubrick</td>
      <td>Stephen King</td>
      <td>Jack Nicholson</td>
      <td>Warner Bros.</td>
      <td>United States</td>
      <td>The Shining Drama 1980 Stanley Kubrick Stephen...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Blue Lagoon</td>
      <td>Adventure</td>
      <td>1980</td>
      <td>Randal Kleiser</td>
      <td>Henry De Vere Stacpoole</td>
      <td>Brooke Shields</td>
      <td>Columbia Pictures</td>
      <td>United States</td>
      <td>The Blue Lagoon Adventure 1980 Randal Kleiser ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Star Wars: Episode V - The Empire Strikes Back</td>
      <td>Action</td>
      <td>1980</td>
      <td>Irvin Kershner</td>
      <td>Leigh Brackett</td>
      <td>Mark Hamill</td>
      <td>Lucasfilm</td>
      <td>United States</td>
      <td>Star Wars: Episode V - The Empire Strikes Back...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airplane!</td>
      <td>Comedy</td>
      <td>1980</td>
      <td>Jim Abrahams</td>
      <td>Jim Abrahams</td>
      <td>Robert Hays</td>
      <td>Paramount Pictures</td>
      <td>United States</td>
      <td>Airplane! Comedy 1980 Jim Abrahams Jim Abraham...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Caddyshack</td>
      <td>Comedy</td>
      <td>1980</td>
      <td>Harold Ramis</td>
      <td>Brian Doyle-Murray</td>
      <td>Chevy Chase</td>
      <td>Orion Pictures</td>
      <td>United States</td>
      <td>Caddyshack Comedy 1980 Harold Ramis Brian Doyl...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Vectorisation de la colonne "cat_features"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english', min_df=20)
word_matrix = vectorizer.fit_transform(df['cat_features'])
word_matrix.shape
```




    (5407, 434)




```python
type(word_matrix)
```




    scipy.sparse._csr.csr_matrix



Le tableau du nombre de mots contient 5407 lignes (une pour chaque film) et 434 colonnes. Voici une explication détaillée du code :

Ce code utilise la bibliothèque scikit-learn (`sklearn`) pour effectuer une vectorisation de texte en utilisant la classe `CountVectorizer`. La vectorisation de texte consiste à représenter le texte sous forme de vecteurs numériques, ce qui est souvent nécessaire lorsqu'on travaille avec des modèles d'apprentissage automatique.

Voici une explication du code :

1. **Importation de la classe `CountVectorizer` :**
   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   ```
   Cette ligne importe la classe `CountVectorizer` de la bibliothèque scikit-learn. `CountVectorizer` est utilisé pour convertir une collection de textes en une matrice de comptage des termes (chaque terme étant un mot) présents dans ces textes.

2. **Création d'une instance de `CountVectorizer` :**
   ```python
   vectorizer = CountVectorizer(stop_words='english', min_df=20)
   ```
   On crée une instance de la classe `CountVectorizer`. Les paramètres spécifiés sont :
   - `stop_words='english'` : Cela indique au vectoriseur d'ignorer les mots fréquents en anglais qui ne portent généralement pas beaucoup d'informations (comme "the", "and", "is", etc.).
   - `min_df=20` : Ce paramètre spécifie que le terme doit apparaître dans au moins 20 documents (ou lignes) pour être inclus dans la matrice. Cela permet de filtrer les termes peu fréquents.

3. **Transformation du texte en une matrice de comptage :**
   ```python
   word_matrix = vectorizer.fit_transform(df['cat_features'])
   ```
   On utilise la méthode `fit_transform` pour transformer la colonne `cat_features` du DataFrame `df` en une matrice de comptage. Chaque ligne de la matrice représente un document (dans ce cas, un élément de la colonne `cat_features`), et chaque colonne représente un terme unique.

4. **Affichage de la forme de la matrice :**
   ```python
   word_matrix.shape
   ```
   Cette ligne affiche la forme de la matrice résultante (`word_matrix`). Cela donne le nombre de documents (lignes) et le nombre de termes uniques (colonnes) dans la matrice.

En résumé, ce code utilise `CountVectorizer` pour transformer la colonne de texte `cat_features` en une matrice de comptage, en ignorant les mots fréquents en anglais et en filtrant les termes peu fréquents. La matrice résultante (`word_matrix`) peut ensuite être utilisée comme entrée pour des modèles d'apprentissage automatique.

**Explication sur l'argument `min_df`** :

L'argument `min_df` (fréquence minimale documentaire) dans `CountVectorizer` est utilisé pour spécifier le nombre minimum de documents (ou lignes) dans lesquels un terme doit apparaître pour être inclus dans la matrice de comptage. En d'autres termes, les termes qui apparaissent dans moins de documents que la valeur spécifiée pour `min_df` seront ignorés lors de la création de la matrice de comptage.

L'impact de `min_df` sur le résultat de la vectorisation est lié à la façon dont il filtre les termes peu fréquents. Voici comment il influence le processus de vectorisation :

1. **Termes peu fréquents :** Lorsqu'on travaille avec des jeux de données textuelles, il est courant que de nombreux termes apparaissent seulement dans quelques documents. Certains de ces termes peuvent être des mots spécifiques à certains documents, des fautes de frappe, ou d'autres termes peu représentatifs de l'ensemble du corpus.

2. **Filtrage des termes peu fréquents :** En spécifiant une valeur pour `min_df`, on filtre les termes qui n'apparaissent pas dans un nombre minimal de documents. Cela permet de se débarrasser de termes rares qui pourraient ne pas contribuer de manière significative à la représentation générale du corpus.

3. **Réduction de la dimensionnalité :** En éliminant les termes peu fréquents, la dimensionnalité de la matrice résultante est réduite, ce qui peut être bénéfique en termes de mémoire et de temps de calcul. Une matrice moins dense peut également aider à améliorer la performance des modèles d'apprentissage automatique en réduisant le bruit introduit par des termes peu fréquents.

4. **Réduction du surajustement (overfitting) :** Lorsqu'on travaille avec des modèles d'apprentissage automatique, réduire le nombre de caractéristiques (termes dans ce contexte) peut aider à éviter le surajustement en éliminant des informations spécifiques au jeu de données d'entraînement qui pourraient ne pas généraliser correctement.

Cependant, le choix de la valeur de `min_df` dépend du domaine spécifique, de la taille du corpus, et des objectifs de la tâche. Une valeur trop basse pourrait inclure des termes peu informatifs, tandis qu'une valeur trop élevée pourrait éliminer des termes utiles. Il est souvent recommandé de tester différentes valeurs de `min_df` et de surveiller l'impact sur les performances du modèle pour trouver la valeur optimale.

Pour plus d'informations sur les hyperparamètres de la classe `CountVectorizer` : https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

La tâche suivante consiste à calculer les similarités cosinus pour chaque paire de lignes :


```python
from sklearn.metrics.pairwise import cosine_similarity
 
sim = cosine_similarity(word_matrix)
type(sim)
```




    numpy.ndarray




```python
sim
```




    array([[1.        , 0.3354102 , 0.35856858, ..., 0.38138504, 0.19069252,
            0.31622777],
           [0.3354102 , 1.        , 0.40089186, ..., 0.42640143, 0.31980107,
            0.35355339],
           [0.35856858, 0.40089186, 1.        , ..., 0.22792115, 0.22792115,
            0.56694671],
           ...,
           [0.38138504, 0.42640143, 0.22792115, ..., 1.        , 0.27272727,
            0.30151134],
           [0.19069252, 0.31980107, 0.22792115, ..., 0.27272727, 1.        ,
            0.30151134],
           [0.31622777, 0.35355339, 0.56694671, ..., 0.30151134, 0.30151134,
            1.        ]])



Ce code utilise la bibliothèque scikit-learn (`sklearn`) pour calculer la similarité cosinus entre les documents représentés par la matrice `word_matrix`. La similarité cosinus est une mesure de similarité entre deux vecteurs dans un espace vectoriel, souvent utilisée pour mesurer la similarité entre des documents dans le contexte de la vectorisation de texte.

Voici une explication du code :

1. **Importation de la fonction `cosine_similarity` :**
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   ```
   Cette ligne importe la fonction `cosine_similarity` de scikit-learn. Cette fonction calcule la similarité cosinus entre les paires de vecteurs.

2. **Calcul de la similarité cosinus :**
   ```python
   sim = cosine_similarity(word_matrix)
   ```
   On utilise la fonction `cosine_similarity` pour calculer la similarité cosinus entre tous les documents représentés dans la matrice `word_matrix`. Le résultat, stocké dans la variable `sim`, est une matrice symétrique où l'élément à la position (i, j) représente la similarité cosinus entre le document i et le document j.

   Notez que la diagonale principale de cette matrice contient des valeurs égales à 1, car la similarité cosinus d'un document avec lui-même est toujours égale à 1.

3. **Interprétation de la matrice de similarité :**
   La matrice de similarité `sim` peut être utilisée pour comprendre les relations de similarité entre les documents. Plus la valeur dans la matrice est proche de 1, plus les documents correspondants sont similaires, selon la mesure cosinus.

   Vous pouvez utiliser cette matrice pour, par exemple, trouver les documents les plus similaires à un document donné, identifier des groupes de documents similaires, ou alimenter des systèmes de recommandation basés sur la similarité entre documents.

En résumé, ce code calcule la similarité cosinus entre tous les documents représentés dans la matrice `word_matrix` et stocke les résultats dans une matrice appelée `sim`.


```python
# Sauvegarde de la matrice de similarité précalculée
import numpy as np
np.save('similarity_matrix.npy', sim)
```

Construisons une fonction qui prend comme arguments des informations relatifs à un film et retourne les n films les plus similaires à ce film :


```python
def get_recommendations(title, df, sim, count=10):
    # Obtenir l'indice de ligne du titre spécifié dans le DataFrame
    index = df.index[df['name'].str.lower() == title.lower()]
    
    # Retourner une liste vide s'il n'y a aucune entrée pour le titre spécifié
    if len(index) == 0:
        return []

    # Vérifier si l'indice est dans les limites de la matrice de similarité
    if index[0] >= len(sim):
        return []

    # Obtenir la ligne correspondante dans la matrice de similarité
    similarities = list(enumerate(sim[index[0]]))
    
    # Trier les scores de similarité dans cette ligne par ordre décroissant
    recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Obtenir les n meilleures recommandations, en ignorant la première entrée de la liste car
    # elle correspond au titre lui-même (et a donc une similarité de 1.0)
    top_recs = recommendations[1:count + 1]

    # Générer une liste de titres à partir des indices dans top_recs
    titles = []

    for i in range(len(top_recs)):
        # Vérifier si l'indice est dans les limites du DataFrame
        if top_recs[i][0] < len(df):
            title = df.iloc[top_recs[i][0]]['name']
            titles.append(title)

    return titles

```


```python
# Application de la fonction #title="The Blue Lagoon",
get_recommendations(
    title="The Blue Lagoon",
    df=df,
    sim=sim,
    count=5
)
```




    ['Return to the Blue Lagoon',
     'Old Gringo',
     'The New Adventures of Pippi Longstocking',
     'Fly Away Home',
     'The Messenger: The Story of Joan of Arc']



La fonction `get_recommendations` prend en entrée le titre d'un film, un DataFrame `df`, une matrice de similarité `sim` et un paramètre optionnel `count` définissant le nombre de recommandations à retourner. Voici une explication détaillée de chaque étape de la fonction :

**Explication détaillée** :

1. **Recherche de l'indice du titre dans le DataFrame :**
   ```python
   index = df.index[df['name'].str.lower() == title.lower()]
   ```
   Cette ligne utilise la méthode `str.lower()` pour s'assurer que la comparaison n'est pas sensible à la casse. Elle recherche l'indice de la ligne correspondant au titre spécifié dans le DataFrame.

2. **Vérification de l'existence du titre dans le DataFrame :**
   ```python
   if (len(index) == 0):
       return []
   ```
   Si aucun indice n'est trouvé (c'est-à-dire si le titre n'existe pas dans le DataFrame), la fonction retourne une liste vide.

3. **Récupération de la ligne correspondante dans la matrice de similarité :**
   ```python
   similarities = list(enumerate(sim[index[0]]))
   ```
   La fonction crée une liste de tuples `(index, similarity_score)` en utilisant la fonction `enumerate` pour obtenir l'indice correspondant dans la matrice de similarité.

4. **Tri des scores de similarité par ordre décroissant :**
   ```python
   recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
   ```
   Les scores de similarité sont triés par ordre décroissant, plaçant les films les plus similaires en haut de la liste.

5. **Sélection des meilleures recommandations (hors titre lui-même) :**
   ```python
   top_recs = recommendations[1:count + 1]
   ```
   Les `count` films les plus similaires sont sélectionnés à partir de la liste triée, en ignorant le premier élément qui correspond au titre lui-même (similarity_score de 1.0).

6. **Génération de la liste des titres recommandés :**
   ```python
   titles = []
   for i in range(len(top_recs)):
       title = df.iloc[top_recs[i][0]]['name']
       titles.append(title)
   ```
   La fonction parcourt les indices des films recommandés (`top_recs`), récupère les titres correspondants à partir du DataFrame original, et les ajoute à la liste `titles`.

7. **Retour de la liste des titres recommandés :**
   ```python
   return titles
   ```
   La fonction retourne la liste des titres recommandés en fonction de la similarité avec le titre spécifié.

Il ne vous reste qu'à réer une interface utilisateur simple avec des champs de saisie et des listes déroulantes pour chaque argument de la fonction get_recommendations. Lorsque l'utilisateur clique sur un bouton, les recommandations sont calculées à l'aide de la fonction et affichées sur la page de l'application.

Dans ce projet, nous utilserons Streamlit pour construire l'application web.


🖥 **Script utils.py contenant les fonctions utilisées dans ce Projet**

```python
# Librairies
import pandas as pd
import numpy as np

def load_clean_movie_data(movie_file):
    data = pd.read_csv(movie_file)
    data.dropna(inplace=True)
    data[['date_of_release', 'country_of_release']] = data['released'].str.extract(r'(\w+ \d+, \d+) \(([^)]+)\)')
    data.drop(['released', 'date_of_release'], axis=1, inplace=True)
    data.dropna(inplace=True)
    data = data[[
        'name',
        'genre',
        'year',
        'director',
        'writer',
        'star',
        'company',
        'country_of_release',
    ]]
    data['year'] = data['year'].astype('str')
    data['cat_features'] = data[data.columns].apply(lambda x: ' '.join(x), axis=1)

    return data

def get_recommendations(title, df, sim, count=10):
    # Obtenir l'indice de ligne du titre spécifié dans le DataFrame
    index = df.index[df['name'].str.lower() == title.lower()]
    
    # Retourner une liste vide s'il n'y a aucune entrée pour le titre spécifié
    if len(index) == 0:
        return []

    # Vérifier si l'indice est dans les limites de la matrice de similarité
    if index[0] >= len(sim):
        return []

    # Obtenir la ligne correspondante dans la matrice de similarité
    similarities = list(enumerate(sim[index[0]]))
    
    # Trier les scores de similarité dans cette ligne par ordre décroissant
    recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Obtenir les n meilleures recommandations, en ignorant la première entrée de la liste car
    # elle correspond au titre lui-même (et a donc une similarité de 1.0)
    top_recs = recommendations[1:count + 1]

    # Générer une liste de titres à partir des indices dans top_recs
    titles = []

    for i in range(len(top_recs)):
        # Vérifier si l'indice est dans les limites du DataFrame
        if top_recs[i][0] < len(df):
            title = df.iloc[top_recs[i][0]]['name']
            titles.append(title)

    return titles
```

Les deux fonctions, `load_clean_movie_data` et `get_recommendations`, sont utilisées pour charger, nettoyer les données de films et générer des recommandations de films basées sur la similarité. Voici une explication détaillée de chaque fonction et de leur rôle :

- **Fonction `load_clean_movie_data`** :

Cette fonction charge les données de films à partir d'un fichier CSV, les nettoie et les prépare pour une utilisation ultérieure.


1. Chargement des données :

    ```python
    data = pd.read_csv(movie_file)
    ```
    Charge les données depuis un fichier CSV dans un DataFrame `data`.

2. Suppression des valeurs manquantes :

    ```python
    data.dropna(inplace=True)
    ```
    Supprime toutes les lignes contenant des valeurs manquantes.

3. Extraction des informations de la colonne `released` :

    ```python
    data[['date_of_release', 'country_of_release']] = data['released'].str.extract(r'(\w+ \d+, \d+) \(([^)]+)\)')
    ```
    Utilise une expression régulière pour extraire la date de sortie et le pays de sortie à partir de la colonne `released`.

4. Suppression des colonnes inutiles :

    ```python
    data.drop(['released', 'date_of_release'], axis=1, inplace=True)
    ```
    Supprime les colonnes `released` et `date_of_release` du DataFrame.

5. Suppression des valeurs manquantes :

    ```python
    data.dropna(inplace=True)
    ```
    Supprime à nouveau toutes les lignes contenant des valeurs manquantes après l'extraction.

6. Sélection des colonnes utiles :

    ```python
    data = data[[
        'name',
        'genre',
        'year',
        'director',
        'writer',
        'star',
        'company',
        'country_of_release',
    ]]
    ```
    Garde uniquement les colonnes nécessaires pour l'analyse.

7. Conversion de l'année en chaîne de caractères :

    ```python
    data['year'] = data['year'].astype('str')
    ```
    Convertit la colonne `year` en type chaîne de caractères.

8. Création d'une colonne de caractéristiques concaténées :

    ```python
    data['cat_features'] = data[data.columns].apply(lambda x: ' '.join(x), axis=1)
    ```
    Crée une nouvelle colonne `cat_features` qui contient une concaténation des valeurs de toutes les colonnes, facilitant ainsi la génération de la matrice de similarité.

9. Retour des données nettoyées :

    ```python
    return data
    ```
    Retourne le DataFrame nettoyé.


- **Fonction `get_recommendations`** :

1. Trouver l'indice du film spécifié :

    ```python
    index = df.index[df['name'].str.lower() == title.lower()]
    ```
    Trouve l'indice de la ligne du film dont le titre correspond au titre spécifié (insensible à la casse).

2. Gérer les cas où le film n'est pas trouvé :

    ```python
    if len(index) == 0:
        return []
    ```
    Si le film n'est pas trouvé dans le DataFrame, retourne une liste vide.

3. Vérifier la validité de l'indice :

    ```python
    if index[0] >= len(sim):
        return []
    ```
    Vérifie que l'indice trouvé est valide par rapport à la matrice de similarité.

4. Obtenir la similarité pour le film spécifié :

    ```python
    similarities = list(enumerate(sim[index[0]]))
    ```
    Obtient les scores de similarité pour le film spécifié en récupérant la ligne correspondante dans la matrice de similarité.

5. Trier les films par similarité décroissante :

    ```python
    recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
    ```
    Trie les films par ordre décroissant de similarité.

6. Sélectionner les films les plus similaires :

    ```python
    top_recs = recommendations[1:count + 1]
    ```
    Sélectionne les `count` meilleures recommandations en ignorant la première entrée (qui correspond au film lui-même).

7. Générer une liste de titres des films recommandés :

    ```python
    titles = []

    for i in range(len(top_recs)):
        if top_recs[i][0] < len(df):
            title = df.iloc[top_recs[i][0]]['name']
            titles.append(title)
    ```
    Pour chaque film recommandé, vérifie que l'indice est valide et ajoute le titre du film à la liste des titres recommandés.

8. Retourner la liste des films recommandés :

    ```python
    return titles
    ```
    Retourne la liste des titres des films recommandés.

Ces deux fonctions permettent de préparer les données de films en les nettoyant et en les formatant, puis de générer des recommandations de films en utilisant une matrice de similarité pré-calculée. L'application de recommandation de films utilise ces fonctions pour fournir une interface interactive où les utilisateurs peuvent obtenir des suggestions de films basées sur un film de référence.


🖥 **Script de l'application web de Recommandation de films (app.py)**

```python
import streamlit as st
import numpy as np
from utils import load_clean_movie_data, get_recommendations

# Chargement et Nettoyage du DataFrame 
movie_data = load_clean_movie_data("movies.csv")


# Affichage du titre de l'application
st.title('Application de Recommandation de films')

# Inputs de l'utilisateur
name = st.selectbox('Nom du film', movie_data['name'].unique())
num_recommendations = st.number_input('Nombre de films à recommander', min_value=1, value=5)

# Bouton pour obtenir les recommandations
if st.button('Obtenir les recommandations'):
    # Charger la matrice de similarité précalculée
    similarity_matrix_loaded = np.load('similarity_matrix.npy')

    # Utiliser la matrice de similarité précalculée pour les recommandations
    recommendations = get_recommendations(
        title=name, df=movie_data, 
        sim=similarity_matrix_loaded, 
        count=num_recommendations
    )
    st.write('Films recommandés :', recommendations)
```

Ce code utilise Streamlit pour créer une application web interactive de recommandation de films. Voici une explication détaillée de chaque partie du code :

- **Importation des Bibliothèques** :

```python
import streamlit as st
import numpy as np
from utils import load_clean_movie_data, get_recommendations
```
Ces lignes importent les bibliothèques nécessaires :
- `streamlit` pour créer l'interface web interactive.
- `numpy` pour manipuler les données numériques, notamment la matrice de similarité.
- Les fonctions `load_clean_movie_data` et `get_recommendations` sont importées du module `utils`, défini ailleurs dans le projet.

- **Chargement et Nettoyage du DataFrame** :

```python
movie_data = load_clean_movie_data("movies.csv")
```
Cette ligne charge et nettoie les données des films à partir d'un fichier CSV nommé `movies.csv` en utilisant la fonction `load_clean_movie_data`. La fonction retourne un DataFrame `movie_data` contenant les informations sur les films.

- **Affichage du Titre de l'Application** :

```python
st.title('Application de Recommandation de films')
```
Cette ligne définit le titre de l'application qui sera affiché en haut de la page.

- **Inputs de l'Utilisateur** :

```python
name = st.selectbox('Nom du film', movie_data['name'].unique())
num_recommendations = st.number_input('Nombre de films à recommander', min_value=1, value=5)
```
Ces lignes créent des widgets interactifs pour l'utilisateur :
- Un menu déroulant (`selectbox`) pour sélectionner le nom d'un film parmi ceux présents dans `movie_data`.
- Un champ de saisie numérique (`number_input`) pour entrer le nombre de recommandations souhaitées, avec une valeur par défaut de 5.

- **Bouton pour Obtenir les Recommandations** :

```python
if st.button('Obtenir les recommandations'):
    # Charger la matrice de similarité précalculée
    similarity_matrix_loaded = np.load('similarity_matrix.npy')

    # Utiliser la matrice de similarité précalculée pour les recommandations
    recommendations = get_recommendations(
        title=name, df=movie_data, 
        sim=similarity_matrix_loaded, 
        count=num_recommendations
    )
    st.write('Films recommandés :', recommendations)
```
Cette section définit le comportement lorsqu'on clique sur le bouton "Obtenir les recommandations" :

1. *Chargement de la Matrice de Similarité* :
    ```python
    similarity_matrix_loaded = np.load('similarity_matrix.npy')
    ```
    Cette ligne charge une matrice de similarité précalculée à partir d'un fichier nommé `similarity_matrix.npy`. Cette matrice est utilisée pour déterminer la similarité entre les films.

2. *Obtention des Recommandations* :
    ```python
    recommendations = get_recommendations(
        title=name, df=movie_data, 
        sim=similarity_matrix_loaded, 
        count=num_recommendations
    )
    ```
    Cette ligne appelle la fonction `get_recommendations` en passant les paramètres nécessaires :
    - `title` : le nom du film sélectionné par l'utilisateur.
    - `df` : le DataFrame contenant les données des films.
    - `sim` : la matrice de similarité chargée.
    - `count` : le nombre de recommandations souhaitées.

3. *Affichage des Recommandations* :
    ```python
    st.write('Films recommandés :', recommendations)
    ```
    Cette ligne affiche les films recommandés dans l'interface utilisateur.

L'application permet à l'utilisateur de sélectionner un film et de spécifier le nombre de recommandations souhaitées. En cliquant sur le bouton, l'application utilise une matrice de similarité précalculée pour trouver et afficher les films similaires. Cette matrice et les fonctions de nettoyage et de recommandation sont essentielles pour le bon fonctionnement de l'application, mais ne sont pas incluses dans le code fourni.


- ***Conclusion du Projet***

Le projet a consisté en la conception et la mise en œuvre d'un système de recommandation de films basé sur la similarité cosinus. Voici les points clés et les réalisations du projet :

1. **Objectif du Projet :** L'objectif principal était de créer un système capable de recommander des films similaires en fonction des caractéristiques textuelles des films.

2. **Utilisation de la Similarité Cosinus :** La similarité cosinus a été choisie comme métrique pour mesurer la similarité entre les films. C'est une mesure efficace pour évaluer la proximité entre des vecteurs de données textuelles.

3. **Prétraitement des Données :** Les données du projet ont été prétraitées, notamment en nettoyant la colonne des dates de sortie, en construisant une colonne de caractéristiques textuelles concaténées, et en utilisant la technique de similarité cosinus sur ces caractéristiques.

4. **Construction de la Matrice de Similarité :** Une matrice de similarité a été calculée à partir des caractéristiques textuelles des films, ce qui a servi de base pour la recommandation.

5. **Fonction de Recommandation :** Une fonction `get_recommendations` a été développée pour fournir des recommandations de films en fonction d'un titre donné. La fonction utilise la matrice de similarité et le DataFrame des films.

6. **Application Streamlit :** Une interface utilisateur simple a été créée en utilisant Streamlit, permettant à l'utilisateur de saisir les informations d'un film et de recevoir des recommandations en temps réel.

7. **Gestion des Erreurs :** Des vérifications ont été ajoutées pour éviter les erreurs potentielles, notamment la vérification de l'existence du titre dans le DataFrame et la vérification des indices dans la matrice de similarité.

8. **Documentation et Commentaires :** Le code a été commenté de manière appropriée pour assurer une compréhension claire du fonctionnement du système. La documentation des fonctions et des étapes importantes a été fournie.

9. **Sauvegarde de la Matrice de Similarité :** Pour optimiser les performances, la matrice de similarité a été sauvegardée dans un fichier afin d'être réutilisée sans avoir besoin d'une nouvelle génération à chaque utilisation de l'application.

10. **Perspectives d'Amélioration :** Le projet pourrait être étendu en ajoutant des fonctionnalités supplémentaires, telles que la prise en compte des évaluations utilisateur pour des recommandations plus personnalisées.

En conclusion, ce projet a permis de mettre en œuvre avec succès un système de recommandation de films basé sur la similarité cosinus, offrant une expérience utilisateur interactive à travers une interface simple grâce à Streamlit. Ce projet fournit une base solide pour d'autres développements dans le domaine des systèmes de recommandation de contenu.