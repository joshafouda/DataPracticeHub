# Projet : Système de recommandations de films

## Description

L'objectif de ce projet est de construire une application de système de recommandations de films. L'utilisateur de cette application renseigne les informations relatives à un film donné et l'application lui retourne les n films les plus similaires à ce film donné.

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

## Image
https://via.placeholder.com/150

## Instructions
1. Étape 1 : Description de l'étape 1
2. Étape 2 : Description de l'étape 2
3. Étape 3 : Description de l'étape 3

## Resources
- [Lien vers une ressource](https://example.com)
- [Lien vers une autre ressource](https://example.com)

## Execution du Projet

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


