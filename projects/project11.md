# Projet : Popularité des langages de Programmation informatique

## Description

Ce projet a pour objectif de visualiser l'évolution de la popularité des différents langages de programmation entre 2004 et 2023. Il utilise des données de popularité mensuelle des langages de programmation pour créer une série chronologique interactive permettant aux utilisateurs de comparer plusieurs langages sur une période spécifique.

## Image
imgs/project11/project11.png

## Instructions

1. **Importation des packages** :
   - Installez les bibliothèques nécessaires : `shiny`, `pandas`, `numpy`, `pathlib`, et `plotnine`.
permettra de disposer du bouton "Run Shiny App" pour exécuter votre application (Voir vidéo : https://youtu.be/XHWQR5B8evo?si=__0ucRzMm2hA-F_5)

2. **Installez l'extension Shiny dans Visual Studio Code** : Cette extension vous 

3. **Importation et préparation des données** :
   - Chargez le fichier CSV contenant les [données de popularité des langages de programmation](https://drive.google.com/file/d/1A_kxtGDHmGcbwjA_fsjvVNRNVEcDvD2F/view?usp=sharing).
   - Convertissez la colonne des dates au format datetime.
   - Transformez les données en format long pour les rendre adaptées à `plotnine`.

4. **Création de l'interface utilisateur** :
   - Créez une interface utilisateur avec une barre latérale.
   - Ajoutez des widgets pour permettre la sélection des langages de programmation et la période de temps.
   - Affichez un graphique de la série chronologique dans le panneau principal.

5. **Développement du serveur (backend)** :
   - Filtrez les données en fonction des langages et de la période sélectionnés par l'utilisateur.
   - Créez un graphique de la série chronologique utilisant `plotnine`.

6. **Lancement de l'application** :
   - Exécutez l'application Shiny pour visualiser les données de popularité des langages de programmation de manière interactive.

7. **Déploiement en ligne de l'application** : Suivez la vidéo pour apprendre comment déployer votre application Shiny : https://youtu.be/XHWQR5B8evo?si=__0ucRzMm2hA-F_5


## Resources

- [Jeu de données](https://drive.google.com/file/d/1A_kxtGDHmGcbwjA_fsjvVNRNVEcDvD2F/view?usp=sharing)
- [Source des données utilisées dans le projet](https://www.kaggle.com/datasets/muhammadkhalid/most-popular-programming-languages-since-2004)
- [Formation Développement web avec Shiny for Python](https://youtu.be/XHWQR5B8evo)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)

## Execution du Projet

🖥 **Script app.py qui comporte le code de lápplication**

```python
# Importation des packages
from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import ggplot, aes, geom_line, theme, element_text, labs

# Importation et préparation des données
def load_data():
    return pd.read_csv(Path(__file__).parent / 'PopularityofProgrammingLanguagesfrom2004to2023.csv')

def data_preparation():
    raw_data = load_data()
    raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%B %Y')
    long_data = raw_data.melt(
        id_vars='Date',
        value_name='popularity',
        var_name='langage'
    ).reset_index(drop=True)
    return long_data

clean_data = data_preparation()
date_start = np.min(clean_data['Date'])
date_end = np.max(clean_data['Date'])
noms = clean_data['langage'].unique()
noms_dict = {l:l for l in noms}

# Interface Utilisateur
app_ui = ui.page_fluid(
    ui.panel_title("Popularité des langages de Programmation"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_selectize(
                id='langage',
                label="Choisis un ou plusieurs langages :",
                choices=noms_dict,
                selected="Python",
                multiple=True
            ),
            ui.input_date_range(
                id='date_range',
                label="Choisis une période :",
                start=date_start,
                end=date_end
            ),
        ),
        ui.panel_main(
            ui.output_plot("PlotTimeserie")
        )
    ),
)

# Serveur (backend)
def server(input, output, session):
    @reactive.Calc
    def filtered_data():
        date_selected_start = pd.to_datetime(input.date_range()[0])
        date_selected_end = pd.to_datetime(input.date_range()[1])

        df = clean_data.loc[(clean_data['langage'].isin(list(input.langage()))) &
                            (clean_data['Date'] >= date_selected_start) &
                            (clean_data['Date'] <= date_selected_end)].reset_index(drop=True)
        
        return df

    
    @output
    @render.plot
    def PlotTimeserie():
        g = ggplot(filtered_data()) + \
        aes(x = 'Date', y = 'popularity', color = 'langage') + \
        geom_line() + \
        labs(x = 'Date', y = 'Popularity [%]', title = 'Popularity over Time') + \
        theme(axis_text=element_text(rotation=90, hjust=1))
        return g


app = App(app_ui, server)

```

Ce projet utilise la bibliothèque `shiny` pour Python afin de créer une application web interactive qui permet de visualiser la popularité des langages de programmation au fil du temps. Voici une explication étape par étape du code :

- Importation des packages

```python
from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import ggplot, aes, geom_line, theme, element_text, labs
```

- **shiny** : Utilisé pour créer l'application web.
- **pandas** : Utilisé pour manipuler les données.
- **numpy** : Utilisé pour les opérations numériques.
- **pathlib** : Utilisé pour manipuler les chemins de fichiers.
- **plotnine** : Utilisé pour créer des graphiques.

- Importation et préparation des données

```python
def load_data():
    return pd.read_csv(Path(__file__).parent / 'PopularityofProgrammingLanguagesfrom2004to2023.csv')

def data_preparation():
    raw_data = load_data()
    raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%B %Y')
    long_data = raw_data.melt(
        id_vars='Date',
        value_name='popularity',
        var_name='langage'
    ).reset_index(drop=True)
    return long_data

clean_data = data_preparation()
date_start = np.min(clean_data['Date'])
date_end = np.max(clean_data['Date'])
noms = clean_data['langage'].unique()
noms_dict = {l:l for l in noms}
```

1. **load_data()** : Charge les données depuis un fichier CSV.
2. **data_preparation()** :
   - Convertit la colonne 'Date' en format datetime.
   - Transforme les données de format large à format long pour faciliter leur utilisation dans les graphiques.
3. **Variables Globales** :
   - `clean_data` : Contient les données préparées.
   - `date_start` et `date_end` : Les dates de début et de fin des données.
   - `noms` : Les noms des langages de programmation.
   - `noms_dict` : Un dictionnaire des noms de langages pour les choix de l'utilisateur.

- Interface Utilisateur

```python
app_ui = ui.page_fluid(
    ui.panel_title("Popularité des langages de Programmation"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_selectize(
                id='langage',
                label="Choisis un ou plusieurs langages :",
                choices=noms_dict,
                selected="Python",
                multiple=True
            ),
            ui.input_date_range(
                id='date_range',
                label="Choisis une période :",
                start=date_start,
                end=date_end
            ),
        ),
        ui.panel_main(
            ui.output_plot("PlotTimeserie")
        )
    ),
)
```

- **app_ui** : Définit l'interface utilisateur de l'application.
  - `ui.page_fluid` : Utilise une mise en page fluide.
  - `ui.panel_title` : Définit le titre de la page.
  - `ui.layout_sidebar` : Utilise une disposition avec une barre latérale.
    - `ui.panel_sidebar` : Contient des widgets pour les choix de l'utilisateur :
      - `ui.input_selectize` : Permet de choisir un ou plusieurs langages.
      - `ui.input_date_range` : Permet de choisir une période de temps.
    - `ui.panel_main` : Contient le graphique généré.

- Serveur (backend)

```python
def server(input, output, session):
    @reactive.Calc
    def filtered_data():
        date_selected_start = pd.to_datetime(input.date_range()[0])
        date_selected_end = pd.to_datetime(input.date_range()[1])

        df = clean_data.loc[(clean_data['langage'].isin(list(input.langage()))) &
                            (clean_data['Date'] >= date_selected_start) &
                            (clean_data['Date'] <= date_selected_end)].reset_index(drop=True)
        
        return df

    @output
    @render.plot
    def PlotTimeserie():
        g = ggplot(filtered_data()) + \
        aes(x = 'Date', y = 'popularity', color = 'langage') + \
        geom_line() + \
        labs(x = 'Date', y = 'Popularity [%]', title = 'Popularity over Time') + \
        theme(axis_text=element_text(rotation=90, hjust=1))
        return g

app = App(app_ui, server)
```

- **server** : Définit les fonctions backend pour l'application.
  - **filtered_data()** :
    - Filtre les données en fonction des choix de l'utilisateur (langages et période de temps).
  - **PlotTimeserie()** :
    - Crée le graphique de la popularité au fil du temps en utilisant `ggplot`.

- Lancement de l'application

```python
app = App(app_ui, server)
```

- **app** : Crée l'application en combinant l'interface utilisateur et le serveur.

Ce projet permet à l'utilisateur de visualiser la popularité des langages de programmation en sélectionnant les langages et la période de temps souhaités, et affiche un graphique interactif basé sur ces choix.