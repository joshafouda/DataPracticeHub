# Projet : Construction d'une application Web pour l'analyse technique des actifs boursiers avec Streamlit

## Description

Ce projet vise à construire une application Web utilisant Streamlit pour effectuer une analyse technique des actifs boursiers. L'application permet de visualiser divers indicateurs techniques pour une sélection d'actions dans les plus importants marchés boursiers du monde (S&P500, CAC40, DAX, FTSE100 et Nikkei225), aidant ainsi les utilisateurs à prendre des décisions d'investissement informées.

## Image
imgs/project8/project8.png

## Instructions

L'interface graphique que vous devez construire doit être semblable à cette application [TechnicalAnalysisWebApp](https://financeapp-rfsbok6xwjgezg6tztvfiy.streamlit.app/TechnicalAnalysis). De plus, voici une vidéo de présentation de l'application à reproduire : https://youtu.be/vNEKXY8V-qQ?si=ofEj0VDxDF7RFn4J

L'application doit offrir les fonctionnalités suivantes aux utilisateurs :

- **Sélection d'indices de marché**

Les utilisateurs peuvent choisir parmi une variété d’indices de marché, notamment :

     S&P 500 : couvre les plus grandes sociétés cotées sur les bourses américaines.

     CAC 40 : Représente les 40 premières sociétés cotées sur Euronext Paris.

     DAX : présente les 30 plus grandes sociétés cotées à la Bourse de Francfort.

     FTSE 100 : se compose des 100 plus grandes sociétés cotées à la Bourse de Londres.

     Nikkei 225 : comprend 225 sociétés de premier plan cotées à la Bourse de Tokyo.

- **Visualisation de la dataframe d'un Actif avec possibilité de télécharger les données au format CSV**

- **Analyse technique**

Votre application doit proposer plusieurs outils d'analyse technique, notamment :

     Moyenne mobile simple (SMA) : aide à identifier les tendances en lissant les données de prix sur un nombre spécifié de périodes.

     Bandes de Bollinger : affiche les conditions potentielles de surachat ou de survente.

     Relative Strength Index (RSI) : mesure la vitesse et l’évolution des mouvements de prix.


La principale utilité de cette application réside dans sa capacité à doter les utilisateurs des outils et des données nécessaires pour prendre des décisions financières éclairées. Voici quelques avantages clés :

     Accès aux données : accès aux données historiques et en temps réel pour divers indices de marché et sociétés.

     Analyse technique : effectuez facilement une analyse technique approfondie.

     Flexibilité : Choisissez votre indice boursier et personnalisez vos paramètres d'analyse.

     Convivial : une interface simple et intuitive rend l'application adaptée aux traders de tous niveaux. Oui


## Resources
- [Composants du S&P500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- [Composants du DAX](https://en.wikipedia.org/wiki/DAX)
- [Composants du NIKKEI225](https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/)
- [Composants du FTSE](https://en.wikipedia.org/wiki/FTSE_100_Index)
- [Composants du CAC40](https://en.wikipedia.org/wiki/CAC_40)
- [Lien de l'application à reproduire](https://financeapp-rfsbok6xwjgezg6tztvfiy.streamlit.app/TechnicalAnalysis)
- [Vidéo présentant l'application à reproduire](https://youtu.be/vNEKXY8V-qQ?si=kSKxU6cPbB1Jxijx)
- [WebScraping des données boursières avec BeautifulSoup et Python : cas du NIKKEI225](https://youtu.be/JaOaHeN3tfg)
- [WebScraping des données boursières du CAC40 avec Python](https://youtu.be/VFFgPj2hNKA)
- [Comment scraper les données boursières du S&P500, CAC40, DAX, FTSE100 et NIKKEI225 ?](https://youtu.be/Y3Bqei-FVvU)
- [Collecter des données financières avec Python sur Yahoo Finance](https://youtu.be/KmZvTDuAiYc)
- [Analyse financière et Gestion des Risques avec Python: Application à la création et l'optimisation des Portefeuilles d'actions ](https://www.amazon.fr/dp/B08NWWYBRR?ref_=ast_author_ofdp)
- [Formation Streamlit](https://www.youtube.com/playlist?list=PLmJWMf9F8euQKADN-mSCpTlp7uYDyCQNF)
- [Comment déployer une web app Streamlit](https://youtu.be/wjRlWuXmlvw)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)


## Execution du Projet

Pour réaliser ce projet, vous aurez besoin d'installer les packages suivants :

streamlit

yfinance

pandas

cufflinks

plotly-express

requests

beautifulsoup4


🖥 **Script utils.py qui comporte les fonctions utiles à l'application**

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import seaborn as sns 
import matplotlib.pyplot as plt



# data functions
@st.cache_resource
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Security"])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_dax_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/DAX")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_nikkei_components():
    # Define the URL
    url = "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table based on its class attribute (you may need to inspect the HTML to get the exact class name)
        table = soup.find('table', {'class': 'tablepress'})

        # Use Pandas to read the table and store it as a DataFrame
        df = pd.read_html(str(table))[0]
        df['Code'] = df['Code'].astype(str) + '.T'
    else:
        print("Failed to retrieve the web page. Status code:", response.status_code)
    tickers = df["Code"].to_list()
    tickers_companies_dict = dict(
        zip(df["Code"], df['Company Name'])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_ftse_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/FTSE_100_Index")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict


@st.cache_resource
def get_cac40_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/CAC_40")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict


@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

@st.cache_resource
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")


def display_data_preview(title, dataframe, file_name="close_stock_prices.csv", key=0):
    data_exp = st.expander(title)
    available_cols = dataframe.columns.tolist()
    columns_to_show = data_exp.multiselect(
        "Columns", 
        available_cols, 
        default=available_cols,
        key=key
    )
    data_exp.dataframe(dataframe[columns_to_show])

    csv_file = convert_df_to_csv(dataframe[columns_to_show])
    data_exp.download_button(
        label="Download selected as CSV",
        data=csv_file,
        file_name=file_name,
        mime="text/csv",
    )    
```

Le script `utils.py` contient plusieurs fonctions utiles pour récupérer des données financières, les transformer et les afficher dans une application Streamlit. Voici une explication détaillée de chaque fonction pour les débutants.

- Importations
Le script commence par importer les bibliothèques nécessaires pour son fonctionnement.

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import seaborn as sns 
import matplotlib.pyplot as plt
```

- Fonctions pour Récupérer les Composants des Indices Boursiers

1. **Récupérer les composants du S&P 500**

```python
@st.cache_resource
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Security"])
    )
    return tickers, tickers_companies_dict
```

- **`@st.cache_resource`** : Cette décoration permet de mettre en cache les résultats de la fonction pour éviter de recharger les données à chaque exécution, améliorant ainsi les performances.
- **`pd.read_html`** : Lit toutes les tables HTML de la page Wikipedia spécifiée.
- **`df[0]`** : Sélectionne la première table trouvée.
- **`tickers`** : Liste des symboles des actions (tickers).
- **`tickers_companies_dict`** : Dictionnaire associant chaque ticker à son nom de société.

2. **Récupérer les composants du DAX**

```python
@st.cache_resource
def get_dax_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/DAX")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict
```

- Semblable à la fonction précédente, mais pour l'indice DAX. Ici, la cinquième table est sélectionnée (`df[4]`).

3. **Récupérer les composants du Nikkei 225**

```python
@st.cache_resource
def get_nikkei_components():
    url = "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'tablepress'})
        df = pd.read_html(str(table))[0]
        df['Code'] = df['Code'].astype(str) + '.T'
    else:
        print("Failed to retrieve the web page. Status code:", response.status_code)
    tickers = df["Code"].to_list()
    tickers_companies_dict = dict(
        zip(df["Code"], df['Company Name'])
    )
    return tickers, tickers_companies_dict
```

- Utilise `requests` pour obtenir le contenu HTML de la page spécifiée.
- Utilise `BeautifulSoup` pour parser le contenu HTML.
- Convertit le contenu HTML de la table en DataFrame.
- Ajoute ".T" aux codes des actions pour indiquer qu'ils sont cotés à Tokyo.

4. **Récupérer les composants du FTSE 100**

```python
@st.cache_resource
def get_ftse_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/FTSE_100_Index")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict
```

- Semblable à la fonction précédente, mais pour l'indice FTSE 100.

5. **Récupérer les composants du CAC 40**

```python
@st.cache_resource
def get_cac40_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/CAC_40")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict
```

- Semblable à la fonction précédente, mais pour l'indice CAC 40.

- Fonction pour Charger les Données de Yahoo Finance

```python
@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)
```

- **`@st.cache_data`** : Décoration pour mettre en cache les données téléchargées, évitant de recharger les mêmes données plusieurs fois.
- **`yf.download`** : Télécharge les données historiques pour un symbole d'action donné entre les dates de début et de fin spécifiées.

- Fonction pour Convertir une DataFrame en CSV

```python
@st.cache_resource
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")
```

- Convertit une DataFrame en fichier CSV encodé en UTF-8. Cette fonction est utile pour permettre aux utilisateurs de télécharger des données sous forme de fichier CSV.

- Fonction pour Afficher un Aperçu des Données

```python
def display_data_preview(title, dataframe, file_name="close_stock_prices.csv", key=0):
    data_exp = st.expander(title)
    available_cols = dataframe.columns.tolist()
    columns_to_show = data_exp.multiselect(
        "Columns", 
        available_cols, 
        default=available_cols,
        key=key
    )
    data_exp.dataframe(dataframe[columns_to_show])

    csv_file = convert_df_to_csv(dataframe[columns_to_show])
    data_exp.download_button(
        label="Download selected as CSV",
        data=csv_file,
        file_name=file_name,
        mime="text/csv",
    )    
```

- **`data_exp = st.expander(title)`** : Crée un expander (une section extensible) avec un titre.
- **`data_exp.multiselect`** : Affiche une liste de cases à cocher permettant aux utilisateurs de sélectionner les colonnes qu'ils souhaitent afficher.
- **`data_exp.dataframe`** : Affiche la DataFrame avec les colonnes sélectionnées.
- **`data_exp.download_button`** : Crée un bouton de téléchargement pour permettre aux utilisateurs de télécharger les données affichées sous forme de fichier CSV.

En utilisant ces fonctions, vous pouvez facilement récupérer, transformer et afficher des données financières dans une application Streamlit, tout en offrant aux utilisateurs des options pour explorer et télécharger ces données.


🖥 **Script TechnicalAnalysis.py qui définit l'interface graphique Streamlit de l'application**

```python
# imports
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import requests
from bs4 import BeautifulSoup

from utils import *

# set offline mode for cufflinks
cf.go_offline()

# sidebar

# inputs for downloading data
st.sidebar.header("Stock Parameters")

# Update available tickers based on market index selection
market_index = st.sidebar.selectbox(
    "Market Index", 
    ["S&P500", "CAC40", "DAX", "FTSE100", "Nikkei225"]
)

if market_index == "S&P500":
    available_tickers, tickers_companies_dict = get_sp500_components()
elif market_index == "CAC40":
    available_tickers, tickers_companies_dict = get_cac40_components()
elif market_index == "DAX":
    available_tickers, tickers_companies_dict = get_dax_components()
elif market_index == "FTSE100":
    available_tickers, tickers_companies_dict = get_ftse_components()
elif market_index == "Nikkei225":
    available_tickers, tickers_companies_dict = get_nikkei_components()

# available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(
    "Ticker", 
    available_tickers, 
    format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
    "Start date", 
    datetime.date(2022, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

# inputs for technical analysis
st.sidebar.header("Technical Analysis Parameters")

volume_flag = st.sidebar.checkbox(label="Add volume")

exp_sma = st.sidebar.expander("SMA")
sma_flag = exp_sma.checkbox(label="Add SMA")
sma_periods= exp_sma.number_input(
    label="SMA Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)

exp_bb = st.sidebar.expander("Bollinger Bands")
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands")
bb_periods= exp_bb.number_input(label="BB Periods", 
                                min_value=1, max_value=50, 
                                value=20, step=1)
bb_std= exp_bb.number_input(label="# of standard deviations", 
                            min_value=1, max_value=4, 
                            value=2, step=1)

exp_rsi = st.sidebar.expander("Relative Strength Index")
rsi_flag = exp_rsi.checkbox(label="Add RSI")
rsi_periods= exp_rsi.number_input(
    label="RSI Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)
rsi_upper= exp_rsi.number_input(label="RSI Upper", 
                                min_value=50, 
                                max_value=90, value=70, 
                                step=1)
rsi_lower= exp_rsi.number_input(label="RSI Lower", 
                                min_value=10, 
                                max_value=50, value=30, 
                                step=1)

# main body

st.title("Technical Analysis")

run_button = st.sidebar.button("Run Analysis")

if run_button:

    df = load_data(ticker, start_date, end_date)

    # data preview part
    display_data_preview("Preview data", df, file_name=f"{ticker}_stock_prices.csv", key=1)

    # technical analysis plot
    title_str = f"{tickers_companies_dict[ticker]}'s stock price"
    qf = cf.QuantFig(df, title=title_str)
    if volume_flag:
        qf.add_volume()
    if sma_flag:
        qf.add_sma(periods=sma_periods)
    if bb_flag:
        qf.add_bollinger_bands(periods=bb_periods,
                            boll_std=bb_std)
    if rsi_flag:
        qf.add_rsi(periods=rsi_periods,
                rsi_upper=rsi_upper,
                rsi_lower=rsi_lower,
                showbands=True)

    fig = qf.iplot(asFigure=True)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, height=500)
```

Le script `TechnicalAnalysis.py` utilise Streamlit pour créer une application Web interactive permettant d'effectuer une analyse technique sur des actifs boursiers. Voici une explication détaillée de chaque section et fonction pour que les débutants puissent comprendre facilement.

- Importations
Les importations initiales chargent les bibliothèques nécessaires.

```python
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import requests
from bs4 import BeautifulSoup
from utils import *
```

- **`yfinance`** : Bibliothèque pour télécharger des données financières.
- **`streamlit`** : Bibliothèque pour créer des applications Web interactives.
- **`datetime`** : Module pour travailler avec les dates.
- **`pandas`** : Bibliothèque pour la manipulation des données.
- **`cufflinks` et `plotly`** : Bibliothèques pour créer des graphiques interactifs.
- **`requests` et `BeautifulSoup`** : Bibliothèques pour récupérer et parser le contenu des pages Web.
- **`from utils import *`** : Importation de fonctions utilitaires définies dans le fichier `utils.py`.

- Configuration de Cufflinks
Cufflinks est configuré pour fonctionner en mode hors ligne.

```python
cf.go_offline()
```

- Barre Latérale
La barre latérale de l'application permet aux utilisateurs de sélectionner des paramètres.

    - Paramètres des Actions
Les utilisateurs peuvent choisir un indice boursier, un ticker, une date de début et une date de fin.

```python
st.sidebar.header("Stock Parameters")

market_index = st.sidebar.selectbox(
    "Market Index", 
    ["S&P500", "CAC40", "DAX", "FTSE100", "Nikkei225"]
)

if market_index == "S&P500":
    available_tickers, tickers_companies_dict = get_sp500_components()
elif market_index == "CAC40":
    available_tickers, tickers_companies_dict = get_cac40_components()
elif market_index == "DAX":
    available_tickers, tickers_companies_dict = get_dax_components()
elif market_index == "FTSE100":
    available_tickers, tickers_companies_dict = get_ftse_components()
elif market_index == "Nikkei225":
    available_tickers, tickers_companies_dict = get_nikkei_components()

ticker = st.sidebar.selectbox(
    "Ticker", 
    available_tickers, 
    format_func=tickers_companies_dict.get
)

start_date = st.sidebar.date_input(
    "Start date", 
    datetime.date(2022, 1, 1)
)

end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")
```

- **`st.sidebar.header("Stock Parameters")`** : Ajoute un en-tête dans la barre latérale pour les paramètres des actions.
- **`st.sidebar.selectbox`** : Crée un menu déroulant pour sélectionner un indice boursier et un ticker.
- **`st.sidebar.date_input`** : Crée des sélecteurs de date pour choisir les dates de début et de fin.
- **`if start_date > end_date`** : Affiche une erreur si la date de début est après la date de fin.

    - Paramètres de l'Analyse Technique
Les utilisateurs peuvent sélectionner des paramètres pour différents indicateurs techniques.

```python
st.sidebar.header("Technical Analysis Parameters")

volume_flag = st.sidebar.checkbox(label="Add volume")

exp_sma = st.sidebar.expander("SMA")
sma_flag = exp_sma.checkbox(label="Add SMA")
sma_periods= exp_sma.number_input(
    label="SMA Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)

exp_bb = st.sidebar.expander("Bollinger Bands")
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands")
bb_periods= exp_bb.number_input(label="BB Periods", 
                                min_value=1, max_value=50, 
                                value=20, step=1)
bb_std= exp_bb.number_input(label="# of standard deviations", 


                            min_value=1, max_value=4, 
                            value=2, step=1)

exp_rsi = st.sidebar.expander("Relative Strength Index")
rsi_flag = exp_rsi.checkbox(label="Add RSI")
rsi_periods= exp_rsi.number_input(
    label="RSI Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)
rsi_upper= exp_rsi.number_input(label="RSI Upper", 
                                min_value=50, 
                                max_value=90, value=70, 
                                step=1)
rsi_lower= exp_rsi.number_input(label="RSI Lower", 
                                min_value=10, 
                                max_value=50, value=30, 
                                step=1)
```

- **`st.sidebar.header("Technical Analysis Parameters")`** : Ajoute un en-tête dans la barre latérale pour les paramètres de l'analyse technique.
- **`st.sidebar.checkbox`** : Crée une case à cocher pour activer ou désactiver l'affichage du volume des transactions.
- **`st.sidebar.expander`** : Crée une section extensible pour les paramètres d'un indicateur technique spécifique (par exemple, SMA, Bollinger Bands, RSI).
- **`exp_sma.checkbox` et `exp_bb.checkbox` et `exp_rsi.checkbox`** : Créent des cases à cocher pour activer ou désactiver les indicateurs SMA, Bollinger Bands et RSI.
- **`exp_sma.number_input`, `exp_bb.number_input`, `exp_rsi.number_input`** : Créent des champs pour saisir les périodes et autres paramètres des indicateurs.

- Corps Principal
Le corps principal de l'application affiche le titre, exécute l'analyse technique lorsque l'utilisateur clique sur le bouton "Run Analysis", et affiche les résultats.

```python
st.title("Technical Analysis")

run_button = st.sidebar.button("Run Analysis")

if run_button:
    df = load_data(ticker, start_date, end_date)

    ## data preview part
    display_data_preview("Preview data", df, file_name=f"{ticker}_stock_prices.csv", key=1)

    ## technical analysis plot
    title_str = f"{tickers_companies_dict[ticker]}'s stock price"
    qf = cf.QuantFig(df, title=title_str)
    if volume_flag:
        qf.add_volume()
    if sma_flag:
        qf.add_sma(periods=sma_periods)
    if bb_flag:
        qf.add_bollinger_bands(periods=bb_periods,
                            boll_std=bb_std)
    if rsi_flag:
        qf.add_rsi(periods=rsi_periods,
                rsi_upper=rsi_upper,
                rsi_lower=rsi_lower,
                showbands=True)

    fig = qf.iplot(asFigure=True)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, height=500)
```

- **`st.title("Technical Analysis")`** : Affiche le titre de l'application.
- **`run_button = st.sidebar.button("Run Analysis")`** : Crée un bouton dans la barre latérale pour lancer l'analyse.
- **`if run_button:`** : Vérifie si le bouton a été cliqué.
    - **`df = load_data(ticker, start_date, end_date)`** : Charge les données financières pour le ticker sélectionné sur la période spécifiée.
    - **`display_data_preview("Preview data", df, file_name=f"{ticker}_stock_prices.csv", key=1)`** : Affiche un aperçu des données chargées avec une option pour télécharger le fichier CSV.
    - **`title_str = f"{tickers_companies_dict[ticker]}'s stock price"`** : Crée un titre pour le graphique.
    - **`qf = cf.QuantFig(df, title=title_str)`** : Initialise un objet QuantFig avec les données financières.
    - **`qf.add_volume()`** : Ajoute le volume des transactions au graphique si l'option est activée.
    - **`qf.add_sma(periods=sma_periods)`** : Ajoute la moyenne mobile simple (SMA) au graphique si l'option est activée.
    - **`qf.add_bollinger_bands(periods=bb_periods, boll_std=bb_std)`** : Ajoute les bandes de Bollinger au graphique si l'option est activée.
    - **`qf.add_rsi(periods=rsi_periods, rsi_upper=rsi_upper, rsi_lower=rsi_lower, showbands=True)`** : Ajoute l'indice de force relative (RSI) au graphique si l'option est activée.
    - **`fig = qf.iplot(asFigure=True)`** : Génère le graphique interactif.
    - **`fig.update_layout(height=500)`** : Définit la hauteur du graphique.
    - **`st.plotly_chart(fig, use_container_width=True, height=500)`** : Affiche le graphique dans l'application Streamlit.

En résumé, ce script permet de créer une application Web interactive pour effectuer une analyse technique sur des actifs boursiers, avec des options pour sélectionner différents indices boursiers, paramètres d'indicateurs techniques, et afficher les résultats sous forme de graphiques interactifs.