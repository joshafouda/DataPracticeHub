# Projet : Prévision des Prix de Clôture des Actions avec Prophet

## Description

Ce projet a pour objectif de prévoir les prix de clôture des actions en utilisant le modèle [Prophet](https://facebook.github.io/prophet/). L'application permet aux utilisateurs de sélectionner une action parmi les principaux indices de marché (S&P 500, CAC 40, DAX, FTSE 100 et Nikkei 225) et de définir les paramètres pour la prévision. Le modèle de prévision est construit en utilisant la bibliothèque Prophet, qui est particulièrement efficace pour les données de séries temporelles avec des observations quotidiennes et des effets saisonniers potentiels.

## Image
imgs/project9/project9.png

## Instructions

- Les utilisateurs doivent pouvoir interagir avec l'application pour :

1. Sélectionner un ticker d'action d'un indice de marché spécifié.

2. Définir une plage de dates pour les données historiques à utiliser.

3. Configurer les paramètres pour le modèle Prophet, y compris le pourcentage de données utilisé pour les tests, la plage de points de changement et les jours fériés du pays.

4. Visualiser les prix de clôture réels et prévus ainsi que les points de changement identifiés par le modèle.

5. Télécharger le modèle Prophet entraîné pour une analyse ou un déploiement ultérieur.

- Assurez-vous d'avoir Python 3 installé ainsi que les bibliothèques suivantes :
- `streamlit`
- `pandas`
- `yfinance`
- `cufflinks`
- `beautifulsoup4`
- `plotly`
- `scikit-learn`
- `prophet`

Vous pouvez installer les bibliothèques nécessaires en utilisant la commande suivante :

```sh
pip install streamlit pandas yfinance cufflinks beautifulsoup4 plotly scikit-learn prophet
```

- Étapes pour Réaliser le Projet

1. **Créer un Script Python :**
   - Créez un nouveau fichier script Python, par exemple, `forecasting_app.py`.

2. **Importer les Bibliothèques Nécessaires :**
   Assurez-vous d'importer toutes les bibliothèques requises au début de votre script

3. **Créer des Fonctions Utilitaires :**
   - Implémentez des fonctions auxiliaires pour charger les données et afficher les aperçus. Assurez-vous de placer ces fonctions dans un fichier séparé, `utils.py`, et de les importer dans votre script principal :
   ```python
   from utils import *
   ```

4. **Configurer l'Interface Streamlit :**
   - Configurez l'interface Streamlit pour les entrées utilisateur, y compris les options de la barre latérale pour sélectionner l'action, la plage de dates et les paramètres du modèle Prophet.

5. **Charger et Afficher les Données :**
   - Récupérez les données boursières de Yahoo Finance en fonction des entrées utilisateur et affichez un aperçu des données.

6. **Entraînement et Prévision avec le Modèle Prophet :**
   - Entraînez le modèle Prophet sur les données historiques, effectuez des prévisions et visualisez les résultats. Incluez des graphiques pour les prix réels et prévus, les points de changement et les prévisions futures.

7. **Option de Téléchargement du Modèle :**
   - Fournissez une option permettant aux utilisateurs de télécharger le modèle Prophet entraîné.

8. **Lancer l'Application :**
   - Lancez l'application Streamlit en utilisant la commande suivante :
   ```sh
   streamlit run forecasting_app.py
   ```

## Resources

- [Composants du S&P500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- [Composants du DAX](https://en.wikipedia.org/wiki/DAX)
- [Composants du NIKKEI225](https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/)
- [Composants du FTSE](https://en.wikipedia.org/wiki/FTSE_100_Index)
- [Composants du CAC40](https://en.wikipedia.org/wiki/CAC_40)
- [Lien de l'application à reproduire](https://financeapp-rfsbok6xwjgezg6tztvfiy.streamlit.app/Forecasting)
- [Prévision du prix du Bitcoin grâce au Machine Learning avec Python et Meta Prophet](https://youtu.be/VTPCC1dggcs)
- [PREVISIONS DE SERIES TEMPORELLES AVEC FACEBOOK PROPHET EN MOINS DE 10 MINUTES](https://youtu.be/d_Yw-tifB_I)
- [Vidéo présentant l'application à reproduire](https://youtu.be/nPVBVCM-Rv8?si=-cyMJ7ysJuycxlU8)
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

Pour réaliser ce projet, vous aurez besoin d'installer les packages suivants dans votre environnement de développement Python :

- `streamlit`
- `pandas`
- `yfinance`
- `cufflinks`
- `beautifulsoup4`
- `plotly`
- `scikit-learn`
- `prophet`


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


🖥 **Script forecasting_app.py qui définit l'interface graphique Streamlit de l'application**

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
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly
import plotly.express as px
from prophet.serialize import model_to_json

from utils import *

st.title("Forecasting Close Price")

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
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

# inputs for technical analysis
st.sidebar.header("Forecasting Process")

exp_prophet = st.sidebar.expander("Prophet Parameters")
test_data_percentage = exp_prophet.number_input("Testing Data Percentage", 0.1, 0.4, 0.2, 0.05)
changepoint_range = exp_prophet.number_input("Changepoint Range", 0.05, 0.95, 0.9, 0.05)
country_holidays = exp_prophet.selectbox("Country Holidays", ['US', 'FR', 'DE', 'JP', 'GB'])
horizon = exp_prophet.number_input("Forecast Horizon (days)", min_value=1, value=365, step=1)
download_prophet = exp_prophet.checkbox(label="Download Model")

#st.subheader("Modeling Process")
modeling_option = st.sidebar.radio("Select Modeling Process", ["Prophet"])


# main body

run_button = st.sidebar.button("Run Forecasting")

if run_button:

    df = load_data(ticker, start_date, end_date)
    #df.dropna(inplace=True)

    # data preview part
    display_data_preview("Preview data", df, key=2)

    # plot close price
    close_plot = st.expander("Close Price Chart")
    # Plot the close price data
    fig = go.Figure()
    title_str = f"{tickers_companies_dict[ticker]}'s Close Price"
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=title_str,
                      xaxis_title='Date',
                      yaxis_title='Close Price ($)')
    st.plotly_chart(fig)

    if modeling_option == "Prophet":

        # Modeling process with Prophet
        st.write("Running Prophet Modeling Process...")
        # ... Add your Prophet modeling code here ...
        df = df[['Close']]
        df = df.reset_index(drop=False)
        df.columns = ['ds', 'y']

        # Sequential train/test split with 80% for training and 20% for testing
        df_train, df_test = train_test_split(df, 
                                             test_size=test_data_percentage, 
                                             shuffle=False, 
                                             random_state=42)

        # Create and fit the model
        prophet = Prophet(changepoint_range=changepoint_range)
        prophet.add_country_holidays(country_name=country_holidays)
        #prophet.add_seasonality(name="annual", period=365, fourier_order=5)
        prophet.fit(df_train)

        # Predictions on test data
        df_future = prophet.make_future_dataframe(
            periods=len(df_test),
            freq="B" # Business Days
        )
        df_pred = prophet.predict(df_future)

        # Prediction preview part
        display_data_preview("Prediction Data", df_pred, file_name=f"{ticker}_pred_data.csv", key=3)

        # Plot the results
        fig = plot_plotly(prophet, df_pred)
        st.plotly_chart(fig)

        # Changepoints

        # Create a dataframe with the changepoints
        changepoints = pd.DataFrame(prophet.changepoints)
        display_data_preview("Changepoints Data", 
                             changepoints, 
                             file_name=f"{ticker}_changepoints.csv",
                             key=4)

        # Create a Plotly figure
        fig = go.Figure()

        # Add a line for the actual data
        fig.add_trace(go.Scatter(
            x=df_pred['ds'],
            y=df_pred['yhat'],
            mode='lines',
            name='Actual Close Price'
        ))

        # Add scatter points for changepoints
        fig.add_trace(go.Scatter(
            x=changepoints['ds'],
            y=df_pred[df_pred['ds'].isin(changepoints['ds'])]['yhat'],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
            ),
            name='Changepoints'
        ))

        # Update the layout for better visualization
        fig.update_layout(
            title=f"{ticker} Price - Actual vs. Changepoints",
            xaxis_title="Date",
            yaxis_title=f"{ticker} Price ($)",
        )

        # Show the interactive Plotly chart
        st.plotly_chart(fig)

        
        # Affichage des prix vs predictions dans un même graphique
        # merge the test values with the forecasts
        SELECTED_COLS = [
            "ds", "yhat", "yhat_lower", "yhat_upper"
        ]

        df_pred = (
            df_pred
            .loc[:, SELECTED_COLS]
            .reset_index(drop=True)
        )
        df_test = df_test.merge(df_pred, on=["ds"], how="left")
        df_test["ds"] = pd.to_datetime(df_test["ds"])
        btc_test = df_test.set_index("ds")

        # Create a figure for the interactive chart
        fig = go.Figure()

        # Plot the actual values ('y')
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['y'], mode='lines', name='Actual', line=dict(color='blue')))

        # Plot the predicted values ('yhat')
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat'], mode='lines', name='Predicted', line=dict(color='orange')))

        # Fill the region between the lower and upper bounds
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, fill='tonexty', fillcolor='rgba(255,165,0,0.3)', name='Uncertainty'))

        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.3)', showlegend=False))

        # Customize the layout
        fig.update_layout(
            title=f"{tickers_companies_dict[ticker]}'s Close Price - Actual vs. Predicted",
            xaxis_title="Date",
            yaxis_title=f"{tickers_companies_dict[ticker]}'s Close Price ($)",
        )

        # Show the interactive chart
        st.plotly_chart(fig)

        # -------------------------------Pevisions (in futur dates) ----------------------------------------------------

        # Create and fit the model
        new_prophet = Prophet(
            changepoint_range=changepoint_range
        )
        new_prophet.add_country_holidays(country_name=country_holidays)
        #prophet.add_seasonality(name="annual", 
                                #period=365, 
                                #fourier_order=5)
        new_prophet.fit(df)

        # Forecasts (in the future)
        future = new_prophet.make_future_dataframe(
            periods=horizon,
            freq="B" # Business Days
        )
        forecasts = new_prophet.predict(future)

        last_date = df['ds'].max()
        forecasts = forecasts[forecasts['ds'] > last_date]

        # Prediction preview part
        display_data_preview("Forecasts Data", forecasts, file_name=f"{ticker}_forecasts_data.csv", key=5)

        # Create a Plotly figure to display the actual and forecasted prices
        fig = go.Figure()

        # Plot the actual prices
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Price'))

        # Plot the forecasted prices
        fig.add_trace(go.Scatter(x=forecasts['ds'], y=forecasts['yhat'], mode='lines', name='Forecasted Price', line=dict(color='red')))

        # Customize the layout
        fig.update_layout(
            title=f"{ticker} Price - Actual vs. Forecasted",
            xaxis_title="Date",
            yaxis_title=f"{ticker} Price",
        )

        # Display the interactive chart using Streamlit
        st.plotly_chart(fig)


        # Download the Model
        if download_prophet:
            with open('serialized_model.json', 'w') as fout:
                fout.write(model_to_json(new_prophet))
            st.success("Prophet Model downloaded successfully as 'serialized_prophet_model.json'")
```



Ce code crée une application web interactive pour prévoir les prix de clôture des actions à l'aide de la bibliothèque `Prophet` de Facebook. Voici une explication détaillée, ligne par ligne, pour aider toute personne à comprendre le fonctionnement du code :

- Importations et configuration initiale

```python
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly
import plotly.express as px
from prophet.serialize import model_to_json

from utils import *
```

1. **Importations des bibliothèques** : Ces lignes importent les bibliothèques nécessaires pour le téléchargement des données boursières (`yfinance`), la création d'une application web (`streamlit`), la manipulation des dates (`datetime`), la gestion des données (`pandas`), la visualisation des données (`cufflinks` et `plotly`), le web scraping (`requests` et `BeautifulSoup`), la modélisation et la prévision (`Prophet`), et les outils auxiliaires (`utils`).

- Titre de l'application

```python
st.title("Forecasting Close Price")
```

2. **Titre de l'application** : Affiche le titre de l'application web sur l'interface utilisateur.

- Configuration de la barre latérale

```python
st.sidebar.header("Stock Parameters")
```

3. **En-tête des paramètres des actions** : Affiche une en-tête dans la barre latérale pour les paramètres des actions.

```python
market_index = st.sidebar.selectbox(
    "Market Index", 
    ["S&P500", "CAC40", "DAX", "FTSE100", "Nikkei225"]
)
```

4. **Sélection de l'indice de marché** : Permet à l'utilisateur de sélectionner un indice de marché parmi une liste prédéfinie.

- Récupération des tickers

```python
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
```

5. **Récupération des composants de l'indice** : Selon l'indice sélectionné, appelle une fonction spécifique pour obtenir les tickers des actions et les noms des entreprises correspondantes.

```python
ticker = st.sidebar.selectbox(
    "Ticker", 
    available_tickers, 
    format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
    "Start date", 
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)
```

6. **Sélection du ticker et des dates** : Permet à l'utilisateur de sélectionner un ticker parmi les tickers disponibles et de spécifier une date de début et une date de fin pour les données à télécharger.

- Validation des dates

```python
if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")
```

7. **Validation des dates** : Affiche un message d'erreur si la date de début est postérieure à la date de fin.

- Paramètres de la prévision avec Prophet

```python
st.sidebar.header("Forecasting Process")

exp_prophet = st.sidebar.expander("Prophet Parameters")
test_data_percentage = exp_prophet.number_input("Testing Data Percentage", 0.1, 0.4, 0.2, 0.05)
changepoint_range = exp_prophet.number_input("Changepoint Range", 0.05, 0.95, 0.9, 0.05)
country_holidays = exp_prophet.selectbox("Country Holidays", ['US', 'FR', 'DE', 'JP', 'GB'])
horizon = exp_prophet.number_input("Forecast Horizon (days)", min_value=1, value=365, step=1)
download_prophet = exp_prophet.checkbox(label="Download Model")
```

8. **Paramètres de la prévision** : Permet à l'utilisateur de spécifier les paramètres pour le modèle Prophet, tels que le pourcentage de données de test, la plage de changement, les vacances par pays, l'horizon de prévision et la possibilité de télécharger le modèle.

- Bouton pour lancer la prévision

```python
run_button = st.sidebar.button("Run Forecasting")
```

9. **Bouton pour lancer la prévision** : Ajoute un bouton pour exécuter le processus de prévision.

- Exécution du processus de prévision

```python
if run_button:

    df = load_data(ticker, start_date, end_date)
    display_data_preview("Preview data", df, key=2)
```

10. **Téléchargement des données** : Si le bouton est cliqué, télécharge les données du ticker sélectionné entre les dates spécifiées et affiche un aperçu des données.

```python
    close_plot = st.expander("Close Price Chart")
    fig = go.Figure()
    title_str = f"{tickers_companies_dict[ticker]}'s Close Price"
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=title_str,
                      xaxis_title='Date',
                      yaxis_title='Close Price ($)')
    st.plotly_chart(fig)
```

11. **Affichage du graphique des prix de clôture** : Crée un graphique interactif des prix de clôture des actions et l'affiche sur l'interface utilisateur.

- Modélisation avec Prophet

```python
    if modeling_option == "Prophet":

        st.write("Running Prophet Modeling Process...")
        df = df[['Close']]
        df = df.reset_index(drop=False)
        df.columns = ['ds', 'y']
```

12. **Préparation des données pour Prophet** : Filtre les données pour ne conserver que les colonnes nécessaires (date et prix de clôture) et renomme les colonnes pour qu'elles soient compatibles avec Prophet.

```python
        df_train, df_test = train_test_split(df, 
                                             test_size=test_data_percentage, 
                                             shuffle=False, 
                                             random_state=42)
```

13. **Division des données** : Divise les données en un ensemble d'entraînement et un ensemble de test, sans mélanger les données pour respecter l'ordre chronologique.

```python
        prophet = Prophet(changepoint_range=changepoint_range)
        prophet.add_country_holidays(country_name=country_holidays)
        prophet.fit(df_train)
```

14. **Création et entraînement du modèle Prophet** : Crée un modèle Prophet en utilisant les paramètres spécifiés et l'entraîne sur les données d'entraînement.

```python
        df_future = prophet.make_future_dataframe(
            periods=len(df_test),
            freq="B" # Business Days
        )
        df_pred = prophet.predict(df_future)
```

15. **Prédictions sur les données de test** : Génère des prévisions pour les dates de l'ensemble de test.

```python
        display_data_preview("Prediction Data", df_pred, file_name=f"{ticker}_pred_data.csv", key=3)
```

16. **Affichage des prévisions** : Affiche les données prédites et permet de les télécharger.

- Affichage des résultats de la prévision

```python
        fig = plot_plotly(prophet, df_pred)
        st.plotly_chart(fig)
```

17. **Graphique des prévisions** : Crée et affiche un graphique interactif des prévisions.

- Changepoints

```python
        changepoints = pd.DataFrame(prophet.changepoints)
        display_data_preview("Changepoints Data", 
                             changepoints, 
                             file_name=f"{ticker}_changepoints.csv",
                             key=4)
```

18. **Affichage des points de changement** : Crée un DataFrame des points de changement et permet de les télécharger.

```python
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_pred['ds'],
            y=df_pred['yhat'],
            mode='lines',
            name='Actual Close Price'
        ))
        fig.add_trace(go.Scatter(
            x=changepoints['ds'],
            y=df_pred[df_pred['ds'].isin(changepoints['ds'])]['yhat'],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
            ),
            name='Changepoints'
        ))
        fig.update_layout(
            title=f"{ticker} Price - Actual vs. Changepoints",
            xaxis_title="Date",
            yaxis_title=f"{ticker} Price ($)",
        )
        st.plotly_chart(fig)
```

19. **Graphique des points de changement** : Crée et affiche un graphique montrant les prix réels et les points de changement détectés.

- Comparaison des prix réels et prédits

```python
        df_pred = (


            df_pred
            .loc[:, SELECTED_COLS]
            .reset_index(drop=True)
        )
        df_test = df_test.merge(df_pred, on=["ds"], how="left")
        df_test["ds"] = pd.to_datetime(df_test["ds"])
        btc_test = df_test.set_index("ds")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['y'], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat'], mode='lines', name='Predicted', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, fill='tonexty', fillcolor='rgba(255,165,0,0.3)', name='Uncertainty'))
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.3)', showlegend=False))

        fig.update_layout(
            title=f"{tickers_companies_dict[ticker]}'s Close Price - Actual vs. Predicted",
            xaxis_title="Date",
            yaxis_title=f"{tickers_companies_dict[ticker]}'s Close Price ($)",
        )
        st.plotly_chart(fig)
```

20. **Affichage des prix réels et prédits** : Crée et affiche un graphique comparant les prix réels aux prix prédits, avec une indication des incertitudes.

- Prévisions futures

```python
        new_prophet = Prophet(
            changepoint_range=changepoint_range
        )
        new_prophet.add_country_holidays(country_name=country_holidays)
        new_prophet.fit(df)
        future = new_prophet.make_future_dataframe(
            periods=horizon,
            freq="B"
        )
        forecasts = new_prophet.predict(future)

        last_date = df['ds'].max()
        forecasts = forecasts[forecasts['ds'] > last_date]

        display_data_preview("Forecasts Data", forecasts, file_name=f"{ticker}_forecasts_data.csv", key=5)
```

21. **Prévisions futures** : Crée un nouveau modèle Prophet pour prévoir les prix futurs sur une période spécifiée par l'utilisateur.

```python
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=forecasts['ds'], y=forecasts['yhat'], mode='lines', name='Forecasted Price', line=dict(color='red')))
        fig.update_layout(
            title=f"{ticker} Price - Actual vs. Forecasted",
            xaxis_title="Date",
            yaxis_title=f"{ticker} Price",
        )
        st.plotly_chart(fig)
```

22. **Graphique des prévisions futures** : Crée et affiche un graphique des prix réels et des prix prévus pour une période future.

- Téléchargement du modèle

```python
        if download_prophet:
            with open('serialized_model.json', 'w') as fout:
                fout.write(model_to_json(new_prophet))
            st.success("Prophet Model downloaded successfully as 'serialized_prophet_model.json'")
```

23. **Téléchargement du modèle** : Permet à l'utilisateur de télécharger le modèle Prophet entraîné en tant que fichier JSON.

- Résumé

Ce code est une application web interactive construite avec Streamlit qui permet aux utilisateurs de prévoir les prix de clôture des actions à l'aide du modèle Prophet. L'utilisateur peut sélectionner un indice de marché, un ticker, des dates, et des paramètres de prévision. L'application télécharge les données, les visualise, entraîne un modèle Prophet, affiche les prévisions et permet de télécharger le modèle.