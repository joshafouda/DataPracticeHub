# Projet : Allocation d'Actifs avec Python pour l'Optimisation de Portefeuille

## Description

Ce projet consiste à construire une application interactive d'allocation d'actifs utilisant Streamlit. L'application permet aux utilisateurs de sélectionner des actifs financiers provenant de divers indices boursiers, de spécifier des dates de début et de fin pour l'analyse, et de choisir différentes méthodes d'allocation d'actifs pour optimiser leur portefeuille. Les utilisateurs peuvent visualiser les prix ajustés, les rendements journaliers, et les frontières efficientes calculées avec différentes techniques d'optimisation.

## Image
imgs/project10/project10.png

## Instructions

- Prérequis :

    - Connaissances de base en Python

    - Familiarité avec les bibliothèques de données et de visualisation telles que Pandas, NumPy, et Plotly

    - Une compréhension des concepts de base de la finance, tels que les rendements et la volatilité des actifs financiers, est un plus

- Étapes à Suivre

    1. Installer les Dépendances
    Assurez-vous que Python est installé sur votre machine. Ensuite, installez les bibliothèques nécessaires en utilisant pip:
    ```bash
    pip install streamlit numpy pandas scipy yfinance cufflinks plotly seaborn matplotlib cvxpy
    ```

    2. Créer la Structure du Projet
    Créez un dossier pour votre projet et organisez les fichiers comme suit :

    ```plaintext
    project_folder/
    |-- app.py
    |-- utils.py
    |-- requirements.txt
    ```

    3. Implémenter les Fonctions Utilitaires

    Dans le fichier `utils.py`, définissez les fonctions nécessaires pour charger les données, calculer les statistiques, et effectuer les différentes optimisations. 

    Ajoutez d'autres fonctions pour les différentes méthodes d'optimisation que vous connaissez

    4. Créer l'Interface Streamlit

    Dans le fichier `app.py`, commencez par importer les bibliothèques nécessaires.

    5. Sélectionner les Actifs et les Dates

    Ajoutez des widgets pour que l'utilisateur puisse sélectionner les indices de marché, les actifs, et les dates de début et de fin.

    Ajoutez le chargement des tickers basés sur l'indice sélectionné et affichez-les avec un widget multiselect.

    6. Charger et Afficher les Données

    Chargez les données des actifs sélectionnés et affichez les prix ajustés et les rendements journaliers.

    7. Implémenter les Méthodes d'Optimisation

    Ajoutez les différentes méthodes d'optimisation, comme les simulations Monte Carlo, l'optimisation avec SciPy et CVXPY, et la parité des risques hiérarchique. 
    
    Créez des boutons pour exécuter l'optimisation choisie et afficher les résultats.

    8. Visualiser les Résultats

    Affichez les résultats de l'optimisation, y compris les graphiques des poids des actifs, les frontières efficientes, et les résumés de performance.

    9. Exécuter l'Application

    Dans le terminal, exécutez l'application Streamlit :

    ```bash
    streamlit run app.py
    ```
    Votre application devrait maintenant être accessible dans votre navigateur, vous permettant de sélectionner des actifs, de choisir une méthode d'optimisation, et de visualiser les résultats de votre portefeuille optimisé.

- Conclusion

En suivant ces étapes, vous serez en mesure de créer une application d'allocation d'actifs interactive et visuellement riche. Cela vous permettra d'explorer différentes techniques d'optimisation de portefeuille et de comprendre comment elles affectent la répartition des actifs et la performance du portefeuille.


## Resources

- [Composants du S&P500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- [Composants du DAX](https://en.wikipedia.org/wiki/DAX)
- [Composants du NIKKEI225](https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/)
- [Composants du FTSE](https://en.wikipedia.org/wiki/FTSE_100_Index)
- [Composants du CAC40](https://en.wikipedia.org/wiki/CAC_40)
- [Lien de l'application à reproduire](https://financeapp-rfsbok6xwjgezg6tztvfiy.streamlit.app/AssetAllocation)
- [Vidéo présentant l'application à reproduire](https://youtu.be/2w_BncUhaUs)
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

🖥 **Script utils.py qui comporte les fonctions utiles à l'application**

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import cvxpy as cp
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


# Function to calculate annualized average returns and covariance matrix
def calculate_statistics(prices_df, n_days=252):
    returns_df = prices_df.pct_change().dropna()
    avg_returns = returns_df.mean() * n_days
    cov_mat = returns_df.cov() * n_days
    return avg_returns, cov_mat

# Function to generate unique markers based on the number of assets
def generate_markers(n_assets):
    marker_pool = ["o", "X", "d", "*", "^", "s"]  # Add more markers if needed
    return marker_pool[:n_assets]


def print_portfolio_summary(perf, weights, assets, name):
    """
    Helper function for printing the performance summary of a portfolio.

    Args:
        perf (pd.Series): Series containing the perf metrics
        weights (np.array): An array containing the portfolio weights
        assets (list): list of the asset names
        name (str): the name of the portfolio
    """
    name_portf = f"{name} portfolio Performance: ------------------"
    st.write(name_portf)
    for index, value in perf.items():
        st.write(f"{index}: {100 * value:.2f}% ", end="", flush=True)
    st.write("\nWeights")
    for x, y in zip(assets, weights):
        st.write(f"{x}: {100*y:.2f}% ", end="", flush=True)



# functions for calculating portfolio returns and volatility
def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

# Function to calculate efficient frontier using SciPy optimization
def get_efficient_frontier_scipy(avg_returns, cov_mat, rtns_range):
        efficient_portfolios_scipy = []

        n_assets = len(avg_returns)
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = n_assets * [1. / n_assets, ]

        for ret in rtns_range:
            constr = (
                {"type": "eq",
                 "fun": lambda x: get_portf_rtn(x, avg_returns) - ret},
                {"type": "eq",
                 "fun": lambda x: np.sum(x) - 1}
            )
            ef_portf_scipy = sco.minimize(get_portf_vol,
                                          initial_guess,
                                          args=(avg_returns, cov_mat),
                                          method="SLSQP",
                                          constraints=constr,
                                          bounds=bounds)
            efficient_portfolios_scipy.append(ef_portf_scipy)

        return efficient_portfolios_scipy


def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (
        (portf_returns - rf_rate) / portf_volatility
    )
    return -portf_sharpe_ratio
    
```

Voici une explication détaillée de chaque fonction dans le script `utils.py` :

- Imports

Ces importations ajoutent les bibliothèques nécessaires pour le projet. Voici à quoi elles servent :
- `streamlit` : créer des applications web interactives.
- `yfinance` : télécharger des données financières de Yahoo Finance.
- `pandas` : manipuler des données tabulaires.
- `numpy` : manipuler des tableaux et effectuer des calculs numériques.
- `scipy.optimize` : optimisation, utilisée pour optimiser les portefeuilles.
- `cvxpy` : optimisation convexe, également pour l'optimisation de portefeuilles.
- `requests` et `BeautifulSoup` : récupérer et analyser des données web (parsing HTML).
- `seaborn` et `matplotlib.pyplot` : visualiser les données.

- Fonctions pour obtenir les composants d'un indice de marché

    - `get_sp500_components`

    Cette fonction récupère les composants du S&P 500 depuis Wikipédia. Elle :
    1. Télécharge la page HTML.
    2. Lit les tables HTML en DataFrame.
    3. Extrait les symboles de ticker et les noms des entreprises.
    4. Renvoie une liste de tickers et un dictionnaire associant chaque ticker à son entreprise.

    Le décorateur `@st.cache_resource` permet de mettre en cache les résultats pour optimiser les performances.

    - Fonctions similaires pour d'autres indices (`get_dax_components`, `get_nikkei_components`, `get_ftse_components`, `get_cac40_components`)

    Ces fonctions suivent le même principe que `get_sp500_components` mais pour d'autres indices (DAX, Nikkei, FTSE 100, CAC 40).

- `load_data`

Cette fonction télécharge les données financières des symboles de ticker spécifiés sur une période donnée en utilisant `yfinance`. Le décorateur `@st.cache_data` met en cache les données pour des performances optimisées.

- `convert_df_to_csv`

Cette fonction convertit un DataFrame en CSV encodé en UTF-8. Cela permet de télécharger les données sous forme de fichier CSV.

- `display_data_preview`

Cette fonction affiche un aperçu des données et permet de télécharger les colonnes sélectionnées sous forme de fichier CSV. Elle :
1. Crée un conteneur extensible (expander).
2. Affiche une liste de sélection de colonnes.
3. Affiche les données sélectionnées.
4. Offre un bouton pour télécharger les données sélectionnées en CSV.

- `calculate_statistics`

Cette fonction calcule les statistiques nécessaires pour l'optimisation de portefeuille :
1. Calcule les rendements quotidiens.
2. Calcule les rendements annuels moyens.
3. Calcule la matrice de covariance annuelle des rendements.
4. Renvoie les rendements moyens et la matrice de covariance.

- `generate_markers`

Cette fonction génère des marqueurs uniques pour un nombre donné d'actifs, utile pour les graphiques.

- `print_portfolio_summary`

Cette fonction affiche un résumé des performances d'un portefeuille. Elle :
1. Affiche les performances du portefeuille (rendement, volatilité, ratio de Sharpe).
2. Affiche les poids des actifs dans le portefeuille.

- `get_portf_rtn` et `get_portf_vol`

Ces fonctions calculent respectivement le rendement et la volatilité d'un portefeuille.

- `get_portf_rtn`

Calcule le rendement du portefeuille en multipliant les poids des actifs par leurs rendements moyens.

- `get_portf_vol`

Calcule la volatilité du portefeuille en utilisant la matrice de covariance des rendements et les poids des actifs.

- `get_efficient_frontier_scipy`

Cette fonction utilise SciPy pour calculer la frontière efficace. Elle :
1. Définit des contraintes (rendement cible et somme des poids égale à 1).
2. Minimise la volatilité pour chaque rendement cible.
3. Renvoie les portefeuilles efficaces.

- `neg_sharpe_ratio`

Cette fonction calcule le ratio de Sharpe négatif, utilisé pour l'optimisation. Le ratio de Sharpe est le rendement excédentaire (par rapport au taux sans risque) par unité de risque (volatilité). La fonction renvoie la valeur négative car les optimiseurs minimisent par défaut.


🖥 **Script app.py qui définit l'interface graphique Streamlit de l'application**

```python
# imports
import streamlit as st
import random
import numpy as np
import scipy.optimize as sco
import cvxpy as cp
import yfinance as yf
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *

st.title("Asset Allocation")

# set offline mode for cufflinks
cf.go_offline()


# Update available tickers based on market index selection
market_index = st.selectbox(
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

start_date = st.date_input(
    "Start date", 
    datetime.date(2019, 1, 1)
)
end_date = st.date_input(
    "End date", 
    datetime.date.today()
)

assets = st.multiselect(
    'Select the assets for the portfolio:', 
    available_tickers, 
    #default=random.sample(available_tickers, 3),
    format_func=tickers_companies_dict.get
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

# inputs for Asset Allocation analysis
st.sidebar.header("Asset Allocation Method")

allocation_method = st.sidebar.radio("Select Asset Allocation Technique", 
                                     ["Monte Carlo simulations", "Scipy optimization", "CVXPY optimization", "Hierarchical Risk Parity"])

if len(assets) > 0:
    assets_data = load_data(assets, start_date, end_date)['Adj Close']
else:
    st.write("Choose some assets to build your Portfolio")

run_allocation_button = st.button("Run Asset Allocation")

if run_allocation_button:
    
    # Plot stock prices
    fig = go.Figure()

    for asset in assets:
        fig.add_trace(go.Scatter(x=assets_data.index, y=assets_data[asset], mode='lines', name=tickers_companies_dict[asset]))

    fig.update_layout(title="Adjusted Close Prices of Selected Stocks",
                    xaxis_title="Date",
                    yaxis_title="Adjusted Close Price",
                    legend=dict(title="Stocks"))

    # Display the plot
    st.plotly_chart(fig)

    # Calculate and display daily returns
    avg_returns, cov_mat = calculate_statistics(assets_data)

    fig_returns = go.Figure()

    for asset in assets:
        fig_returns.add_trace(go.Scatter(x=assets_data.index, y=assets_data[asset].pct_change(), mode='lines', name=tickers_companies_dict[asset]))

    fig_returns.update_layout(title="Daily Returns of Selected Stocks",
                            xaxis_title="Date",
                            yaxis_title="Daily Returns",
                            legend=dict(title="Company Names"))

    # Display the plot
    st.plotly_chart(fig_returns)

    # Simulate random portfolio weights
    np.random.seed(42)
    N_PORTFOLIOS = 10 ** 5
    n_assets = len(assets)
    weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Calculate the portfolio metrics
    portf_rtns = np.dot(weights, avg_returns)
    portf_vol = []
    for i in range(0, len(weights)):
        vol = np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i])))
        portf_vol.append(vol)
    portf_vol = np.array(portf_vol)
    portf_sharpe_ratio = portf_rtns / portf_vol

    # Create a DataFrame containing all the data
    portf_results_df = pd.DataFrame(
        {"returns": portf_rtns,
        "volatility": portf_vol,
        "sharpe_ratio": portf_sharpe_ratio}
    )

    display_data_preview("Preview data", portf_results_df, key=6)
        

    if allocation_method == "Monte Carlo simulations":
        
        # Locate the points creating the Efficient Frontier
        N_POINTS = 100
        ef_rtn_list = []
        ef_vol_list = []

        possible_ef_rtns = np.linspace(
            portf_results_df["returns"].min(), 
            portf_results_df["returns"].max(), 
            N_POINTS
        )
        possible_ef_rtns = np.round(possible_ef_rtns, 2)    
        portf_rtns = np.round(portf_rtns, 2)

        for rtn in possible_ef_rtns:
            if rtn in portf_rtns:
                ef_rtn_list.append(rtn)
                matched_ind = np.where(portf_rtns == rtn)
                ef_vol_list.append(np.min(portf_vol[matched_ind]))

        # Create the Efficient Frontier plot using Matplotlib
        st.subheader("Efficient Frontier")
        fig_ef, ax_ef = plt.subplots(figsize=(10, 6))

        # Scatter plot for individual portfolios
        scatter = ax_ef.scatter(
            x=portf_results_df["volatility"],
            y=portf_results_df["returns"],
            c=portf_results_df["sharpe_ratio"],
            cmap="RdYlGn",
            edgecolors="black",
            marker="o",
            alpha=0.8,
        )
        ax_ef.set(xlabel="Volatility", ylabel="Expected Returns", title="Efficient Frontier")

        # Line plot for Efficient Frontier
        ax_ef.plot(ef_vol_list, ef_rtn_list, "b--")

        # Markers for individual assets
        MARKERS = generate_markers(n_assets)
        for asset_index in range(n_assets):
            ax_ef.scatter(
                x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
                y=avg_returns[asset_index],
                marker=MARKERS[asset_index],
                s=150,
                color="black",
                label=tickers_companies_dict[assets[asset_index]],
            )

        # Add colorbar
        cbar = fig_ef.colorbar(scatter)
        cbar.set_label("Sharpe Ratio")

        # Add legend
        ax_ef.legend()

        # Remove spines and tighten layout
        sns.despine()
        plt.tight_layout()

        # Display the Efficient Frontier chart
        st.pyplot(fig_ef)

        # Display the portfolio performance summary
        st.subheader("Portfolio Performance Summary")
        max_sharpe_ind = np.argmax(portf_results_df["sharpe_ratio"])
        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

        min_vol_ind = np.argmin(portf_results_df["volatility"])
        min_vol_portf = portf_results_df.loc[min_vol_ind]

        max_return_ind = np.argmax(portf_results_df["returns"])
        max_return_portf = portf_results_df.loc[max_return_ind]

        # Bar chart showing the calculated weight of each asset in the Maximum Sharpe Ratio portfolio
        # Maximum Sharpe Ratio Portfolio
        weight_chart_data = pd.DataFrame({"Assets": assets, "Weights": weights[max_sharpe_ind]}).sort_values(by="Weights", ascending=False)
        weight_chart = px.bar(weight_chart_data, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                              title="Asset Weights in Maximum Sharpe Ratio Portfolio", color="Assets")
        st.plotly_chart(weight_chart)
        print_portfolio_summary(perf=max_sharpe_portf, weights=weights[max_sharpe_ind], assets=assets, name="Maximum Sharpe Ratio")

        # Minimum Volatility Portfolio
        weight_chart_data2 = pd.DataFrame({"Assets": assets, "Weights": weights[min_vol_ind]}).sort_values(by="Weights", ascending=False)
        weight_chart2 = px.bar(weight_chart_data2, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                              title="Asset Weights in Minimum Volatility Portfolio", color="Assets")
        st.plotly_chart(weight_chart2)
        print_portfolio_summary(min_vol_portf, weights[min_vol_ind], assets, name="Minimum Volatility")

        # Maximum Return Portfolio
        weight_chart_data3 = pd.DataFrame({"Assets": assets, "Weights": weights[max_return_ind]}).sort_values(by="Weights", ascending=False)
        weight_chart3 = px.bar(weight_chart_data3, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                              title="Asset Weights in Maximum Return Portfolio", color="Assets")
        st.plotly_chart(weight_chart3)
        print_portfolio_summary(perf=max_return_portf, weights=weights[max_return_ind], assets=assets, name="Maximum Return")


    elif allocation_method == "Scipy optimization":

        rtns_range = np.linspace(-0.1, 0.55, 200)
        
        # Calculate the Efficient Frontier using SciPy optimization
        efficient_portfolios_scipy = get_efficient_frontier_scipy(avg_returns, cov_mat, rtns_range)

        # Extract the volatilities of the efficient portfolios
        vols_range_scipy = [x["fun"] for x in efficient_portfolios_scipy]

        # Plot the Efficient Frontier using SciPy optimization
        with sns.plotting_context("paper"):
            fig_scipy, ax_scipy = plt.subplots()
            portf_results_df.plot(kind="scatter", x="volatility",
                                y="returns", c="sharpe_ratio",
                                cmap="RdYlGn", edgecolors="black",
                                ax=ax_scipy)
            ax_scipy.plot(vols_range_scipy, rtns_range, "b--", linewidth=3)
            ax_scipy.set(xlabel="Volatility",
                        ylabel="Expected Returns",
                        title="Efficient Frontier - SciPy Optimization")

            sns.despine()
            plt.tight_layout()

        # Display the Efficient Frontier chart using SciPy optimization
        st.pyplot(fig_scipy)

        # Minimum Volatility Portfolio
        min_vol_ind_scipy = np.argmin(vols_range_scipy)
        min_vol_portf_rtn_scipy = rtns_range[min_vol_ind_scipy]
        min_vol_portf_vol_scipy = efficient_portfolios_scipy[min_vol_ind_scipy]["fun"]

        min_vol_portf_scipy = {
            "Return": min_vol_portf_rtn_scipy,
            "Volatility": min_vol_portf_vol_scipy,
            "Sharpe Ratio": (min_vol_portf_rtn_scipy / min_vol_portf_vol_scipy)
        }
        #st.write(min_vol_portf_scipy)
        weight_chart_data4 = pd.DataFrame({"Assets": assets, "Weights": efficient_portfolios_scipy[min_vol_ind_scipy]["x"]}).sort_values(by="Weights", ascending=False)
        weight_chart4 = px.bar(weight_chart_data4, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                              title="Asset Weights in Minimum Volatility Portfolio", color="Assets")
        st.plotly_chart(weight_chart4)
        print_portfolio_summary(min_vol_portf_scipy, 
                        efficient_portfolios_scipy[min_vol_ind_scipy]["x"], 
                        assets, 
                        name="Minimum Volatility")


        # Maximum Sharpe Ratio Portfolio
        n_assets = len(avg_returns)
        RF_RATE = 0

        args = (avg_returns, cov_mat, RF_RATE)
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for asset in range(n_assets))
        initial_guess = n_assets * [1. / n_assets]

        max_sharpe_portf_scipy = sco.minimize(neg_sharpe_ratio, 
                                        x0=initial_guess, 
                                        args=args,
                                        method="SLSQP", 
                                        bounds=bounds, 
                                        constraints=constraints)
        
        max_sharpe_portf_w_scipy = max_sharpe_portf_scipy["x"]
        max_sharpe_portf_scipy = {
            "Return": get_portf_rtn(max_sharpe_portf_w_scipy, avg_returns),
            "Volatility": get_portf_vol(max_sharpe_portf_w_scipy, 
                                        avg_returns, 
                                        cov_mat),
            "Sharpe Ratio": -max_sharpe_portf_scipy["fun"]
        }
        #st.write(max_sharpe_portf)
        weight_chart_data5 = pd.DataFrame({"Assets": assets, "Weights": max_sharpe_portf_w_scipy}).sort_values(by="Weights", ascending=False)
        weight_chart5 = px.bar(weight_chart_data5, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                              title="Asset Weights in Maximum Sharpe Ratio Portfolio", color="Assets")
        st.plotly_chart(weight_chart5)
        print_portfolio_summary(max_sharpe_portf_scipy, max_sharpe_portf_w_scipy, assets, name="Maximum Sharpe Ratio")

  

    elif allocation_method == "CVXPY optimization":
        avg_returns = avg_returns.values
        cov_mat = cov_mat.values

        # Set up the optimization problem
        weights = cp.Variable(n_assets)
        gamma_par = cp.Parameter(nonneg=True)
        portf_rtn_cvx = avg_returns @ weights 
        portf_vol_cvx = cp.quad_form(weights, cov_mat)
        objective_function = cp.Maximize(
            portf_rtn_cvx - gamma_par * portf_vol_cvx
        )
        problem = cp.Problem(
            objective_function, 
            [cp.sum(weights) == 1, weights >= 0]
        )

        # Calculate the Efficient Frontier
        N_POINTS = 25
        portf_rtn_cvx_ef = []
        portf_vol_cvx_ef = []
        weights_ef = []
        gamma_range = np.logspace(-3, 3, num=N_POINTS)

        for gamma in gamma_range:
            gamma_par.value = gamma
            problem.solve()
            portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
            portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
            weights_ef.append(weights.value)

        # Plot the Efficient Frontier, together with the individual assets
        fig_cvx, ax_cvx = plt.subplots()
        MARKERS = generate_markers(n_assets)
        ax_cvx.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-")
        for asset_index in range(n_assets):
            plt.scatter(x=np.sqrt(cov_mat[asset_index, asset_index]), 
                        y=avg_returns[asset_index], 
                        marker=MARKERS[asset_index], 
                        label=assets[asset_index],
                        s=150)
        ax_cvx.set(title="Efficient Frontier",
            xlabel="Volatility", 
            ylabel="Expected Returns")
        ax_cvx.legend()

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_cvx)


    elif allocation_method == "Hierarchical Risk Parity":
        st.write("----------------------------------------will be available soon-------------------------------------------")
        
```

Voici une explication détaillée du script `app.py` qui définit l'application Streamlit pour l'allocation d'actifs. Chaque section du code est expliquée de manière à ce qu'un débutant puisse comprendre.

- Importation des bibliothèques nécessaires

```python
import streamlit as st
import random
import numpy as np
import scipy.optimize as sco
import cvxpy as cp
import yfinance as yf
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *
```

Ces lignes importent les bibliothèques nécessaires. `streamlit` est utilisé pour créer l'application web, `numpy` pour les calculs numériques, `scipy.optimize` et `cvxpy` pour l'optimisation, `yfinance` pour récupérer les données boursières, `pandas` pour la manipulation des données, `cufflinks`, `plotly`, `seaborn` et `matplotlib` pour la visualisation des données, et enfin `utils` pour les fonctions auxiliaires.

- Configuration de l'application Streamlit

```python
st.title("Asset Allocation")
cf.go_offline()
```

Ici, le titre de l'application est défini et `cufflinks` est configuré pour fonctionner hors ligne.

- Sélection de l'indice de marché

```python
market_index = st.selectbox(
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
```

L'utilisateur peut sélectionner un indice de marché à partir d'un menu déroulant. En fonction de la sélection, les composants de l'indice (tickers et noms des entreprises) sont récupérés en utilisant des fonctions définies dans `utils.py`.

- Sélection des dates de début et de fin

```python
start_date = st.date_input(
    "Start date", 
    datetime.date(2019, 1, 1)
)
end_date = st.date_input(
    "End date", 
    datetime.date.today()
)
```

L'utilisateur peut choisir les dates de début et de fin pour l'analyse des données boursières.

- Sélection des actifs

```python
assets = st.multiselect(
    'Select the assets for the portfolio:', 
    available_tickers, 
    format_func=tickers_companies_dict.get
)
```

L'utilisateur peut sélectionner plusieurs actifs parmi les composants de l'indice choisi. `format_func` est utilisé pour afficher les noms des entreprises à la place des tickers.

- Vérification des dates

```python
if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")
```

Si la date de début est après la date de fin, un message d'erreur est affiché.

- Sélection de la méthode d'allocation d'actifs

```python
st.sidebar.header("Asset Allocation Method")
allocation_method = st.sidebar.radio("Select Asset Allocation Technique", 
                                     ["Monte Carlo simulations", "Scipy optimization", "CVXPY optimization", "Hierarchical Risk Parity"])
```

L'utilisateur peut choisir la méthode d'allocation d'actifs via un menu radio dans la barre latérale.

- Chargement des données boursières

```python
if len(assets) > 0:
    assets_data = load_data(assets, start_date, end_date)['Adj Close']
else:
    st.write("Choose some assets to build your Portfolio")
```

Si des actifs ont été sélectionnés, les données boursières ajustées sont chargées pour ces actifs. Sinon, un message invitant à choisir des actifs est affiché.

- Bouton pour exécuter l'allocation d'actifs

```python
run_allocation_button = st.button("Run Asset Allocation")
```

Un bouton est ajouté pour lancer le processus d'allocation d'actifs.

- Visualisation des prix ajustés et des rendements quotidiens

```python
if run_allocation_button:
    fig = go.Figure()
    for asset in assets:
        fig.add_trace(go.Scatter(x=assets_data.index, y=assets_data[asset], mode='lines', name=tickers_companies_dict[asset]))
    fig.update_layout(title="Adjusted Close Prices of Selected Stocks",
                      xaxis_title="Date",
                      yaxis_title="Adjusted Close Price",
                      legend=dict(title="Stocks"))
    st.plotly_chart(fig)
    
    avg_returns, cov_mat = calculate_statistics(assets_data)
    fig_returns = go.Figure()
    for asset in assets:
        fig_returns.add_trace(go.Scatter(x=assets_data.index, y=assets_data[asset].pct_change(), mode='lines', name=tickers_companies_dict[asset]))
    fig_returns.update_layout(title="Daily Returns of Selected Stocks",
                              xaxis_title="Date",
                              yaxis_title="Daily Returns",
                              legend=dict(title="Company Names"))
    st.plotly_chart(fig_returns)
```

Si le bouton d'allocation est pressé, les prix ajustés et les rendements quotidiens des actifs sélectionnés sont tracés et affichés.

- Simulation des portefeuilles avec Monte Carlo

```python
if run_allocation_button:
    np.random.seed(42)
    N_PORTFOLIOS = 10 ** 5
    n_assets = len(assets)
    weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    portf_rtns = np.dot(weights, avg_returns)
    portf_vol = []
    for i in range(0, len(weights)):
        vol = np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i])))
        portf_vol.append(vol)
    portf_vol = np.array(portf_vol)
    portf_sharpe_ratio = portf_rtns / portf_vol

    portf_results_df = pd.DataFrame(
        {"returns": portf_rtns,
        "volatility": portf_vol,
        "sharpe_ratio": portf_sharpe_ratio}
    )

    display_data_preview("Preview data", portf_results_df, key=6)
```

Les poids des portefeuilles sont générés aléatoirement pour un grand nombre de portefeuilles (100 000). Les rendements, volatilités et ratios de Sharpe des portefeuilles sont calculés et stockés dans un DataFrame.

- Tracé de la frontière efficiente avec Monte Carlo

```python
if allocation_method == "Monte Carlo simulations":
    N_POINTS = 100
    ef_rtn_list = []
    ef_vol_list = []

    possible_ef_rtns = np.linspace(
        portf_results_df["returns"].min(), 
        portf_results_df["returns"].max(), 
        N_POINTS
    )
    possible_ef_rtns = np.round(possible_ef_rtns, 2)    
    portf_rtns = np.round(portf_rtns, 2)

    for rtn in possible_ef_rtns:
        if rtn in portf_rtns:
            ef_rtn_list.append(rtn)
            matched_ind = np.where(portf_rtns == rtn)
            ef_vol_list.append(np.min(portf_vol[matched_ind]))

    st.subheader("Efficient Frontier")
    fig_ef, ax_ef = plt.subplots(figsize=(10, 6))
    scatter = ax_ef.scatter(
        x=portf_results_df["volatility"],
        y=portf_results_df["returns"],
        c=portf_results_df["sharpe_ratio"],
        cmap="RdYlGn",
        edgecolors="black",
        marker="o",
        alpha=0.8,
    )
    ax_ef.set(xlabel="Volatility", ylabel="Expected Returns", title="Efficient Frontier")
    ax_ef.plot(ef_vol_list, ef_rtn_list, "b--")
    MARKERS = generate_markers(n_assets)
    for asset_index in range(n_assets):
        ax_ef.scatter(
            x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
            y=avg_returns[asset_index],
            marker=MARKERS[asset_index],
            s=150,
            color="black",
            label=tickers_companies_dict[assets[asset_index]],
        )

    cbar = fig_ef.colorbar(scatter)
    cbar.set_label("Sharpe Ratio")
    ax_ef.legend()
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_ef)
```

La frontière efficiente est tracée en utilisant les portefeuilles simulés. Les portefeuilles individuels sont représentés par des points de couleur (ratio de Sharpe), et la frontière efficiente est tracée en pointillés bleus.

- Affichage des portefeuilles optimaux avec Monte Carlo

```python
max_sharpe_ind = np.argmax(portf_results_df["shar

pe_ratio"])
max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

min_vol_ind = np.argmin(portf_results_df["volatility"])
min_vol_portf = portf_results_df.loc[min_vol_ind]

max_return_ind = np.argmax(portf_results_df["returns"])
max_return_portf = portf_results_df.loc[max_return_ind]

weight_chart_data = pd.DataFrame({"Assets": assets, "Weights": weights[max_sharpe_ind]}).sort_values(by="Weights", ascending=False)
weight_chart = px.bar(weight_chart_data, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                      title="Asset Weights in Maximum Sharpe Ratio Portfolio", color="Assets")
st.plotly_chart(weight_chart)
print_portfolio_summary(perf=max_sharpe_portf, weights=weights[max_sharpe_ind], assets=assets, name="Maximum Sharpe Ratio")

weight_chart_data2 = pd.DataFrame({"Assets": assets, "Weights": weights[min_vol_ind]}).sort_values(by="Weights", ascending=False)
weight_chart2 = px.bar(weight_chart_data2, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                      title="Asset Weights in Minimum Volatility Portfolio", color="Assets")
st.plotly_chart(weight_chart2)
print_portfolio_summary(min_vol_portf, weights[min_vol_ind], assets, name="Minimum Volatility")

weight_chart_data3 = pd.DataFrame({"Assets": assets, "Weights": weights[max_return_ind]}).sort_values(by="Weights", ascending=False)
weight_chart3 = px.bar(weight_chart_data3, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                      title="Asset Weights in Maximum Return Portfolio", color="Assets")
st.plotly_chart(weight_chart3)
print_portfolio_summary(perf=max_return_portf, weights=weights[max_return_ind], assets=assets, name="Maximum Return")
```

Les portefeuilles optimaux (Maximum Sharpe Ratio, Minimum Volatility, Maximum Return) sont identifiés et affichés. Les poids des actifs dans chaque portefeuille optimal sont présentés sous forme de graphique à barres.

- Optimisation avec SciPy

```python
elif allocation_method == "Scipy optimization":
    rtns_range = np.linspace(-0.1, 0.55, 200)
    efficient_portfolios_scipy = get_efficient_frontier_scipy(avg_returns, cov_mat, rtns_range)
    vols_range_scipy = [x["fun"] for x in efficient_portfolios_scipy]
    with sns.plotting_context("paper"):
        fig_scipy, ax_scipy = plt.subplots()
        portf_results_df.plot(kind="scatter", x="volatility",
                              y="returns", c="sharpe_ratio",
                              cmap="RdYlGn", edgecolors="black",
                              ax=ax_scipy)
        ax_scipy.plot(vols_range_scipy, rtns_range, "b--", linewidth=3)
        ax_scipy.set(xlabel="Volatility",
                     ylabel="Expected Returns",
                     title="Efficient Frontier - SciPy Optimization")
        sns.despine()
        plt.tight_layout()
    st.pyplot(fig_scipy)
```

La frontière efficiente est calculée et tracée en utilisant l'optimisation avec SciPy.

- Optimisation avec CVXPY

```python
elif allocation_method == "CVXPY optimization":
    avg_returns = avg_returns.values
    cov_mat = cov_mat.values

    weights = cp.Variable(n_assets)
    gamma_par = cp.Parameter(nonneg=True)
    portf_rtn_cvx = avg_returns @ weights 
    portf_vol_cvx = cp.quad_form(weights, cov_mat)
    objective_function = cp.Maximize(
        portf_rtn_cvx - gamma_par * portf_vol_cvx
    )
    problem = cp.Problem(
        objective_function, 
        [cp.sum(weights) == 1, weights >= 0]
    )

    N_POINTS = 25
    portf_rtn_cvx_ef = []
    portf_vol_cvx_ef = []
    weights_ef = []
    gamma_range = np.logspace(-3, 3, num=N_POINTS)

    for gamma in gamma_range:
        gamma_par.value = gamma
        problem.solve()
        portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
        portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
        weights_ef.append(weights.value)

    fig_cvx, ax_cvx = plt.subplots()
    MARKERS = generate_markers(n_assets)
    ax_cvx.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-")
    for asset_index in range(n_assets):
        plt.scatter(x=np.sqrt(cov_mat[asset_index, asset_index]), 
                    y=avg_returns[asset_index], 
                    marker=MARKERS[asset_index], 
                    label=assets[asset_index],
                    s=150)
    ax_cvx.set(title="Efficient Frontier",
        xlabel="Volatility", 
        ylabel="Expected Returns")
    ax_cvx.legend()
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_cvx)
```

La frontière efficiente est calculée et tracée en utilisant l'optimisation avec CVXPY.

- Allocation hiérarchique des risques

```python
elif allocation_method == "Hierarchical Risk Parity":
    st.write("----------------------------------------will be available soon-------------------------------------------")
```

Cette méthode d'allocation n'est pas encore disponible et affiche un message indiquant que la fonctionnalité sera bientôt disponible.

- Conclusion

Le script `app.py` utilise Streamlit pour créer une application web interactive permettant aux utilisateurs de sélectionner des actifs financiers, de choisir une méthode d'allocation d'actifs et de visualiser les portefeuilles optimaux et la frontière efficiente. L'application intègre plusieurs techniques d'optimisation et utilise des bibliothèques populaires pour la récupération de données et la visualisation.