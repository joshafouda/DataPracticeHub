# Projet : Robo-Conseiller en Investissement à la Bourse du CAC40

## Description

Ce projet utilise Streamlit pour créer une application web interactive qui aide les investisseurs à allouer leurs actifs dans le marché boursier CAC40 en fonction de leurs caractéristiques personnelles et de leur tolérance au risque. L'application prédit la tolérance au risque d'un investisseur en fonction de plusieurs facteurs et utilise cette information pour déterminer l'allocation optimale des actifs. Elle affiche également les performances du portefeuille résultant.

## Image
imgs/project12/project12.png


## Instructions

1. **Importation des Packages**
   - Importez les bibliothèques nécessaires telles que `streamlit`, `pandas`, `numpy`, `plotly.express`, et `cvxopt`.

2. **Chargement des Données**
   - Chargez les données des investisseurs depuis le fichier `InputData.csv`.
   - Chargez les données des actifs du CAC40 depuis le fichier `CAC40Data.csv`.
   - Nettoyez les données des actifs en supprimant les colonnes avec plus de 30% de valeurs manquantes et en remplissant les valeurs manquantes restantes.

3. **Définition de la Fonction de Prédiction de la Tolérance au Risque**
   - Créez une fonction `predict_riskTolerance` qui charge un modèle de machine learning pré-entraîné et utilise ce modèle pour prédire la tolérance au risque d'un investisseur donné. Téléchargez le mod`le via ce [lien](https://drive.google.com/file/d/1XK8tLdJmAoJ_xlSIVElHdziN_U7Xcbh7/view?usp=sharing)

4. **Définition de la Fonction d'Allocation des Actifs**
   - Créez une fonction `get_asset_allocation` qui utilise la programmation quadratique pour déterminer l'allocation optimale des actifs en fonction de la tolérance au risque de l'investisseur et des rendements historiques des actifs sélectionnés.

5. **Création de l'Application Streamlit**
   - Définissez le titre de l'application et les sous-titres pour les différentes sections.
   - Utilisez des widgets de la barre latérale pour saisir les caractéristiques de l'investisseur, telles que l'âge, la valeur nette, le revenu, le niveau d'éducation, le statut marital, le nombre d'enfants, l'occupation et la volonté de prendre des risques.
   - Ajoutez un bouton pour calculer la tolérance au risque de l'investisseur en fonction des caractéristiques saisies.
   - Permettez à l'utilisateur de saisir la tolérance au risque et de sélectionner les actifs pour le portefeuille.
   - Ajoutez un bouton pour soumettre les informations et générer l'allocation des actifs et les performances du portefeuille.

6. **Affichage des Résultats**
   - Utilisez `plotly.express` pour créer des graphiques interactifs affichant l'allocation des actifs et les performances du portefeuille sur la période choisie.


## Resources

- [Dossier des données](https://drive.google.com/drive/folders/1RgvvdV5ffqVLfXftAjL6U9dz8O4ZB5-H?usp=sharing)
- [Présentation de l'application à construire](https://youtu.be/pEWxhmP5yN0)
- [Modèle ML pré-entraîné](https://drive.google.com/file/d/1XK8tLdJmAoJ_xlSIVElHdziN_U7Xcbh7/view?usp=sharing)
- [Formation Streamlit](https://www.youtube.com/playlist?list=PLmJWMf9F8euQKADN-mSCpTlp7uYDyCQNF)
- [Comment déployer une web app Streamlit](https://youtu.be/wjRlWuXmlvw)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)


## Execution du Projet

🖥 **Script app.py qui comporte le code de lápplication**

```python
import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import plotly.express as px
import cvxopt as opt
from cvxopt import solvers

investors = pd.read_csv('data/InputData.csv', index_col = 0)

assets = pd.read_csv('data/CAC40Data.csv',index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
assets.drop(labels=drop_list, axis=1, inplace=True)
assets=assets.fillna(method='ffill')

def predict_riskTolerance(X_input):

    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    # estimate accuracy on validation set
    predictions = loaded_model.predict(X_input)
    return predictions

#Asset allocation given the Return, variance
def get_asset_allocation(riskTolerance,stock_ticker):
    #ipdb.set_trace()
    assets_selected = assets.loc[:,stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1-riskTolerance

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    w=portfolios['x'].T
    #print (w)
    Alloc =  pd.DataFrame(data = np.array(portfolios['x']),index = assets_selected.columns)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    returns_final=(np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final,axis =1)
    returns_sum_pd = pd.DataFrame(returns_sum, index = assets.index )
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0,:] + 100
    return Alloc,returns_sum_pd

# Define the Streamlit app
st.title('Investment Advisor in the CAC40 stock market (by Josué AFOUDA)')

st.subheader('Step 2: Asset Allocation and Portfolio Performance')
st.sidebar.title('Step 1: Enter Investor Characteristics')

# Investor Characteristics
with st.sidebar:

    age = st.slider('Age:', min_value=investors['AGE07'].min(), max_value=70, value=25)
    net_worth = st.slider('NetWorth:', min_value=-1000000, max_value=3000000, value=10000)
    income = st.slider('Income:', min_value=-1000000, max_value=3000000, value=100000)
    education = st.slider('Education Level (scale of 4):', min_value=1, max_value=4, value=2)
    married = st.slider('Married:', min_value=1, max_value=2, value=1)
    kids = st.slider('Kids:', min_value=investors['KIDS07'].min(), max_value=investors['KIDS07'].max(), value=3)
    occupation = st.slider('Occupation:', min_value=1, max_value=4, value=3)
    willingness = st.slider('Willingness to take Risk:', min_value=1, max_value=4, value=3)

    if st.sidebar.button('Calculate Risk Tolerance'):
        X_input = [[age, education, married, kids, occupation, income, willingness, net_worth]]
        risk_tolerance_prediction = predict_riskTolerance(X_input)
        st.sidebar.write(f'Predicted Risk Tolerance: {round(float(risk_tolerance_prediction[0]*100), 2)}')

# Risk Tolerance Charts

risk_tolerance_text = st.text_input('Risk Tolerance (scale of 100):')
selected_assets = st.multiselect('Select the assets for the portfolio:', 
                                 options=list(assets.columns), 
                                 default=['Air Liquide', 'Airbus', 'Alstom', 'AXA', 'BNP Paribas'])

# Asset Allocation and Portfolio Performance

if st.button('Submit'):
    Alloc, returns_sum_pd = get_asset_allocation(float(risk_tolerance_text), selected_assets)

    # Display Asset Allocation chart
    st.subheader('Asset Allocation: Mean-Variance Allocation')
    fig_alloc = px.bar(Alloc, x=Alloc.index, y=Alloc.iloc[:, 0], 
                       labels={'index': 'Assets', '0': 'Allocation'})
    st.plotly_chart(fig_alloc)

    # Display Portfolio Performance chart
    st.subheader('Portfolio value of €100 investment')
    fig_performance = px.line(returns_sum_pd, x=returns_sum_pd.index, y=returns_sum_pd.iloc[:, 0], labels={'index': 'Date', '0': 'Portfolio Value'})
    st.plotly_chart(fig_performance)
```

Ce code utilise Streamlit pour créer une application web interactive destinée à aider les investisseurs à allouer leurs actifs dans le marché boursier CAC40 en fonction de leurs caractéristiques personnelles et de leur tolérance au risque.

- Importation des Packages

```python
import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import plotly.express as px
import cvxopt as opt
from cvxopt import solvers
```

Ces bibliothèques sont importées pour diverses fonctionnalités :
- `streamlit` pour créer l'interface web.
- `pandas` pour la manipulation des données.
- `joblib` pour charger un modèle de machine learning.
- `numpy` pour les calculs numériques.
- `plotly.express` pour créer des graphiques interactifs.
- `cvxopt` pour résoudre des problèmes d'optimisation quadratique.

- Chargement et Préparation des Données

```python
investors = pd.read_csv('data/InputData.csv', index_col=0)

assets = pd.read_csv('data/CAC40Data.csv', index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
assets.drop(labels=drop_list, axis=1, inplace=True)
assets = assets.fillna(method='ffill')
```

- `investors` contient les données des investisseurs.
- `assets` contient les données des actifs du CAC40.
- Les colonnes avec plus de 30% de valeurs manquantes sont supprimées.
- Les valeurs manquantes restantes sont remplies par propagation en avant (`ffill`).

- Fonction de Prédiction de la Tolérance au Risque

```python
def predict_riskTolerance(X_input):
    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    predictions = loaded_model.predict(X_input)
    return predictions
```

Cette fonction charge un modèle de machine learning pré-entraîné (`finalized_model.sav`) et prédit la tolérance au risque d'un investisseur basé sur les caractéristiques fournies.

- Fonction d'Allocation des Actifs

```python
def get_asset_allocation(riskTolerance, stock_ticker):
    assets_selected = assets.loc[:, stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1 - riskTolerance

    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
    w = portfolios['x'].T
    Alloc = pd.DataFrame(data=np.array(portfolios['x']), index=assets_selected.columns)

    returns_final = (np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final, axis=1)
    returns_sum_pd = pd.DataFrame(returns_sum, index=assets.index)
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0, :] + 100
    return Alloc, returns_sum_pd
```

Cette fonction utilise la programmation quadratique pour déterminer l'allocation optimale des actifs en fonction de la tolérance au risque et des rendements historiques des actifs sélectionnés.

- Définition de l'Application Streamlit

```python
st.title('Investment Advisor in the CAC40 stock market (by Josué AFOUDA)')

st.subheader('Step 2: Asset Allocation and Portfolio Performance')
st.sidebar.title('Step 1: Enter Investor Characteristics')
```

Ces lignes définissent le titre de l'application et les sous-titres pour les différentes sections.

- Caractéristiques de l'Investisseur

```python
with st.sidebar:
    age = st.slider('Age:', min_value=investors['AGE07'].min(), max_value=70, value=25)
    net_worth = st.slider('NetWorth:', min_value=-1000000, max_value=3000000, value=10000)
    income = st.slider('Income:', min_value=-1000000, max_value=3000000, value=100000)
    education = st.slider('Education Level (scale of 4):', min_value=1, max_value=4, value=2)
    married = st.slider('Married:', min_value=1, max_value=2, value=1)
    kids = st.slider('Kids:', min_value=investors['KIDS07'].min(), max_value=investors['KIDS07'].max(), value=3)
    occupation = st.slider('Occupation:', min_value=1, max_value=4, value=3)
    willingness = st.slider('Willingness to take Risk:', min_value=1, max_value=4, value=3)

    if st.sidebar.button('Calculate Risk Tolerance'):
        X_input = [[age, education, married, kids, occupation, income, willingness, net_worth]]
        risk_tolerance_prediction = predict_riskTolerance(X_input)
        st.sidebar.write(f'Predicted Risk Tolerance: {round(float(risk_tolerance_prediction[0] * 100), 2)}')
```

Ces widgets permettent à l'utilisateur de saisir les caractéristiques de l'investisseur et de calculer la tolérance au risque en utilisant le modèle de machine learning.

- Sélection des Actifs et Affichage des Résultats

```python
risk_tolerance_text = st.text_input('Risk Tolerance (scale of 100):')
selected_assets = st.multiselect('Select the assets for the portfolio:', 
                                 options=list(assets.columns), 
                                 default=['Air Liquide', 'Airbus', 'Alstom', 'AXA', 'BNP Paribas'])

if st.button('Submit'):
    Alloc, returns_sum_pd = get_asset_allocation(float(risk_tolerance_text), selected_assets)

    st.subheader('Asset Allocation: Mean-Variance Allocation')
    fig_alloc = px.bar(Alloc, x=Alloc.index, y=Alloc.iloc[:, 0], 
                       labels={'index': 'Assets', '0': 'Allocation'})
    st.plotly_chart(fig_alloc)

    st.subheader('Portfolio value of €100 investment')
    fig_performance = px.line(returns_sum_pd, x=returns_sum_pd.index, y=returns_sum_pd.iloc[:, 0], 
                              labels={'index': 'Date', '0': 'Portfolio Value'})
    st.plotly_chart(fig_performance)
```

- L'utilisateur saisit la tolérance au risque et sélectionne les actifs pour le portefeuille.
- Lorsqu'il appuie sur le bouton "Submit", l'application calcule l'allocation des actifs et les performances du portefeuille.
- Les résultats sont affichés sous forme de graphiques interactifs. 

Ce code crée une interface utilisateur intuitive et interactive pour aider les investisseurs à prendre des décisions éclairées concernant leur portefeuille d'investissement basé sur leur profil de risque.