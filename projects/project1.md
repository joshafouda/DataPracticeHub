# Projet : Service Machine Learning de prediction de la probabilité de défaut de paiement

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
- [Détection des Défauts de paiement des clients de cartes de crédit : Episode 1](https://youtu.be/loHmMjtnpjs?si=qVcmxXIOtFUH3lZH)
- [Détection des Défauts de paiement des clients de cartes de crédit : Episode 2](https://youtu.be/Fgf3iIOpMzY?si=xB4FWFB627rgReG2)
- [Comment déployer une web app Streamlit](https://youtu.be/wjRlWuXmlvw)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Tutoriels de Machine Learning](https://www.youtube.com/playlist?list=PLmJWMf9F8euTuNEnfnV-qdaVOOL8cIY9Q)
- [Documentation Scikit-Learn](https://scikit-learn.org/stable/documentation.html)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)

## Execution du Projet

Pour ce projet, vous pouvez travailler dans l'environnement de developpement Python de votre choix. Nous recommandons l'utilisation de Visual Studio Code (VSC).

🖥 **Exploration des donnees et Developpement du Modèle ML**

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

```python
print(raw_df.ID.nunique())
raw_df.info()
```

La colonne ID identifie de manière unique chaque observation de la dataframe. Donc chaque ligne de la dataframe correspond à un seul client. Voici la signification de chaque variable (Source : https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) :

- ***ID*** : Identifiant de chaque client

- ***LIMIT_BAL*** : Montant du crédit accordé en dollars taïwanais (comprend le crédit individuel et le crédit familial/complémentaire)

- ***SEX*** : Sexe (1=homme, 2=femme)

- ***EDUCATION*** : Niveau d'éducation (1=diplômé d'une école supérieure, 2=université, 3=lycée, 4=autre, 5=inconnu, 6=inconnu)

- ***MARRIAGE*** : Statut matrimonial (1=marié, 2=célibataire, 3=autre)

- ***AGE*** : Âge en années

- ***PAY_0*** : Statut de paiement en septembre 2005 (-1=paiement régulier, 1=retard de paiement d'un mois, 2=retard de paiement de deux mois, … 8=retard de paiement de huit mois, 9=retard de paiement de neuf mois et plus)

- ***PAY_2*** : Statut de paiement en août 2005 (échelle identique à celle ci-dessus)

- ***PAY_3*** : Statut de paiement en juillet 2005 (échelle identique à celle ci-dessus)

- ***PAY_4*** : Statut de paiement en juin 2005 (échelle identique à celle ci-dessus)

- ***PAY_5*** : Statut de paiement en mai 2005 (échelle identique à celle ci-dessus)

- ***PAY_6*** : Statut de paiement en avril 2005 (échelle identique à celle ci-dessus)

- ***BILL_AMT1*** : Montant de la facture en septembre 2005 (en dollars taïwanais)

- ***BILL_AMT2*** : Montant de la facture en août 2005 (en dollars taïwanais)

- ***BILL_AMT3*** : Montant de la facture en juillet 2005 (en dollars taïwanais)

- ***BILL_AMT4*** : Montant de la facture en juin 2005 (en dollars taïwanais)

- ***BILL_AMT5*** : Montant de la facture en mai 2005 (en dollars taïwanais)

- ***BILL_AMT6*** : Montant de la facture en avril 2005 (en dollars taïwanais)

- ***PAY_AMT1*** : Montant du paiement précédent en septembre 2005 (en dollars taïwanais)

- ***PAY_AMT2*** : Montant du paiement précédent en août 2005 (en dollars taïwanais)

- ***PAY_AMT3*** : Montant du paiement précédent en juillet 2005 (en dollars taïwanais)

- ***PAY_AMT4*** : Montant du paiement précédent en juin 2005 (en dollars taïwanais)

- ***PAY_AMT5*** : Montant du paiement précédent en mai 2005 (en dollars taïwanais)

- ***PAY_AMT6*** : Montant du paiement précédent en avril 2005 (en dollars taïwanais)

- ***default payment next month*** : Défaut de paiement (1=oui, 0=non)

D'après la description des variables, les colonnes `SEX`, `EDUCATION` et `MARRIAGE` sont en réalité des colonnes catégorielles car elles décrivent des catégories ou des classes discrètes plutôt que des valeurs numériques continues.

Par ailleurs, les colonnes `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5` et `PAY_6` ne sont pas catégorielles, mais plutôt des variables ordinales. Elles représentent le statut de paiement pour différents mois, et elles ont une échelle d'ordre, ce qui signifie que les valeurs ont une signification hiérarchique. Ces colonnes sont souvent utilisées pour indiquer le nombre de mois de retard dans le paiement, et elles sont généralement codées avec des entiers tels que -1 pour "paiement régulier", 1 pour "retard d'un mois", 2 pour "retard de deux mois", et ainsi de suite. Les valeurs numériques dans ces colonnes ont une signification progressive, ce qui en fait des variables ordinales. Vous pouvez les considérer comme des variables numériques discrètes, et elles peuvent être incluses dans des analyses statistiques et des modèles d'apprentissage automatique en tant que telles. Par exemple, vous pourriez utiliser ces variables pour analyser les comportements de paiement des clients et leur relation avec les défauts de crédit.

```python
df = FormattageRawData()
df
```

- **Analyse exploratoire des donnees** :

L'analyse exploratoire des données est une étape essentielle dans tout projet de Machine Learning. Cette étape vous permet de comprendre vos données, de mettre en évidence des tendances, des relations et des anomalies, et de préparer les données pour la modélisation.

```python
# Résumé statistique des variables numériques
df.describe().T

# Résumé statistique des variables catégorielles
df.describe(include=['category']).T

# Distribution des variables 
plot_distributions(df)
```


```python
# Distributions des varibles selon le sexe
plot_discretize_distributions(df, 'sex')
```

L'analyse du sexe dans un contexte de Machine Learning appliqué à la prédiction des défauts de paiement peut être importante pour plusieurs raisons :

- **Compréhension des différences de comportement de remboursement** : Les hommes et les femmes peuvent avoir des comportements de remboursement différents envers les prêts ou les cartes de crédit. En examinant ces différences, vous pourriez identifier des tendances qui pourraient être utiles dans la prédiction des défauts de paiement.

- **Caractéristiques explicatives** : Le sexe peut être une caractéristique explicative qui influence le risque de défaut de paiement. Par exemple, il est possible que les hommes aient tendance à présenter un risque de défaut de paiement plus élevé que les femmes en fonction de certaines variables, telles que le niveau d'éducation, l'âge ou le statut matrimonial.

- **Segmentation de la clientèle** : Dans le cadre de la gestion du risque de crédit, la segmentation de la clientèle par sexe peut être utile. Par exemple, vous pourriez développer des modèles de prédiction de défaut de paiement spécifiques pour les hommes et les femmes, en tenant compte des caractéristiques qui sont importantes pour chaque groupe.

- **Équité et éthique** : L'analyse du sexe dans le Machine Learning peut également être importante du point de vue de l'équité et de l'éthique. Elle peut aider à identifier et à corriger d'éventuels biais de genre dans les modèles de prédiction, garantissant ainsi un traitement équitable des individus.

- **Adaptation de stratégies de prêt** : En comprenant comment le sexe peut influencer le risque de défaut de paiement, les institutions financières peuvent adapter leurs stratégies de prêt, de fixation des taux d'intérêt et de limites de crédit pour mieux répondre aux besoins de chaque groupe.

Cependant, il est essentiel de traiter les données liées au sexe avec sensibilité, de respecter les réglementations sur la protection de la vie privée et de veiller à ce que toute utilisation des données soit conforme aux normes éthiques et légales en vigueur. Il est également important de reconnaître que le sexe n'est qu'une variable parmi d'autres qui peuvent influencer la prédiction des défauts de paiement, et il doit être pris en compte en conjonction avec d'autres caractéristiques pertinentes pour développer des modèles de prédiction précis et équitables.


```python
plot_discretize_distributions(df, 'default_payment_next_month')

# Matrice de corrélation
corr_mat = df[[c for c in df.columns if c not in ['id', 'sex', 'education', 'marriage']]].corr()
plot_correlation_matrix(corr_mat)
```


```python
# Analyse de la relation entre l'âge et le montant du crédit accordé
sns.jointplot(
    data=df, x="age", y="limit_bal",
    hue="default_payment_next_month",
    height=8
)
plt.title("Age vs. Montant du crédit")
plt.show()

sns.boxplot(data=df, y="limit_bal", x="marriage", hue="default_payment_next_month");

# Distribution du montant du crédit accordé par niveau d'éducation
sns.violinplot(data=df, y="limit_bal", x="education", hue="default_payment_next_month", split=True)
plt.xticks(rotation=45)
plt.show()

pct_default_by_category(df, "education")
```

la plupart des défauts surviennent parmi les clients ayant fait des études secondaires, tandis que le moins de défauts se produit dans la catégorie Autres.

```python
pct_default_by_category(df, "marriage")
```


- **Division des données en 3 ensembles : Entrainement, Validation et Test** :

Il est courant de diviser les données en trois ensembles distincts : l'ensemble d'entraînement, l'ensemble de validation et l'ensemble de test. Cette approche est souvent utilisée pour ajuster les hyperparamètres du modèle et évaluer sa performance de manière plus robuste.

Voici comment vous pouvez effectuer cette division en utilisant scikit-learn :

```python
# Séparer les caractéristiques (X) de la variable cible (y)
X = df.drop(columns=['id', 'default_payment_next_month'])
y = df['default_payment_next_month']

# Diviser les données en ensembles d'entraînement, de validation et de test
# Par exemple, 60% pour l'entraînement, 20% pour la validation et 20% pour le test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Afficher la taille des ensembles d'entraînement, de validation et de test
print("Taille de l'ensemble d'entraînement :", X_train.shape, y_train.shape)
print("Taille de l'ensemble de validation :", X_val.shape, y_val.shape)
print("Taille de l'ensemble de test :", X_test.shape, y_test.shape)
print("")
print("Distribution des classes dans chaque ensemble ----")
print(f"Total: {y.value_counts(normalize=True).values}")
print(f"Train: {y_train.value_counts(normalize=True).values}")
print(f"Valid: {y_val.value_counts(normalize=True).values}")
print(f"Test: {y_test.value_counts(normalize=True).values}")

X_train.info()

# Nombre de valeurs manquantes par colonne
X_train.isna().sum()
```

- **Pipeline de Modélisation** :


```python
cat_features = ['sex', 'education', 'marriage']

num_features = [c for c in X_train.columns if c not in cat_features]
num_features

# Création de transformateurs pour les colonnes catégorielles et numériques
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False))  # Utilisation de l'encodage one-hot
])

num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Normalisation des caractéristiques numériques
])

# Création du transformateur de colonnes en utilisant ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# Création du modèle (dans cet exemple, nous utilisons un RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Création du pipeline complet
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

pipeline

# Entraînement du modèle en utilisant l'ensemble d'entraînement
pipeline.fit(X_train, y_train)

# Evaluation de la performance du pipeline modèle
LABELS = ["No Default", "Default"]
performance_evaluation_report(
    pipeline, X_val, y_val, labels=LABELS, 
    show_plot=True, show_pr_curve=True
)
```


- **Réglage des hyperparamètres avec une recherche par grille (Grid Search)** :

```python
k_fold = StratifiedKFold(5, shuffle=True, random_state=42)
```

Ce code crée un objet de validation croisée stratifiée (StratifiedKFold) pour diviser un ensemble de données en cinq plis (folds) pour la validation croisée (cross-validation) avec les caractéristiques suivantes :

* n_splits=5 : Cela signifie que l'ensemble de données sera divisé en cinq plis, ce qui est couramment utilisé pour la validation croisée à cinq plis.

* shuffle=True : Cela indique que les données seront mélangées aléatoirement avant d'être divisées en plis. Le mélange aléatoire des données est utile pour garantir que les plis ne sont pas biaisés en fonction de l'ordre des données. Le mélange aléatoire est effectué en utilisant la graine (seed) définie par random_state=42, ce qui garantit que le mélange est reproductible si la même graine est utilisée.

En résumé, l'objet k_fold de validation croisée stratifiée sera utilisé pour diviser l'ensemble de données en cinq plis de manière aléatoire et stratifiée, ce qui est couramment utilisé pour évaluer la performance d'un modèle sur différents sous-ensembles de données dans le cadre de la validation croisée. La stratification garantit que la répartition des classes cibles (si vous effectuez une classification) est préservée dans chaque pli, ce qui est important pour obtenir des résultats de validation croisée fiables.

```python
# Evaluation du pipeline à l'aide d'une validation croisée
scores_cv = cross_val_score(pipeline, X_train, y_train, cv=k_fold)
print(scores_cv)
print(np.mean(scores_cv))
print(np.std(scores_cv))

# Ajout d'autres métriques à la validation croisée
cv_scores = cross_validate(pipeline, X_train, y_train, cv=k_fold,
                           scoring=["accuracy", "precision", "recall",
                                    "roc_auc"])
pd.DataFrame(cv_scores)

# Définition de la grille de paramètres
param_grid = {
    "model__criterion": ["entropy", "gini"],
    "model__max_depth": range(7, 11),
    "model__n_estimators": [150, 200, 250]
}
# on peut aussi régler des hyperparamètres des composants du preprocessor
  # Exemple : "preprocessor__numerical__outliers__n_std": [3, 4]

# Recherche des meilleurs hyperparamètres
classifier_gs = GridSearchCV(pipeline, param_grid,
                             scoring="roc_auc", cv=k_fold,
                             n_jobs=-1, verbose=1)

classifier_gs.fit(X_train, y_train)

print(f"Best parameters: {classifier_gs.best_params_}")
print(f"roc_auc (Training set): {classifier_gs.best_score_:.4f}")
print(f"roc_auc (Validation set): {metrics.roc_auc_score(y_val, classifier_gs.predict(X_val)):.4f}")

# Evaluation du meilleur modèle sur les données de test
LABELS = ["No Default", "Default"]
performance_evaluation_report(
    classifier_gs, X_test, y_test, labels=LABELS, 
    show_plot=True, show_pr_curve=True
)
```

- **Enregistrement du modele** :

Si vous êtes satisfait de la performance de votre modèle, vous pouvez l'enregistrer.

```python
import joblib

# Enregistrez le modèle dans un fichier
model_filename = "classifier_gs_model.pkl"
joblib.dump(classifier_gs, model_filename)

# Charger le modèle depuis le fichier
loaded_model = joblib.load("classifier_gs_model.pkl")

# Vous pouvez maintenant utiliser loaded_model pour faire des prédictions, etc.

# Vérifier que le modèle chargé est le même que celui enregistré
performance_evaluation_report(
    loaded_model, X_test, y_test, labels=LABELS, 
    show_plot=True, show_pr_curve=True
)
```

- **Utilisation du modèle** :

```python
def make_prediction(model, features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    probability = np.round(probability * 100, 2)
    return prediction, probability

make_prediction(loaded_model, X_train.head(3))
```

🖥 **Contenu du module credit_card_default_utils.py et explication de chaque fonction**

```python
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Fonction pour télécharger les données brutes
def DownloadRawData():
  # URL du fichier ZIP
  url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

  # Nom du fichier ZIP (sans chemin, donc dans le répertoire principal)
  zip_filename = "credit_card_data.zip"

  # Téléchargement du fichier ZIP
  response = requests.get(url)
  with open(zip_filename, 'wb') as f:
      f.write(response.content)

  # Extraction du fichier ZIP
  with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
      zip_ref.extractall()

# Fonction pour importer les données brutes
def ReadRawData():
  # Chemin du fichier de données extrait
  data_file = "default of credit card clients.xls"

  # Charger le fichier de données dans un DataFrame Pandas
  data = pd.read_excel(data_file, header=1)  # header=1 pour ignorer la première ligne d'en-tête

  # Returner la dataframe data
  return data


# Reformattage des données brutes
def FormattageRawData():
  # Lire les données brutes
  raw_data = ReadRawData()

  # Renommer les noms de colonne
  raw_data.columns = raw_data.columns.str.lower().str.replace(" ", "_")
  months = ["sep", "aug", "jul", "jun", "may", "apr"]
  variables = ["payment_status", "bill_statement", "previous_payment"]
  new_column_names = [x + "_" + y for x in variables for y in months]
  rename_dict = {x: y for x, y in zip(raw_data.loc[:, "pay_0":"pay_amt6"].columns, new_column_names)}
  raw_data.rename(columns=rename_dict, inplace=True)

  # Mapper les nombres aux chaines de caractères
  gender_dict = {1: "Male",
                2: "Female"}
  education_dict = {0: "Others",
                    1: "Graduate school",
                    2: "University",
                    3: "High school",
                    4: "Others",
                    5: "Others",
                    6: "Others"}
  marital_status_dict = {0: "Others",
                        1: "Married",
                        2: "Single",
                        3: "Others"}
  payment_status = {-2: "Unknown",
                    -1: "Payed duly",
                    0: "Unknown",
                    1: "Payment delayed 1 month",
                    2: "Payment delayed 2 months",
                    3: "Payment delayed 3 months",
                    4: "Payment delayed 4 months",
                    5: "Payment delayed 5 months",
                    6: "Payment delayed 6 months",
                    7: "Payment delayed 7 months",
                    8: "Payment delayed 8 months",
                    9: "Payment delayed >= 9 months"}
  raw_data["sex"] = raw_data["sex"].map(gender_dict)
  raw_data["education"] = raw_data["education"].map(education_dict)
  raw_data["marriage"] = raw_data["marriage"].map(marital_status_dict)

  # Convertir les colonnes 'sex', 'education', 'default_payment_next_month' et 'marriage' en variables catégorielles
  categorical_columns = ['sex', 'marriage', 'education', 'default_payment_next_month']
  raw_data[categorical_columns] = raw_data[categorical_columns].astype('category')

  # Convertir les colonnes payment_status en variables ordinales
  payment_order = list(payment_status.keys())
  payment_categories = pd.CategoricalDtype(categories=payment_order, ordered=True)
  payment_columns = ['payment_status_sep', 'payment_status_aug', 'payment_status_jul',
                     'payment_status_jun', 'payment_status_may', 'payment_status_apr']
  raw_data[payment_columns] = raw_data[payment_columns].astype(payment_categories)

  # Sauvegarde au format csv
  raw_data.to_csv("credit_card_default.csv", index=False)

  # Retourner les données reformattées
  return raw_data


# Fonction pour tracer les distributions de toutes les variables
def plot_distributions(df):
  # Sélectionner uniquement les colonnes numériques
  numeric_columns = df.select_dtypes(include='number').drop(columns=['id'])

  # Afficher un histogramme pour chaque colonne numérique
  plt.figure(figsize=(15, 20))  # Adapter la taille de la figure en fonction du nombre de colonnes
  num_cols = len(numeric_columns.columns)

  # Créer une grille de sous-graphiques adaptée au nombre de colonnes numériques
  for i, column in enumerate(numeric_columns.columns):
      plt.subplot(5, 3, i + 1)  # Adapté à 15 colonnes numériques
      plt.hist(df[column], bins=20, color='blue', alpha=0.7)
      plt.title(f'Histogramme de {column}')
      plt.xlabel(column)
      plt.ylabel('Fréquence')

  plt.tight_layout()
  plt.show()
  plt.close()

  # Afficher des boîtes à moustaches pour chaque colonne numérique (avec axes séparés)
  plt.figure(figsize=(15, 20))  # Adapter la taille de la figure en fonction du nombre de colonnes
  num_cols = len(numeric_columns.columns)

  # Créer une grille de sous-graphiques pour les boîtes à moustaches
  for i, column in enumerate(numeric_columns.columns):
      plt.subplot(5, 3, i + 1)  # Adapté à 15 colonnes numériques
      sns.set(style="whitegrid")
      sns.boxplot(x=df[column], palette="Set2")
      plt.title(f'Boîte à moustaches de {column}')
      plt.xlabel(column)

  plt.tight_layout()
  plt.show()
  plt.close()


  # Colonnes à analyser
  columns_to_analyze = [
      'sex', 'education', 'marriage', 'default_payment_next_month',
      'payment_status_sep', 'payment_status_aug', 'payment_status_jul',
      'payment_status_jun', 'payment_status_may', 'payment_status_apr'
  ]

  # Diagrammes à barres
  for column in columns_to_analyze:
      print(f"Analyse univariée de la colonne '{column}':\n")

      # Compter les occurrences de chaque catégorie
      value_counts = df[column].value_counts(normalize=True)
      print(f"Fréquence des catégories :\n{value_counts}\n")

      # Afficher un graphique à barres pour visualiser la distribution
      plt.figure(figsize=(8, 6))
      sns.countplot(data=df, x=column, palette='Set1')
      plt.title(f'Distribution de {column}')
      plt.xlabel(column)
      plt.ylabel('Fréquence')
      plt.show()
      plt.close()

      # Statistiques descriptives
      print(f"Statistiques descriptives pour {column}:\n")
      print(df[column].describe())

      print("\n" + "="*50 + "\n")


# affichez les distributions en discrétisant suivant une variable catégorielle
def plot_discretize_distributions(df, cat_var):
  # Créer des boîtes à moustaches pour chaque colonne numérique en les segmentant par sexe
  plt.figure(figsize=(15, 20))

  # Sélectionner uniquement les colonnes numériques
  numeric_columns = df.select_dtypes(include='number').drop(columns=['id'])

  # Créer une grille de sous-graphiques pour les boîtes à moustaches
  for i, column in enumerate(numeric_columns.columns):
      plt.subplot(5, 3, i + 1)  # Adapté à 15 colonnes numériques
      sns.set(style="whitegrid")
      sns.boxplot(data=df, x=column, y=cat_var, palette="Set2")
      plt.title(column + ' par ' + cat_var)
      plt.xlabel(column)
      plt.ylabel(cat_var)

  plt.tight_layout()
  plt.show()
  plt.close()


  # Colonnes à analyser par sexe
  columns_to_analyze = [
      'sex', 'education', 'marriage', 'default_payment_next_month',
      'payment_status_sep', 'payment_status_aug', 'payment_status_jul',
      'payment_status_jun', 'payment_status_may', 'payment_status_apr'
  ]
  columns_to_analyze.remove(cat_var)

  # Créer des graphiques à barres pour chaque colonne
  for column in columns_to_analyze:
      plt.figure(figsize=(10, 6))
      sns.countplot(data=df, x=column, hue=cat_var, palette='Set1')

      # Personnalisation du graphique
      plt.title(column + ' par ' + cat_var)
      plt.xlabel(column)
      plt.ylabel('Fréquence')
      plt.xticks(rotation=45)  # Faire pivoter les étiquettes de l'axe des x pour plus de lisibilité

      # Afficher le graphique
      plt.legend(title=cat_var)
      plt.show()
      plt.close()


# Matrice de corrélation
def plot_correlation_matrix(corr_mat):
  sns.set(style="white")
  mask = np.zeros_like(corr_mat, dtype=bool)
  mask[np.triu_indices_from(mask)] = True
  fig, ax = plt.subplots(figsize=(12, 10))
  cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
  sns.heatmap(
      corr_mat, mask=mask, cmap=cmap, annot=True,
      fmt=".1f", vmin=-1, vmax=1, center=0, square=True,
      linewidths=.5, cbar_kws={"shrink": .5}, ax=ax
  )
  ax.set_title("Matrice de Correlation", fontsize=16)
  sns.set(style="darkgrid")


def pct_default_by_category(df, cat_var):
  # Pourcentage de défauts de paiement
  ax = df.groupby(cat_var)["default_payment_next_month"] \
  .value_counts(normalize=True) \
  .unstack() \
  .plot(kind="barh", stacked="True")
  ax.set_title("Pourcentage de défauts de paiement",
  fontsize=16)
  ax.legend(title="Defaut de paiement", bbox_to_anchor=(1,1))
  plt.show()


# Fonction d'évaluation des modèles
def performance_evaluation_report(model, X_test, y_test, show_plot=False, labels=None, show_pr_curve=False):
    """
    Function for creating a performance report of a classification model.

    Parameters
    ----------
    model : scikit-learn estimator
        A fitted estimator for classification problems.
    X_test : pd.DataFrame
        DataFrame with features matching y_test
    y_test : array/pd.Series
        Target of a classification problem.
    show_plot : bool
        Flag whether to show the plot
    labels : list
        List with the class names.
    show_pr_curve : bool
        Flag whether to also show the PR-curve. For this to take effect,
        show_plot must be True.

    Return
    ------
    stats : pd.Series
        A series with the most important evaluation metrics
    """

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    cm = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(
        y_test, y_pred_prob)
    pr_auc = metrics.auc(recall, precision)

    if show_plot:

        if labels is None:
            labels = ["Negative", "Positive"]

        N_SUBPLOTS = 3 if show_pr_curve else 2
        PLOT_WIDTH = 20 if show_pr_curve else 12
        PLOT_HEIGHT = 5 if show_pr_curve else 6

        fig, ax = plt.subplots(
            1, N_SUBPLOTS, figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        fig.suptitle("Evaluation de la Performance du Modèle", fontsize=16)

        # plot 1: confusion matrix ----

        # preparing more descriptive labels for the confusion matrix
        cm_counts = [f"{val:0.0f}" for val in cm.flatten()]
        cm_percentages = [f"{val:.2%}" for val in cm.flatten()/np.sum(cm)]
        cm_labels = [f"{v1}\n{v2}" for v1, v2 in zip(cm_counts,cm_percentages)]
        cm_labels = np.asarray(cm_labels).reshape(2,2)

        sns.heatmap(cm, annot=cm_labels, fmt="", linewidths=.5, cmap="Greens",
                    square=True, cbar=False, ax=ax[0],
                    annot_kws={"ha": "center", "va": "center"})
        ax[0].set(xlabel="Predicted label",
                  ylabel="Actual label", title="Confusion Matrix")
        ax[0].xaxis.set_ticklabels(labels)
        ax[0].yaxis.set_ticklabels(labels)

        # plot 2: ROC curve ----

        metrics.RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1], name="")
        ax[1].set_title("ROC Curve")
        ax[1].plot(fp/(fp+tn), tp/(tp+fn), "ro",
                   markersize=8, label="Decision Point")
        ax[1].plot([0, 1], [0, 1], "r--")

        if show_pr_curve:

            metrics.PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax[2], name="")
            ax[2].set_title("Precision-Recall Curve")

    stats = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "specificity": (tn / (tn + fp)),
        "f1_score": metrics.f1_score(y_test, y_pred),
        "cohens_kappa": metrics.cohen_kappa_score(y_test, y_pred),
        "matthews_corr_coeff": metrics.matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": metrics.average_precision_score(y_test, y_pred_prob)
    }

    return stats
```

Expliquons chacune de ces fonctions :

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


***Explication de la fonction FormattageRawData()***

La fonction FormattageRawData effectue plusieurs tâches de prétraitement sur les données brutes extraites. Voici comment cette fonction travaille :

1. Lecture des données brutes :

    - Tout d'abord, elle utilise la fonction ReadRawData pour lire les données brutes à partir du fichier Excel et les stocker dans le DataFrame raw_data.

2. Renommage des noms de colonne :

    - La fonction renomme les noms de colonne en mettant en minuscules et en remplaçant les espaces par des underscores (_) pour rendre les noms de colonne plus conviviaux et cohérents.

3. Création de nouveaux noms de colonne :

    - La fonction crée de nouveaux noms de colonne en combinant des variables et des mois, par exemple, "payment_status_sep", "payment_status_aug", etc. Ces nouveaux noms de colonne sont stockés dans la liste new_column_names.

    - Elle définit un dictionnaire rename_dict pour mapper les anciens noms de colonne aux nouveaux noms, puis renomme effectivement les colonnes du DataFrame.

4. Mappage des nombres aux chaînes de caractères :

    - La fonction utilise des dictionnaires pour mapper des valeurs numériques aux chaînes de caractères pour les colonnes "sex," "education," "marriage," et "payment_status." Par exemple, elle mappe 1 à "Male" et 2 à "Female" pour la colonne "sex."

5. Conversion des colonnes en variables catégorielles :

    - Certaines colonnes, telles que "sex," "marriage," "education," et "default_payment_next_month," sont converties en variables catégorielles en utilisant le type de données Pandas "category."

6. Conversion des colonnes "payment_status" en variables ordinales :

    - Les colonnes "payment_status_sep," "payment_status_aug," etc., sont converties en variables ordinales. Cela signifie que l'ordre des catégories a une signification. Par exemple, les paiements retardés de 1 mois sont ordonnés avant ceux de 2 mois. Ces colonnes sont également converties en utilisant le type de données Pandas "CategoricalDtype" avec un ordre spécifique défini par le dictionnaire payment_status.

7. Sauvegarde au format CSV :

    - Enfin, la fonction sauvegarde le DataFrame formaté au format CSV dans un fichier nommé "credit_card_default.csv" en utilisant la méthode to_csv.

8. Retour des données reformatées :

    - La fonction retourne le DataFrame raw_data contenant les données après le prétraitement.

Ainsi, cette fonction prépare les données brutes pour une analyse ultérieure en les renommant, en mappant des valeurs, en convertissant des colonnes en types appropriés et en les sauvegardant dans un fichier CSV. Cela permet d'obtenir un ensemble de données prêt pour l'analyse ou l'apprentissage automatique.


***Explication de la fonction plot_distributions()*** :

La fonction `plot_distributions` prend un DataFrame df en entrée et génère des visualisations pour analyser les distributions des données. Elle effectue plusieurs tâches pour les données numériques et catégorielles :

1. Histogrammes pour les données numériques :

    - Elle commence par sélectionner uniquement les colonnes numériques du DataFrame (en excluant une colonne appelée 'id' si elle existe).

    - Ensuite, elle génère un ensemble d'histogrammes pour chaque colonne numérique, en utilisant la bibliothèque matplotlib.pyplot. Chaque histogramme représente la distribution des valeurs d'une colonne numérique. Les paramètres tels que le nombre de bacs (bins), la couleur et l'alpha sont personnalisables.

2. Boîtes à moustaches (box plots) pour les données numériques :

    - Après les histogrammes, la fonction génère des boîtes à moustaches pour chaque colonne numérique. Les boîtes à moustaches montrent des mesures de tendance centrale et de dispersion pour chaque colonne, ce qui permet de détecter les valeurs aberrantes et d'analyser la répartition des données.

3. Diagrammes à barres pour les données catégorielles :

    - Ensuite, la fonction sélectionne un ensemble de colonnes catégorielles à analyser, telles que le "sexe," "éducation," "mariage," et les "statuts de paiement" pour différents mois.

    - Pour chaque colonne catégorielle, elle génère un diagramme à barres pour visualiser la distribution des catégories et leurs fréquences relatives. Elle utilise la bibliothèque seaborn pour créer ces graphiques à barres.

4. Statistiques descriptives :

    - Enfin, la fonction affiche des statistiques descriptives pour chaque colonne catégorielle. Ces statistiques comprennent le nombre total d'observations, le nombre de catégories uniques, la catégorie la plus fréquente, et d'autres informations statistiques de base.

La fonction est utile pour explorer et comprendre la distribution des données dans un DataFrame. Elle fournit des informations visuelles et statistiques pour les données numériques et catégorielles, ce qui peut être essentiel dans le processus de prétraitement et d'analyse des données.


***Explication de la fonction plot_discretize_distributions()*** :

La fonction `plot_discretize_distributions` est conçue pour explorer et analyser la distribution des données numériques en les segmentant par rapport à une variable catégorielle spécifique cat_var. Elle fournit une vue comparative des distributions des données numériques en fonction des catégories de la variable cat_var. Voici comment la fonction fonctionne :

1. Boîtes à moustaches segmentées par catégorie :

    - La fonction commence par créer un ensemble de boîtes à moustaches pour chaque colonne numérique du DataFrame df, en les segmentant par la variable catégorielle cat_var. Cela signifie qu'elle génère un graphique à boîtes pour chaque colonne numérique, mais elle divise les données en fonction des différentes catégories de cat_var. Par exemple, si cat_var est la variable "sexe," la fonction générera des boîtes à moustaches pour chaque colonne numérique (comme "montant du crédit" ou "âge"), mais les affichera séparément pour les catégories "Homme" et "Femme."

    - Chaque boîte à moustaches montre la distribution des données numériques pour une catégorie spécifique de cat_var. Elle peut révéler des différences dans la distribution des données entre les catégories.

2. Diagrammes à barres segmentés par catégorie :

    - Ensuite, la fonction crée des diagrammes à barres pour un ensemble de colonnes catégorielles (définies dans columns_to_analyze) en les segmentant également par cat_var. Cela signifie que pour chaque colonne catégorielle, elle génère des diagrammes à barres qui montrent la distribution des catégories de cette colonne, mais en les divisant par les catégories de cat_var. Par exemple, si cat_var est "sexe," la fonction générera des diagrammes à barres pour "éducation" ou "mariage," montrant comment la distribution de ces catégories varie en fonction du sexe ("Homme" et "Femme").

    - Les diagrammes à barres permettent de visualiser la répartition des catégories de chaque colonne catégorielle en fonction des différentes catégories de cat_var.

3. Personnalisation et affichage :

    - Pour chaque graphique, la fonction personnalise le titre, les étiquettes des axes, et effectue d'autres ajustements pour une meilleure lisibilité.

    - Enfin, elle affiche les graphiques à boîtes à moustaches et les diagrammes à barres.

L'objectif de cette fonction est d'aider à identifier des tendances ou des différences dans les données numériques en fonction des catégories de la variable cat_var. Cela peut être particulièrement utile pour l'analyse exploratoire des données et pour comprendre comment les caractéristiques numériques varient en fonction de différentes catégories.


***Explique la fonction plot_correlation_matrix *** :

La fonction `plot_correlation_matrix` est utilisée pour générer une carte thermique (heatmap) de la matrice de corrélation entre les différentes variables numériques. Cette carte thermique est un moyen de visualiser la force et la direction des relations linéaires entre les variables numériques d'un ensemble de données. Voici comment cette fonction fonctionne :

1. Préparation de la carte thermique :

    - La fonction commence par préparer la figure et les axes pour la carte thermique. Elle crée une figure de taille (12, 10), ce qui permet d'obtenir un graphique de grande taille pour une meilleure lisibilité.

    - Elle initialise un masque booléen `mask` de la même forme que la matrice de corrélation `corr_mat`. Le masque est utilisé pour masquer la partie supérieure de la carte thermique (pour éviter les duplications) en définissant `True` pour les positions qui doivent être masquées.

2. Choix de la palette de couleurs :

    - La palette de couleurs est définie avec `sns.diverging_palette`. Cela permet de choisir une palette de couleurs qui va du bleu (valeurs négatives) au rouge (valeurs positives) avec un centre neutre. Cela signifie que les valeurs négatives apparaîtront en bleu, les valeurs positives en rouge, et les valeurs proches de zéro en blanc. Cette palette de couleurs permet de visualiser les corrélations positives et négatives.

3. Création de la carte thermique :

La fonction `sns.heatmap` est utilisée pour créer la carte thermique. Elle prend en compte les éléments suivants :

    - `corr_mat`: La matrice de corrélation à afficher.

    - `mask`: Le masque booléen pour masquer la moitié supérieure de la carte.

    - `cmap`: La palette de couleurs à utiliser.

    - `annot`: Si True, les valeurs de corrélation sont annotées dans chaque cellule de la carte.

    - `fmt`: Le format des valeurs annotées (1 décimale dans cet exemple).

    - `vmin` et `vmax`: Les valeurs minimales et maximales de la plage de couleur (dans cet exemple, de -1 à 1 pour les corrélations).

    - `center`: La valeur au centre de la palette de couleurs (0 dans cet exemple).

    - `square`: Si True, assure que les cellules de la carte sont carrées.

    - `linewidths`: Épaisseur des lignes de séparation entre les cellules.

    - `cbar_kws`: Les paramètres de la barre de couleur (taille).

    - `ax`: L'axe sur lequel la carte thermique est dessinée.

4. Personnalisation et titrage :

    - La fonction ajoute un titre à la carte thermique en utilisant ax.set_title. Le titre indique qu'il s'agit de la "Matrice de Corrélation".

5. Réglage du style :

    - Enfin, la fonction utilise sns.set pour ajuster le style du graphique. Elle utilise "darkgrid," un style de grille sombre.

L'objectif de cette fonction est de visualiser rapidement la corrélation entre les variables numériques d'un ensemble de données. Cela permet d'identifier les relations linéaires entre les variables et d'explorer comment elles sont liées les unes aux autres.


***Explication de la fonction pct_default_by_category*** :

La fonction `pct_default_by_category` est conçue pour générer un graphique à barres empilées qui affiche le pourcentage de défauts de paiement pour différentes catégories d'une variable catégorielle (c'est-à-dire, une variable qui prend des valeurs discrètes ou des catégories). Voici comment cette fonction fonctionne :

1. Agrégation des données :

    - La fonction commence par regrouper les données du DataFrame df en fonction de la variable catégorielle `cat_var`. Elle compte également le nombre de défauts de paiement (`default_payment_next_month`) pour chaque catégorie en utilisant `value_counts(normalize=True)`. L'option `normalize=True` permet d'obtenir les pourcentages au lieu des comptages bruts.

2. Création du graphique à barres empilées :

    - Une fois que les données sont agrégées, la fonction utilise la méthode plot pour générer un graphique à barres empilées (`kind="barh"`, `stacked=True`). Le graphique à barres empilées est utilisé pour montrer la répartition des défauts de paiement dans chaque catégorie de `cat_var`.

3. Titre et légende :

    - La fonction ajoute un titre au graphique en utilisant `ax.set_title`. Le titre est défini comme "Pourcentage de défauts de paiement" avec une police de taille 16.

    - Elle ajoute également une légende au graphique pour indiquer les catégories de défaut de paiement (par exemple, "Défaut de paiement" ou "Pas de défaut de paiement"). La légende est placée en dehors du graphique à l'aide de `bbox_to_anchor=(1,1)`.

4. Affichage du graphique :

    - Enfin, la fonction utilise `plt.show()` pour afficher le graphique.

L'objectif de cette fonction est d'illustrer visuellement comment les défauts de paiement sont répartis dans différentes catégories d'une variable catégorielle. Les barres empilées permettent de comparer les pourcentages de défauts de paiement pour chaque catégorie et d'identifier rapidement les tendances ou les différences.


***Explication de la fonction performance_evaluation_report*** :

La fonction `performance_evaluation_report` est conçue pour évaluer les performances d'un modèle de classification en générant un rapport qui inclut plusieurs métriques d'évaluation. Voici comment cette fonction fonctionne :

1. Prédiction :

    - La fonction commence par utiliser le modèle donné pour prédire les étiquettes de classe sur l'ensemble de test `X_test`. Elle stocke les prédictions dans `y_pred` et les probabilités de classe positives (si disponibles) dans `y_pred_prob`.

2. Matrices de confusion et métriques :

    - La fonction calcule la matrice de confusion à partir des étiquettes réelles y_test et des prédictions `y_pred`. La matrice de confusion est décomposée en quatre valeurs : vrais positifs (tp), faux positifs (fp), vrais négatifs (tn) et faux négatifs (fn).

3. Courbe ROC et métriques associées :

    - La fonction calcule la courbe ROC (Receiver Operating Characteristic) en utilisant les taux de faux positifs (fpr) et les taux de vrais positifs (tpr) calculés à partir des probabilités de classe positives et des étiquettes réelles. Elle calcule également la valeur de l'aire sous la courbe ROC (ROC-AUC).

4. Courbe PR et métriques associées (optionnel) :

    - Si l'option `show_pr_curve` est activée, la fonction calcule la courbe de précision-rappel (Precision-Recall) en utilisant la précision et le rappel calculés à partir des probabilités de classe positives et des étiquettes réelles. Elle calcule également la valeur de l'aire sous la courbe PR (PR-AUC).

5. Création du graphique (optionnel) :

    - Si l'option `show_plot` est activée, la fonction génère un graphique avec jusqu'à trois sous-graphiques, en fonction des options `show_pr_curve`. Les sous-graphiques sont les suivants :

        - Matrice de confusion avec des étiquettes des axes des x et des y, ainsi que des pourcentages dans les cellules.

        - Courbe ROC avec le ROC-AUC, un point de décision et la ligne de référence (diagonale).

        - Courbe PR avec le PR-AUC.

6. Métriques de performance :

    - La fonction calcule plusieurs métriques de performance, notamment l'exactitude (accuracy), la précision (precision), le rappel (recall), la spécificité (specificity), le score F1 (f1_score), le coefficient kappa de Cohen (cohens_kappa), le coefficient de corrélation de Matthews (matthews_corr_coeff), l'aire sous la courbe ROC (roc_auc), l'aire sous la courbe PR (pr_auc) et la précision moyenne (average_precision).

7. Renvoi des métriques :

    - La fonction renvoie un objet de type `pd.Series` contenant toutes les métriques calculées. Cette série est structurée de manière à fournir un aperçu complet des performances du modèle.

En résumé, la fonction `performance_evaluation_report` effectue des prédictions, calcule diverses métriques de performance et génère un rapport visuel sous forme de graphique (si demandé). Cela permet d'obtenir une évaluation complète des performances du modèle de classification.


🖥 **Script app.py de l'application web pour la prédiction automatique du défaut de paiement**

```python

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Charger le modèle pré-entraîné
model = joblib.load("classifier_gs_model.pkl")

# Définir les catégories pour le diagramme
categories = ['Pas de défaut de paiement', 'Défaut de paiement']

# Fonction pour faire des prédictions
def make_prediction(features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    probability = np.round(probability * 100, 2)
    return prediction, probability

# Interface utilisateur de l'application
st.title("Application de Prédiction de Défaut de Paiement")

st.write("Cette application utilise un modèle de Machine Learning pour prédire si un client sera en défaut de paiement ou non en fonction des caractéristiques fournies.")

st.sidebar.header("Informations sur le client")

# Saisie des caractéristiques du client
limit_bal = st.sidebar.number_input("Montant du crédit", min_value=0, value=50000)
sex = st.sidebar.selectbox("Sexe du client", ['Female', 'Male'])
education = st.sidebar.selectbox("Niveau d'éducation du client", ['Graduate school', 'University', 'High school', 'Others'])
marriage = st.sidebar.selectbox("Statut matrimonial du client", ['Single', 'Married', 'Others'])
age = st.sidebar.number_input("Âge du client", min_value=18, max_value=100, value=30)

# Saisie des statuts de paiement
payment_status_sep = st.sidebar.selectbox("Statut de paiement en septembre", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_aug = st.sidebar.selectbox("Statut de paiement en août", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_jul = st.sidebar.selectbox("Statut de paiement en juillet", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_jun = st.sidebar.selectbox("Statut de paiement en juin", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_may = st.sidebar.selectbox("Statut de paiement en mai", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_apr = st.sidebar.selectbox("Statut de paiement en avril", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)

# Saisie des relevés de facturation
bill_statement_sep = st.sidebar.number_input("Relevé de facturation en septembre", min_value=0, value=0)
bill_statement_aug = st.sidebar.number_input("Relevé de facturation en août", min_value=0, value=0)
bill_statement_jul = st.sidebar.number_input("Relevé de facturation en juillet", min_value=0, value=0)
bill_statement_jun = st.sidebar.number_input("Relevé de facturation en juin", min_value=0, value=0)
bill_statement_may = st.sidebar.number_input("Relevé de facturation en mai", min_value=0, value=0)
bill_statement_apr = st.sidebar.number_input("Relevé de facturation en avril", min_value=0, value=0)

# Saisie des paiements précédents
previous_payment_sep = st.sidebar.number_input("Paiement précédent en septembre", min_value=0, value=0)
previous_payment_aug = st.sidebar.number_input("Paiement précédent en août", min_value=0, value=0)
previous_payment_jul = st.sidebar.number_input("Paiement précédent en juillet", min_value=0, value=0)
previous_payment_jun = st.sidebar.number_input("Paiement précédent en juin", min_value=0, value=0)
previous_payment_may = st.sidebar.number_input("Paiement précédent en mai", min_value=0, value=0)
previous_payment_apr = st.sidebar.number_input("Paiement précédent en avril", min_value=0, value=0)

# Créer un DataFrame à partir des caractéristiques
input_data = pd.DataFrame({
    'limit_bal': [limit_bal],
    'sex': [sex],
    'education': [education],
    'marriage': [marriage],
    'age': [age],
    'payment_status_sep': [payment_status_sep],
    'payment_status_aug': [payment_status_aug],
    'payment_status_jul': [payment_status_jul],
    'payment_status_jun': [payment_status_jun],
    'payment_status_may': [payment_status_may],
    'payment_status_apr': [payment_status_apr],
    'bill_statement_sep': [bill_statement_sep],
    'bill_statement_aug': [bill_statement_aug],
    'bill_statement_jul': [bill_statement_jul],
    'bill_statement_jun': [bill_statement_jun],
    'bill_statement_may': [bill_statement_may],
    'bill_statement_apr': [bill_statement_apr],
    'previous_payment_sep': [previous_payment_sep],
    'previous_payment_aug': [previous_payment_aug],
    'previous_payment_jul': [previous_payment_jul],
    'previous_payment_jun': [previous_payment_jun],
    'previous_payment_may': [previous_payment_may],
    'previous_payment_apr': [previous_payment_apr]
})

# Prédiction
if st.sidebar.button("Prédire"):
    prediction, probability = make_prediction(input_data)
    st.subheader("Probabilités :")
    prob_df = pd.DataFrame({'Catégories': categories, 'Probabilité': probability[0]})
    fig = px.bar(prob_df, x='Catégories', y='Probabilité', text='Probabilité', labels={'Probabilité': 'Probabilité (%)'})
    st.plotly_chart(fig)

    st.subheader("Résultat de la prédiction :")
    if prediction[0] == 1:
        st.error("Le client sera en défaut de paiement.")
    else:
        st.success("Le client ne sera pas en défaut de paiement.")

```

Ce code Python utilise Streamlit pour créer une interface utilisateur interactive qui permet de prédire le défaut de paiement d'un client en utilisant un modèle de machine learning pré-entrainé. Voici une explication détaillée de chaque section du code :

- **Importation des Bibliothèques** :

```python
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
```
Ces lignes importent les bibliothèques nécessaires :
- `streamlit` pour créer l'interface web interactive.
- `joblib` pour charger le modèle de machine learning pré-entrainé.
- `pandas` pour manipuler les données sous forme de DataFrame.
- `numpy` pour effectuer des calculs numériques.
- `plotly.express` pour créer des visualisations interactives.

- **Charger le Modèle Pré-entraîné** :

```python
model = joblib.load("classifier_gs_model.pkl")
```
Cette ligne charge un modèle de machine learning pré-entraîné à partir d'un fichier nommé `classifier_gs_model.pkl`.

- **Définir les Catégories pour le Diagramme** :

```python
categories = ['Pas de défaut de paiement', 'Défaut de paiement']
```
Cette ligne définit les catégories de sortie du modèle, à savoir s'il y a ou non un défaut de paiement.

- **Fonction pour Faire des Prédictions** :

```python
def make_prediction(features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    probability = np.round(probability * 100, 2)
    return prediction, probability
```
Cette fonction prend en entrée les caractéristiques du client, fait une prédiction en utilisant le modèle, calcule les probabilités associées à chaque catégorie, et retourne la prédiction ainsi que les probabilités.

- **Interface Utilisateur de l'Application** :

```python
st.title("Application de Prédiction de Défaut de Paiement")
st.write("Cette application utilise un modèle de Machine Learning pour prédire si un client sera en défaut de paiement ou non en fonction des caractéristiques fournies.")
```
Ces lignes définissent le titre et une description de l'application.

- **Saisie des Caractéristiques du Client** :

```python
st.sidebar.header("Informations sur le client")
limit_bal = st.sidebar.number_input("Montant du crédit", min_value=0, value=50000)
sex = st.sidebar.selectbox("Sexe du client", ['Female', 'Male'])
education = st.sidebar.selectbox("Niveau d'éducation du client", ['Graduate school', 'University', 'High school', 'Others'])
marriage = st.sidebar.selectbox("Statut matrimonial du client", ['Single', 'Married', 'Others'])
age = st.sidebar.number_input("Âge du client", min_value=18, max_value=100, value=30)
```
Ces lignes créent des widgets dans la barre latérale pour entrer les informations personnelles du client, comme le montant du crédit, le sexe, le niveau d'éducation, le statut matrimonial et l'âge.

- **Saisie des Statuts de Paiement** :

```python
payment_status_sep = st.sidebar.selectbox("Statut de paiement en septembre", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_aug = st.sidebar.selectbox("Statut de paiement en août", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_jul = st.sidebar.selectbox("Statut de paiement en juillet", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_jun = st.sidebar.selectbox("Statut de paiement en juin", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_may = st.sidebar.selectbox("Statut de paiement en mai", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_apr = st.sidebar.selectbox("Statut de paiement en avril", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
```
Ces lignes permettent de saisir les statuts de paiement du client pour les six derniers mois.

- **Saisie des Relevés de Facturation et des Paiements Précédents** :

```python
bill_statement_sep = st.sidebar.number_input("Relevé de facturation en septembre", min_value=0, value=0)
bill_statement_aug = st.sidebar.number_input("Relevé de facturation en août", min_value=0, value=0)
bill_statement_jul = st.sidebar.number_input("Relevé de facturation en juillet", min_value=0, value=0)
bill_statement_jun = st.sidebar.number_input("Relevé de facturation en juin", min_value=0, value=0)
bill_statement_may = st.sidebar.number_input("Relevé de facturation en mai", min_value=0, value=0)
bill_statement_apr = st.sidebar.number_input("Relevé de facturation en avril", min_value=0, value=0)

previous_payment_sep = st.sidebar.number_input("Paiement précédent en septembre", min_value=0, value=0)
previous_payment_aug = st.sidebar.number_input("Paiement précédent en août", min_value=0, value=0)
previous_payment_jul = st.sidebar.number_input("Paiement précédent en juillet", min_value=0, value=0)
previous_payment_jun = st.sidebar.number_input("Paiement précédent en juin", min_value=0, value=0)
previous_payment_may = st.sidebar.number_input("Paiement précédent en mai", min_value=0, value=0)
previous_payment_apr = st.sidebar.number_input("Paiement précédent en avril", min_value=0, value=0)
```
Ces lignes permettent de saisir les relevés de facturation et les paiements précédents pour les six derniers mois.

- **Créer un DataFrame à partir des Caractéristiques** :

```python
input_data = pd.DataFrame({
    'limit_bal': [limit_bal],
    'sex': [sex],
    'education': [education],
    'marriage': [marriage],
    'age': [age],
    'payment_status_sep': [payment_status_sep],
    'payment_status_aug': [payment_status_aug],
    'payment_status_jul': [payment_status_jul],
    'payment_status_jun': [payment_status_jun],
    'payment_status_may': [payment_status_may],
    'payment_status_apr': [payment_status_apr],
    'bill_statement_sep': [bill_statement_sep],
    'bill_statement_aug': [bill_statement_aug],
    'bill_statement_jul': [bill_statement_jul],
    'bill_statement_jun': [bill_statement_jun],
    'bill_statement_may': [bill_statement_may],
    'bill_statement_apr': [bill_statement_apr],
    'previous_payment_sep': [previous_payment_sep],
    'previous_payment_aug': [previous_payment_aug],
    'previous_payment_jul': [previous_payment_jul],
    'previous_payment_jun': [previous_payment_jun],
    'previous_payment_may': [previous_payment_may],
    'previous_payment_apr': [previous_payment_apr]
})
```
Cette section crée un DataFrame `input_data` contenant les caractéristiques saisies par l'utilisateur.

- **Prédiction** :

```python
if st.sidebar.button("Prédire"):
    prediction, probability = make_prediction(input_data)
    st.subheader("Probabilités :")
    prob_df = pd.DataFrame({'Catégories': categories, 'Probabilité': probability[0]})
    fig = px.bar(prob_df, x='Catégories', y='Probabilité', text='Probabilité', labels={'Probabilité': 'Probabilité (%)'})
    st.plotly_chart(fig)

    st.subheader("Résultat de la prédiction :")
    if prediction[0] == 1:
        st.error("Le client sera en défaut de paiement.")
    else:
        st.success("Le client ne sera pas en défaut de paiement.")
```
Lorsque l'utilisateur clique sur le bouton "Prédire" :
1. Les caractéristiques du client sont envoyées à la fonction `make_prediction` pour obtenir une prédiction et les probabilités.
2. Les probabilités sont affichées sous forme de graphique à barres.
3. Le résultat de la prédiction est affiché sous forme de message de succès ou d'erreur en fonction de la prédiction du modèle.

Le code permet à un utilisateur d'entrer diverses caractéristiques liées à un client et d'obtenir une prédiction sur la probabilité de défaut de paiement en utilisant un modèle de machine learning pré-entraîné. L'interface utilisateur est créée avec Streamlit, rendant le processus interactif et accessible même à des utilisateurs non techniques.

Après avoir créé votre application web, vous pouvez maintenant le déployer en ligne pour que les utilisateurs finaux puissent l'utiliser.