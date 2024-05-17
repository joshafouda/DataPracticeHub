# Projet : Utilisation de modèles pré-entrainés de Deep Learning pour la classification de textes

## Description
Le projet actuel vise à développer un système de classification de texte robuste et précis en exploitant les modèles pré-entraînés proposés par Hugging Face (https://huggingface.co/), une entreprise axée sur l'Intelligence Artificielle (AI) dont l'objectif est de faire progresser et de démocratiser l'IA. La classification de texte est une tâche fondamentale dans le domaine du traitement automatique du langage naturel, permettant d'attribuer des étiquettes ou des catégories à des documents textuels afin de faciliter leur gestion, leur recherche, ou leur analyse.

## Image
imgs/project3/project3.png

## Instructions

1. Installation des Packages

Avant de commencer le projet, vous devez installer les packages nécessaires :

```bash
pip install torch
pip install transformers
```

2. Exploration et Chargement des Modèles Pré-entrainés

- **Hugging Face** : Familiarisez-vous avec les modèles pré-entraînés disponibles sur [Hugging Face](https://huggingface.co/models). Choisissez un modèle adapté à la tâche de classification de texte.

- **Importez le modèle** : Utilisez la bibliothèque `transformers` pour importer et charger le modèle pré-entraîné que vous avez choisi.

3. Implémentation Technique

Travaillez dans un notebook pour vous bien travailler votre code.

4. Création d'une Application Web avec Streamlit

Utilisez Streamlit pour créer une interface utilisateur simple où l'utilisateur peut saisir du texte et obtenir une analyse de sentiment.

5. Documentation et Partage

- **Documentez le Projet** : Rédigez une documentation complète du projet, décrivant les étapes suivies, les choix technologiques, les tests réalisés et les résultats obtenus.

- **Partagez le Projet** : Publiez votre code sur un dépôt GitHub et partagez l'URL de l'application déployée pour que d'autres puissent la tester et l'utiliser.

En suivant ces instructions, vous développerez un système de classification de texte efficace et robuste, intégré dans une application web interactive, en exploitant les modèles pré-entraînés de Hugging Face et la simplicité de Streamlit pour l'interface utilisateur.

## Resources
- [Analyse de sentiments avec un modèle Machine Learning pré-entraîné de Hugging Face](https://youtu.be/47luwoizXKs)
- [Comment déployer une web app Streamlit](https://youtu.be/wjRlWuXmlvw)
- [Documentation Streamlit](https://docs.streamlit.io/)
- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)

## Execution du Projet

Vous devez installer ces deux packages :

```bash
pip install torch
```

```bash
pip install transformers
```

L'intérêt majeur de l'utilisation des modèles pré-entraînés comme ceux de Hugging Face réside dans leur capacité à capturer des structures et des motifs linguistiques complexes à partir de vastes ensembles de données. Ces modèles ont été pré-entraînés sur des corpus massifs, généralement en utilisant des techniques de deep learning telles que les transformers. Le transfert de connaissances de ces modèles pré-entraînés vers des tâches spécifiques, telles que la classification de texte, offre plusieurs avantages :

1. **Gain de temps et d'effort :** Les modèles pré-entraînés ont déjà appris des représentations utiles du langage à partir de grandes quantités de données. Cela évite la nécessité de créer un modèle à partir de zéro pour chaque tâche, ce qui économise considérablement du temps et des ressources.

2. **Performances améliorées :** En exploitant des modèles pré-entraînés, on bénéficie de la capacité du modèle à comprendre les nuances du langage, y compris les subtilités sémantiques et grammaticales. Cela conduit souvent à des performances supérieures par rapport à des modèles créés spécifiquement pour une tâche particulière.

3. **Adaptabilité :** Les modèles pré-entraînés peuvent être fine-tunés sur des tâches spécifiques, ce qui permet de personnaliser le modèle pour s'adapter aux besoins particuliers du projet. Cela offre une flexibilité considérable sans sacrifier les avantages du pré-entraînement initial.

4. **Évolution continue :** Hugging Face et d'autres entreprises similaires mettent régulièrement à jour leurs modèles pré-entraînés en les entraînant sur des corpus de données plus récents. Cela garantit que le modèle continue de bénéficier des avancées dans le langage naturel, sans nécessiter une mise à jour manuelle constante.

En résumé, l'utilisation de modèles pré-entraînés offre une approche puissante et efficace pour aborder la tâche de classification de texte, en exploitant la richesse des informations apprises à partir de vastes ensembles de données textuelles.

- **Applications de la clasification automatique de texte et Objectif du Projet**

La classification automatique de texte trouve de nombreuses applications dans le monde réel, contribuant à automatiser et à améliorer divers aspects des processus informatiques. Voici quelques-unes des applications les plus courantes :

1. **Analyse des sentiments :** La classification de texte est souvent utilisée pour déterminer les sentiments exprimés dans un texte, que ce soit sur les réseaux sociaux, les critiques de produits, ou les commentaires en ligne. Cela permet aux entreprises de comprendre les opinions des utilisateurs et d'ajuster leurs stratégies en conséquence.

2. **Filtrage de spam :** Les systèmes de filtrage de spam utilisent des techniques de classification de texte pour identifier et bloquer les messages indésirables, tels que les e-mails de spam. Cela permet de maintenir la propreté des boîtes de réception et d'améliorer l'efficacité de la communication électronique.

3. **Catégorisation de documents :** Dans le domaine de la gestion de l'information, la classification de texte est utilisée pour organiser automatiquement de grands ensembles de documents en catégories spécifiques. Cela facilite la recherche, la navigation et la récupération d'informations.

4. **Assistance à la clientèle automatisée :** Les chatbots et les systèmes de réponse automatique utilisent la classification de texte pour comprendre les requêtes des utilisateurs et fournir des réponses appropriées. Cela permet une assistance clientèle rapide et efficace, en traitant automatiquement des questions courantes.

5. **Détection de la fraude :** Dans les secteurs financiers, la classification de texte peut être utilisée pour détecter les activités frauduleuses en analysant les transactions et les communications associées. Cela contribue à renforcer la sécurité des transactions.

6. **Catégorisation de contenu :** Les plates-formes de médias sociaux, les sites de partage de contenu et les moteurs de recherche utilisent la classification de texte pour catégoriser et organiser les articles, images et vidéos, facilitant ainsi la découverte de contenu pertinent.

7. **Reconnaissance d'entités nommées (NER) :** La classification de texte est souvent utilisée pour extraire et classer automatiquement des entités spécifiques telles que les noms de personnes, d'organisations, de lieux, etc. Cela est utile dans des domaines tels que l'indexation de documents et l'analyse de données.

8. **Diagnostic médical :** Dans le domaine de la santé, la classification de texte peut être utilisée pour analyser des rapports médicaux et aider au diagnostic en classifiant automatiquement les descriptions de symptômes, de maladies et de traitements.

Ces applications démontrent la polyvalence de la classification automatique de texte et son impact significatif sur l'efficacité des systèmes automatisés dans divers domaines.

Dans ce projet, vous apprendrez à intégrer un modèle pré-entrainé de classification de texte dans une application web permettant aux utilsateurs de prédire le sentiment d'un texte.

- **Implémentation technique du Projet**


```python
from transformers import pipeline
```


Le code que vous avez fourni importe un module appelé "pipeline" du package "transformers". 

La bibliothèque "transformers" de Hugging Face est une bibliothèque populaire pour travailler avec des modèles de traitement automatique du langage naturel (TALN) pré-entraînés, tels que BERT, GPT, etc. Le module "pipeline" de cette bibliothèque simplifie l'utilisation de ces modèles pour des tâches spécifiques.

Voici une explication plus détaillée du code :

1. **`from transformers import pipeline` :**
   - Cette ligne importe la classe `pipeline` du module `transformers`.

2. **Utilisation du module `pipeline` :**
   - Une fois que le module est importé, on peut créer un objet pipeline pour effectuer différentes tâches de traitement du langage naturel sans avoir à configurer manuellement un modèle pré-entraîné.

En résumé, le code que vous avez fourni permet d'utiliser facilement les modèles pré-entraînés de Hugging Face pour effectuer différentes tâches de TALN en utilisant la classe `pipeline`.

Voici un exemple typique d'utilisation de la classe `pipeline` pour une tâche d'analyse de sentiments :


```python
# Charger le modèle d'analyse des sentiments par défaut de Hugging Face
model = pipeline('sentiment-analysis')
```


Ce code utilise la bibliothèque Hugging Face Transformers pour créer un objet pipeline spécifique à la tâche de l'analyse de sentiment. Voici une explication détaillée du code :

1. **`model = pipeline('sentiment-analysis')` :**
   - Cette ligne crée un objet pipeline spécifique à la tâche d'analyse de sentiment. L'argument `'sentiment-analysis'` indique que le pipeline doit être configuré pour effectuer une tâche d'analyse de sentiment.

   - L'objet `model` ainsi créé est une instance de la classe `pipeline` configurée pour l'analyse de sentiment. Cela signifie que vous pouvez utiliser cet objet `model` pour effectuer rapidement des analyses de sentiment sur différents textes sans avoir à gérer manuellement le modèle pré-entraîné sous-jacent.


```python
# Utilisation du modèle
text = "Good book. Good explanation. Who want to learn python tkinter, this book is best choice."
result = model(text)
print(type(result))
result
```

    <class 'list'>





    [{'label': 'POSITIVE', 'score': 0.9997617602348328}]



Ce code utilise un modèle pré-entraîné pour effectuer une analyse de sentiment sur un texte spécifique. Voici une explication détaillée :

1. **`text = "Good book. Good explanation. Who want to learn python tkinter, this book is best choice."` :**
   - Cette ligne crée une variable `text` qui contient le texte sur lequel l'analyse de sentiment sera effectuée. Le texte semble être une évaluation positive d'un livre et de son contenu, avec une recommandation pour ceux qui veulent apprendre Python avec Tkinter.

2. **`result = model(text)` :**
   - Cette ligne utilise l'objet `model` (probablement créé avec `pipeline('sentiment-analysis')`, comme mentionné précédemment) pour effectuer une analyse de sentiment sur le texte spécifié.

   - `text` est le texte sur lequel l'analyse de sentiment est appliquée.

   - Le résultat de l'analyse de sentiment est stocké dans la variable `result`. Le format du résultat dépend du modèle utilisé, mais généralement, il contient des informations telles que le label du sentiment (par exemple, 'POSITIVE', 'NEGATIVE', 'NEUTRAL') et peut inclure un score ou une probabilité associé à ce label.

En résumé, ce code applique un modèle pré-entraîné pour effectuer une analyse de sentiment sur un texte donné, stocke le résultat dans la variable `result`, et ce résultat pourrait ensuite être utilisé pour obtenir des informations sur le sentiment exprimé dans le texte.


```python
result[0]['label']
```




    'POSITIVE'




```python
result[0]['score']
```




    0.9997617602348328




```python
# Autre exemple d'utilisation du modèle
text = "Bad product. It worked for 1 month then trash, it's shameful"
result = model(text)
result
```




    [{'label': 'NEGATIVE', 'score': 0.999680757522583}]




```python
# Autre exemple d'utilisation du modèle
text = "Very nice product, but no longer works after 8 days. Changed batteries and worked again for a few hours."
result = model(text)
result
```




    [{'label': 'NEGATIVE', 'score': 0.9868576526641846}]



Essayons les mêmes textes présédents en Français :


```python
# Autre exemple d'utilisation du modèle
text = "Bon bouquin. Bonne explication. Qui veut apprendre Python Tkinter, ce livre est le meilleur choix."
result = model(text)
result
```




    [{'label': 'NEGATIVE', 'score': 0.9252129793167114}]



Ce résultat montre que le modèle fonctionne bien uniquement pour des textes en Anglais car le texte présenté est à sentiment positif mais le modèle dit que c'est négatif.


```python
text = "Mauvais produit. Ça a fonctionné 1 mois puis poubelle, c'est honteux"
result = model(text)
result
```




    [{'label': 'POSITIVE', 'score': 0.5270236134529114}]



Le résultat ci-dessous confirme le fait que ce modèle fonctionne bien uniquement pour les textes en anglais.

Il doit certainement avoir d'autres modèles pré-entrainés d'analyse de sentiments dans Hugging Face qui prennent en compte plusieurs langues. Voir dans la bibliothèque des modèles : https://huggingface.co/models

🖥 **Création d'une web app Streamlit d'analyse de sentiments**

Ci-dessous, un code simple permettant d'intégrer facilement un modèle de Hugging Face pour l'analyse de sentiments de textes saisis par l'utilsateur :

````python
# Code de l'application :
import streamlit as st 
from utils import get_sentiment_model

st.title("Analyse de sentiments à l'aide d'un modèle pré-entraîné de Hugging Face")
text = st.text_input("Saisissez le texte (en anglais) à analyser")

# Chargement du modèle pré-entrainé
model = get_sentiment_model()

if text:
    result = model(text)
    st.write("Sentiment :", result[0]['label'])
    st.write("Score :", result[0]['score'])
```

Ce code est une application simple construite avec Streamlit, une bibliothèque Python utilisée pour créer des applications web interactives pour l'analyse de données. L'application utilise un modèle pré-entraîné de Hugging Face pour effectuer une analyse de sentiments sur un texte en anglais. Voici une explication détaillée du code :

1. **Import des bibliothèques :**
   - `import streamlit as st` : Importe la bibliothèque Streamlit et la renomme en `st` pour faciliter son utilisation.
   - `from utils import get_sentiment_model` : Importe une fonction `get_sentiment_model` depuis un fichier appelé `utils`. Cette fonction est utilisée pour charger le modèle pré-entraîné.

2. **Définition de l'interface utilisateur avec Streamlit :**
   - `st.title("Analyse de sentiments à l'aide d'un modèle pré-entraîné de Hugging Face")` : Ajoute un titre à l'application web.

   - `text = st.text_input("Saisissez le texte (en anglais) à analyser")` : Crée un champ de saisie de texte où l'utilisateur peut entrer le texte à analyser. La saisie est stockée dans la variable `text`.

3. **Chargement du modèle pré-entraîné :**
   - `model = get_sentiment_model()` : Appelle la fonction `get_sentiment_model` pour charger le modèle pré-entraîné. Cette fonction retourne un objet modèle qui peut être utilisé pour effectuer une analyse de sentiment (explication détaillée de la fonction ci-dessous)

4. **Analyse de sentiment :**
   - `if text:` : Vérifie si l'utilisateur a saisi du texte.

      - `result = model(text)` : Si du texte a été saisi, utilise le modèle pré-entraîné pour effectuer une analyse de sentiment sur le texte. Le résultat est stocké dans la variable `result`.

      - `st.write("Sentiment :", result[0]['label'])` : Affiche le label du sentiment obtenu à partir du résultat de l'analyse. Par exemple, cela pourrait être "Positif" ou "Négatif".

      - `st.write("Score :", result[0]['score'])` : Affiche le score ou la probabilité associé au label du sentiment. Cela peut représenter à quel point le modèle est confiant dans la classification du sentiment.

L'application Streamlit fournit une interface utilisateur simple où l'utilisateur peut saisir du texte, et après avoir cliqué sur un bouton ou appuyé sur "Enter", l'application utilise un modèle pré-entraîné pour analyser le sentiment du texte et affiche le résultat.

```python
# Code de la fonction get_sentiment_model
@st.cache_resource()
def get_sentiment_model():
    return pipeline('sentiment-analysis')
```

Ce code utilise le décorateur `@st.cache_resource()` fourni par Streamlit pour créer un cache de ressources autour de la fonction `get_sentiment_model()`. Cela signifie que le résultat retourné par cette fonction sera mis en cache, ce qui peut être utile pour éviter de charger le modèle à chaque fois que la fonction est appelée. Voici une explication détaillée :

1. **`@st.cache_resource()` :**
   - C'est un décorateur fourni par Streamlit qui indique à l'application de mettre en cache le résultat de la fonction décorée. Le cache est géré automatiquement par Streamlit.

2. **Définition de la fonction `get_sentiment_model()` :**
   - `def get_sentiment_model():` : Déclare une fonction nommée `get_sentiment_model` qui ne prend aucun argument.

   - `return pipeline('sentiment-analysis')` : Retourne un objet pipeline configuré pour effectuer une tâche d'analyse de sentiment. Il utilise probablement la bibliothèque Hugging Face Transformers pour créer ce pipeline, comme indiqué par le code `pipeline('sentiment-analysis')`.

3. **Mise en cache du résultat :**
   - Grâce au décorateur `@st.cache_resource()`, le résultat retourné par la fonction `get_sentiment_model()` sera mis en cache. Cela signifie que la première fois que la fonction est appelée, le modèle sera chargé, et les appels ultérieurs utiliseront la version mise en cache du modèle sans avoir besoin de le recharger à chaque fois.

   - Le caching est particulièrement utile pour des opérations coûteuses en termes de temps, telles que le chargement de modèles pré-entraînés, afin d'améliorer les performances de l'application en évitant des opérations répétitives coûteuses.

En résumé, ce code définit une fonction `get_sentiment_model()` qui retourne un modèle pré-entraîné pour l'analyse de sentiment, et utilise le décorateur `@st.cache_resource()` pour mettre en cache le résultat de cette fonction, évitant ainsi de charger le modèle à chaque appel de la fonction dans le contexte d'une application Streamlit.

🖥 **Conclusion du Projet**

Dans le cadre de ce projet d'analyse de sentiments à l'aide de modèles pré-entraînés de Hugging Face et de l'interface utilisateur développée avec Streamlit, nous avons réussi à créer une application interactive permettant aux utilisateurs d'analyser le sentiment d'un texte en anglais. L'utilisation de modèles pré-entraînés a permis une mise en œuvre rapide et efficace de l'analyse de sentiments, offrant des résultats pertinents.

**Pistes d'Amélioration :**

1. **Multilinguisme :** Élargir la prise en charge à plusieurs langues en intégrant des modèles pré-entraînés spécifiques à chaque langue. Cela améliorerait l'utilité de l'application pour un public plus diversifié.

2. **Interface Utilisateur Améliorée :** Enrichir l'interface utilisateur en ajoutant des fonctionnalités telles que la possibilité de visualiser des statistiques sur les sentiments analysés, d'explorer les résultats dans un format plus convivial, ou même d'intégrer des graphiques interactifs.

3. **Personnalisation du Modèle :** Permettre à l'utilisateur de personnaliser le modèle en ajustant certains paramètres ou en sélectionnant des variantes de modèles pré-entraînés pour mieux répondre à des besoins spécifiques.

4. **Évaluation de la Confiance :** Ajouter une fonctionnalité d'évaluation de la confiance, montrant aux utilisateurs à quel point le modèle est confiant dans sa prédiction. Cela pourrait être utile pour des cas où l'analyse de sentiment doit être plus nuancée.

5. **Intégration de Retour Utilisateur :** Mettre en place un mécanisme de collecte de retours utilisateurs pour améliorer continuellement la précision et la pertinence des résultats fournis par le modèle.

6. **Optimisation des Performances :** Optimiser les performances de l'application en utilisant des techniques telles que le chargement asynchrone pour réduire les temps d'attente et améliorer l'expérience utilisateur.

7. **Sécurité des Données :** Renforcer les mécanismes de sécurité pour garantir la confidentialité des données saisies par les utilisateurs, en particulier si l'application est déployée en ligne.

En explorant ces pistes d'amélioration, nous pourrions augmenter la robustesse, l'utilité et la convivialité de l'application d'analyse de sentiments, offrant ainsi une meilleure expérience utilisateur et des résultats plus fiables pour différentes utilisations. Ce projet représente un point de départ solide, et son évolution peut être guidée par les besoins spécifiques du public cible et les évolutions futures dans le domaine du traitement automatique du langage naturel.

🖥 **Fichier utils.py**

```python
# Librairies
import streamlit as st 
from transformers import pipeline

@st.cache_resource()
def get_sentiment_model():
    return pipeline('sentiment-analysis')
```