# Projet : Construction d'un Pipeline ETL avec PySpark pour l'analyse des trajets en Taxi

## Description

Le secteur des transports, en particulier les services de taxi, constitue un élément crucial de l'infrastructure urbaine de New York. Avec des millions de voyages effectués chaque année, les données générées par ces voyages fournissent une mine d'informations précieuses. En tant que Data Scientist nouvellement embauché dans une société d'analyse des transport, votre Manager vous demande de collecter des données sur les trajets en taxi de la ville de New York à partir de la source officielle fournie par la *NYC Taxi and Limousine Commission (TLC)*. Une fois les données collectés, vous devez construire un pipeline de données qui permettra d'exécuter un processus ETL (Extract, Tranform, Load) dans le but ultime de charger les données nettoyées/tranformées dans un fichier PARQUET afin des les utliser pour de futures analyses.

## Image
imgs/project6/project6.png

## Instructions

1. **Téléchargement des fichiers de données sur les trajets en taxi jaune (Yellow Taxi Trip Records)** : Ces fichiers sont actuellement disponibles au format PARQUET sur le site de la [NYC Taxi and Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Vous devez écrire un script python *download_data_files.py* qui permettra d'effectuer automatiquement ce téléchargement sur la période d'étude souhaitée (par exemple Janvier 2021 à Février 2024). Cela dépendra aussi de la puissance de votre machine. Vous pouvez choisir une période beaucoup plus courte si nécessaire. L'exécution de ce script aura pour sortie un dossier nommé "histo_data_files" qui contiendra tous les fichiers PARQUET téléchargés.

2. **Ecriture du Script Python qui contiendra les fonctions d'extraction, de transformation et de chargement des données** : Ce script nommé *etl_functions.py* devra contenir 3 fonctions : 

    - extract : pour extraire les données dún fichier PARQUET et retourner une dataframe PySpark

    - transform : pour filtrer une dataframe PySpark en supprimant les observations avec des valeurs manquantes ou pour lesquelles trip_distance, passager_count ou total_amount est inférieur ou égal à 0 ; Fusionner la DataFrame filtrée avec les informations de [zone](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) ; Concatener les informations de l'arrondissement pour les lieux de prise en charge (variable "puborough") et de dépôt (variable "doborough") ; supprimer les valeurs manquantes et retourner la dataframe ainsi transformée

    - load : pour sauvegarder une dataframe PySPark au format PARQUET

Les codes de toutes les fonctions de ce projet doivent être écrites dans des blocs try ... except en y intégrand des messages de logging pour faciliter le débogage en cas de problème. Cela fait partie des bonnes pratiques de Software Engineering.

3. **Ecriture du Script Python qui permettra d'exécuter le processus ETL** : Ce script *etl_pipeline.py* fera appel au module *etl_functions.py* et son exécution aura comme résultat un fichier PARQUET des données transformées : "data_loaded/transformed_taxi_data.parquet".


## Resources

- [Source des données](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- [Données sur les zones de prise en charge et de dépose](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)

- [Comment automatiser le téléchargement des fichiers de données Big Data avec Python ?](https://youtu.be/6Mzmg4E78R0)

- [Tutoriel sur PySpark](https://youtu.be/QCuQzktfQV4)

- [𝐏𝐲𝐒𝐩𝐚𝐫𝐤 𝐞𝐧 𝐩𝐫𝐚𝐭𝐢𝐪𝐮𝐞 : 𝐂𝐚𝐬 𝐝'𝐮𝐬𝐚𝐠𝐞𝐬 𝐫𝐞́𝐞𝐥𝐬 𝐞𝐭 𝐞𝐱𝐞𝐦𝐩𝐥𝐞𝐬 𝐩𝐫𝐚𝐭𝐢𝐪𝐮𝐞𝐬 𝐞𝐧 𝐃𝐚𝐭𝐚 𝐒𝐜𝐢𝐞𝐧𝐜𝐞 𝐞𝐭 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 (Version Papier)](https://www.amazon.fr/gp/product/B0C9K6GTNH/ref=dbs_a_def_rwt_hsch_vamf_tkin_p1_i7)

- [𝐏𝐲𝐒𝐩𝐚𝐫𝐤 𝐞𝐧 𝐩𝐫𝐚𝐭𝐢𝐪𝐮𝐞 : 𝐂𝐚𝐬 𝐝'𝐮𝐬𝐚𝐠𝐞𝐬 𝐫𝐞́𝐞𝐥𝐬 𝐞𝐭 𝐞𝐱𝐞𝐦𝐩𝐥𝐞𝐬 𝐩𝐫𝐚𝐭𝐢𝐪𝐮𝐞𝐬 𝐞𝐧 𝐃𝐚𝐭𝐚 𝐒𝐜𝐢𝐞𝐧𝐜𝐞 𝐞𝐭 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 (Version PDF)]( https://afoudajosue.gumroad.com/l/yeatg)

- [Documentation sur l'installation de Spark](https://spark.apache.org/downloads.html)

- [Installation et Configuration d'un environnement Python avec VSC](https://youtu.be/6NYsMiFqH3E)


## Execution du Projet

🖥 **Téléchargement des données historiques : Script download_data_files.py**

```python
import os
import requests
import time
import hashlib
from datetime import datetime

def download_histo_data(path_to_histo_data_folder):
    """
    Downloads yellow taxi trip PARQUET files from 2021 to current year into the specified folder.

    Parameters:
    path_to_histo_data_folder (str): Path to the folder where files will be saved.

    Returns:
    None
    """
    try:
        # Create a folder to store downloaded files if it doesn't exist
        os.makedirs(path_to_histo_data_folder, exist_ok=True)

        # Current year
        current_year = datetime.now().year

        # Loop over years from current year to 2021
        for year in range(current_year, 2020, -1):
            for month in range(1, 13):
                # Construct download URL based on year and month
                download_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
                file_name = f"{path_to_histo_data_folder}/yellow_tripdata_{year}-{month:02d}.parquet"

                # Check if the file already exists
                if os.path.exists(file_name):
                    print(f"The file {file_name} already exists, skipping to the next file...")
                    continue

                # Download the file with error handling
                try:
                    print(f"Downloading {file_name}...")
                    response = requests.get(download_url, stream=True)
                    if response.status_code == 200:
                        with open(file_name, "wb") as f:
                            for chunk in response.iter_content(chunk_size=1024):
                                f.write(chunk)
                        print(f"{file_name} downloaded successfully!")
                    else:
                        print(f"Failed to download {file_name}. HTTP status code: {response.status_code}")
                except Exception as e:
                    print(f"An error occurred while downloading {file_name}: {str(e)}")

                # Pause for 1 second between each download to avoid overloading the remote server
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    print("Download completed!")

# Date of the day
print("Date of historical data download:", datetime.today())

# Path to the folder to save historical data
path_to_histo_data_folder = "histo_data_files"

# Call the function to download historical data
download_histo_data(path_to_histo_data_folder)
```

Le script `download_data_files.py` est conçu pour télécharger des fichiers de données de trajets en taxi jaune au format PARQUET depuis un serveur distant pour chaque mois de l'année, de 2021 jusqu'à l'année en cours, et les sauvegarder dans un dossier spécifié. Voici une explication détaillée de chaque partie du script :

- Importation des bibliothèques nécessaires

```python
import os
import requests
import time
from datetime import datetime
```

- `os` : pour des opérations sur le système de fichiers, comme créer des dossiers ou vérifier l'existence de fichiers.
- `requests` : pour faire des requêtes HTTP et télécharger des fichiers.
- `time` : pour introduire des pauses dans le script.
- `datetime` : pour travailler avec les dates et les heures.

- Définition de la fonction principale

```python
def download_histo_data(path_to_histo_data_folder):
```

Cette fonction, `download_histo_data`, télécharge les fichiers de données historiques de trajets en taxi jaune et les enregistre dans un dossier spécifié par le paramètre `path_to_histo_data_folder`.

- Création du dossier de destination

```python
    try:
        os.makedirs(path_to_histo_data_folder, exist_ok=True)
```

- `os.makedirs` : crée un dossier, ainsi que tous les dossiers parents nécessaires. Le paramètre `exist_ok=True` permet d'éviter une erreur si le dossier existe déjà.

- Détermination de l'année en cours

```python
        current_year = datetime.now().year
```

- `datetime.now().year` : obtient l'année actuelle.

- Boucle sur les années et les mois

```python
        for year in range(current_year, 2020, -1):
            for month in range(1, 13):
```

- `range(current_year, 2020, -1)` : crée une plage d'années allant de l'année en cours à 2021 (inclus).
- `range(1, 13)` : crée une plage de mois de janvier (1) à décembre (12).

- Construction de l'URL et du nom de fichier

```python
                download_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
                file_name = f"{path_to_histo_data_folder}/yellow_tripdata_{year}-{month:02d}.parquet"
```

- `f"{year}-{month:02d}"` : formatte l'année et le mois avec deux chiffres pour le mois (par exemple, `2021-01`).

- Vérification de l'existence du fichier

```python
                if os.path.exists(file_name):
                    print(f"The file {file_name} already exists, skipping to the next file...")
                    continue
```

- `os.path.exists(file_name)` : vérifie si le fichier existe déjà pour éviter de le télécharger à nouveau.

- Téléchargement du fichier

```python
                try:
                    print(f"Downloading {file_name}...")
                    response = requests.get(download_url, stream=True)
                    if response.status_code == 200:
                        with open(file_name, "wb") as f:
                            for chunk in response.iter_content(chunk_size=1024):
                                f.write(chunk)
                        print(f"{file_name} downloaded successfully!")
                    else:
                        print(f"Failed to download {file_name}. HTTP status code: {response.status_code}")
                except Exception as e:
                    print(f"An error occurred while downloading {file_name}: {str(e)}")
```

- `requests.get(download_url, stream=True)` : fait une requête HTTP pour télécharger le fichier. Le paramètre `stream=True` permet de télécharger le fichier par petits morceaux (chunks).
- `response.status_code` : vérifie si la requête a réussi (code 200).
- `with open(file_name, "wb") as f` : ouvre le fichier en mode binaire pour écrire (`"wb"`).
- `response.iter_content(chunk_size=1024)` : lit le contenu de la réponse par morceaux de 1024 octets pour économiser de la mémoire.
- `f.write(chunk)` : écrit chaque morceau dans le fichier.

- Pause entre les téléchargements

```python
                time.sleep(1)
```

- `time.sleep(1)` : attend 1 seconde entre chaque téléchargement pour éviter de surcharger le serveur.

- Gestion des erreurs globales

```python
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    print("Download completed!")
```

- `except Exception as e` : capture toute exception non prévue pendant l'exécution du code.
- `print(f"An error occurred: {str(e)}")` : affiche un message d'erreur.

- Exécution du script

```python
# Date of the day
print("Date of historical data download:", datetime.today())

# Path to the folder to save historical data
path_to_histo_data_folder = "histo_data_files"

# Call the function to download historical data
download_histo_data(path_to_histo_data_folder)
```

- `print("Date of historical data download:", datetime.today())` : affiche la date actuelle.
- `path_to_histo_data_folder = "histo_data_files"` : définit le chemin du dossier où les fichiers seront sauvegardés.
- `download_histo_data(path_to_histo_data_folder)` : appelle la fonction pour télécharger les données.

Ce script permet de s'assurer que toutes les données historiques de trajets en taxi jaune sont téléchargées et sauvegardées localement, tout en gérant les erreurs et en évitant de télécharger plusieurs fois les mêmes fichiers.

🖥 **Fonctions du processus ETL : Script etl_functions.py**

```python
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, LongType, DoubleType, StringType
from pyspark.sql import DataFrame
import pandas as pd

# Create a SparkSession
spark = SparkSession.builder \
    .appName("ETL Pipeline") \
    .getOrCreate()

# https://stackoverflow.com/questions/59096125/spark-2-4-parquet-column-cannot-be-converted-in-file-column-impressions-exp
spark.conf.set("spark.sql.parquet.enableVectorizedReader","false")

# Ajuster la configuration pour la représentation en chaîne des plans d'exécution
spark.conf.set("spark.sql.debug.maxToStringFields", "255")

def extract(file_path):
    """
    Extracts data from a Parquet file and returns a PySpark DataFrame.

    Parameters:
    file_path (str): The path to the Parquet file.

    Returns:
    pyspark.sql.DataFrame: DataFrame containing the data from the Parquet file.
    """
    try:
        # Initialize logging
        logging.basicConfig(filename='extract.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Reading Parquet file into DataFrame
        df = spark.read.parquet(file_path)

        logging.info("Parquet file extracted successfully.")

        return df
    except Exception as e:
        logging.error("An error occurred while extracting Parquet file: %s", str(e))
        return None


def transform(df):
    """
    Filters the DataFrame by removing observations with missing values or where trip_distance, passenger_count, or total_amount is less than or equal to 0.
    Merges the filtered DataFrame with zone information.
    Concatenates borough information for pickup and dropoff locations.
    
    Parameters:
    df (pyspark.sql.DataFrame): Input DataFrame.

    Returns:
    pyspark.sql.DataFrame: Transformed DataFrame.
    """
    try:
        # Initialize logging
        logging.basicConfig(filename='transform.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Renaming columns to lowercase
        for column_name in df.columns:
            df = df.withColumnRenamed(column_name, column_name.lower())

        # Dropping rows with missing values and filtering by specified conditions
        filtered_df = df.na.drop(subset=["passenger_count", "total_amount"]) \
            .filter((col("trip_distance") > 0) & (col("passenger_count") > 0) & (col("total_amount") > 0))

        # Read zone information 
        df_zones = pd.read_csv("fichier_des_zones.csv")

        # Select relevant columns from df_zones
        df_zones_subset = df_zones[['locationid', 'borough', 'zone', 'service_zone']]

        # Convert df_zones-subset to a Spark dataframe
        df_zones_subset = spark.createDataFrame(df_zones_subset)

        # Merge filtered DataFrame with zone information for pickup locations
        merged = filtered_df.join(df_zones_subset,
                           on=filtered_df.pulocationid == df_zones_subset.locationid,
                           how="left")
        
        # Renommage des colonnes
        merged = merged.withColumnRenamed("borough", "puborough") \
                                 .withColumnRenamed("zone", "puzone") \
                                 .withColumnRenamed("service_zone", "pu_service_zone") #\
                                 #.withColumnRenamed("locationid", "pulocationid")

        merged = merged.drop("locationid")

        # Merge filtered DataFrame with zone information for dropoff locations
        merged = merged.join(df_zones_subset,
                     on=merged.dolocationid == df_zones_subset.locationid,
                     how="left")
        
        # Renommage des colonnes
        merged = merged.withColumnRenamed("borough", "doborough") \
                                 .withColumnRenamed("zone", "dozone") \
                                 .withColumnRenamed("service_zone", "do_service_zone") #\
                                 #.withColumnRenamed("locationid", "dolocationid")
        
        merged = merged.drop("locationid")

        # Concatenate borough information for pickup and dropoff locations
        merged = merged.withColumn("Itineraire Arrondissement", concat_ws(" - ", merged["puborough"], merged["doborough"]))
        merged = merged.withColumn("Itineraire zone", concat_ws(" - ", merged["puzone"], merged["dozone"]))

        # Drop rows with null values
        merged = merged.dropna()

        logging.info("Data transformation completed successfully.")

        return merged
    
    except Exception as e:
        logging.error("An error occurred while transforming the data: %s", str(e))
        return None


def load(df: DataFrame, file_path: str):
    """
    Saves a PySpark DataFrame to Parquet format.

    Parameters:
    df (pyspark.sql.DataFrame): Input PySpark DataFrame.
    file_path (str): The path where the Parquet file will be saved.

    Returns:
    None
    """
    try:
        # Set up logging
        logging.basicConfig(filename='load.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Save DataFrame to Parquet
        df.write.parquet(file_path, mode="append")

        logging.info("Data loaded successfully into Parquet.")

    except Exception as e:
        logging.error("An error occurred while saving DataFrame to Parquet: %s", str(e))
```


Le script `etl_functions.py` définit trois fonctions principales pour un pipeline ETL (Extract, Transform, Load) en utilisant PySpark et SQLAlchemy. Voici une explication détaillée de chaque section du script et des fonctions définies :

- Importation des bibliothèques nécessaires

```python
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, LongType, DoubleType, StringType
from pyspark.sql import DataFrame
import pandas as pd
```

- `logging` : pour la journalisation des événements du script.
- `pyspark.sql` : pour manipuler les DataFrames avec PySpark.
- `pandas` : pour manipuler les données tabulaires en Python.


- Création de la session Spark

```python
# Create a SparkSession
spark = SparkSession.builder \
    .appName("ETL Pipeline") \
    .getOrCreate()

# https://stackoverflow.com/questions/59096125/spark-2-4-parquet-column-cannot-be-converted-in-file-column-impressions-exp
spark.conf.set("spark.sql.parquet.enableVectorizedReader","false") 

# Ajuster la configuration pour la représentation en chaîne des plans d'exécution
spark.conf.set("spark.sql.debug.maxToStringFields", "255")
```

- Initialisation d'une session Spark.
- Configuration de Spark pour désactiver le lecteur vectorisé pour les fichiers Parquet et ajuster la limite de champs pour la représentation en chaîne.

- Fonction d'extraction des données

- **Paramètre** : `file_path` - le chemin du fichier Parquet à extraire.
- **Retour** : un DataFrame PySpark contenant les données extraites.
- **Fonctionnement** :
  - Configure la journalisation.
  - Lit le fichier Parquet et le charge dans un DataFrame PySpark.
  - Journalise le succès ou l'échec de l'opération.

- Fonction de transformation des données

- **Paramètre** : `df` - le DataFrame PySpark à transformer.
- **Retour** : un DataFrame PySpark transformé.
- **Fonctionnement** :
  - Configure la journalisation.
  - Renomme les colonnes en minuscules.
  - Supprime les lignes avec des valeurs manquantes et filtre les données selon certaines conditions.
  - Lit les informations de zones à partir dún fichier CSV
  - Fusionne le DataFrame filtré avec les informations de zones pour les lieux de prise en charge et de dépose.
  - Renomme les colonnes résultantes pour les rendre plus explicites.
  - Concatène les informations des arrondissements et des zones.
  - Supprime les lignes avec des valeurs nulles.


- Fonction de chargement des données

- **Paramètres** :
  - `df` - le DataFrame PySpark à sauvegarder.
  - `file_path` - le chemin où le fichier Parquet sera sauvegardé.
- **Retour** : Aucun.
- **Fonctionnement** :
  - Configure la journalisation.
  - Sauvegarde le DataFrame en format Parquet avec le mode "append" pour ajouter les données sans écraser les fichiers existants.
  - Journalise le succès ou l'échec de l'opération.

- Conclusion

Le script `etl_functions.py` configure une session Spark puis définit trois fonctions pour extraire, transformer et charger des données. Ces fonctions utilisent la journalisation pour enregistrer les succès et les erreurs, ce qui facilite le suivi et le débogage du processus ETL.


🖥 **Exécution du processus ETL : Script etl_pipeline.py**

```python
import etl_functions
import logging
import os

# Configuration des logs
logging.basicConfig(filename='pyspark_etl_pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
#input_file_path = "histo_data_files"  # Adjust the path to your Parquet files
output_folder = "data_loaded"  # Dossier pour enregistrer les données tranformees

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Liste des fichiers Parquet dans le dossier
parquet_files = [f for f in os.listdir("histo_data_files") if f.endswith('.parquet')]

# Pour chaque fichier Parquet
for file_name in parquet_files:
    # Extract data
    logging.info("Starting data extraction...")
    input_df = etl_functions.extract("histo_data_files/" + file_name)
    logging.info("Data extraction completed.")

    # Transform data
    logging.info("Starting data transformation...")
    transformed_df = etl_functions.transform(input_df)
    logging.info("Data transformation completed.")

    # Load data
    if transformed_df is not None:
        logging.info("Starting data loading...")
        output_file_path = output_folder + "/transformed_taxi_data.parquet"
        etl_functions.load(transformed_df, output_file_path)
        logging.info("Data loading completed.")
```


Le script `etl_pipeline.py` est conçu pour orchestrer les étapes d'un pipeline ETL (Extract, Transform, Load) en utilisant les fonctions définies dans `etl_functions.py`. Voici une explication détaillée de chaque section du script :

- Importation des bibliothèques et modules nécessaires

```python
import etl_functions
import logging
import os
```

- `etl_functions` : importation des fonctions ETL définies dans un autre script (`etl_functions.py`).
- `logging` : pour la journalisation des événements du script.
- `os` : pour les opérations sur le système de fichiers, comme vérifier l'existence de dossiers ou lister des fichiers.

- Configuration de la journalisation

```python
logging.basicConfig(filename='pyspark_etl_pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

- Configure la journalisation pour enregistrer les messages dans un fichier `pyspark_etl_pipeline.log` avec le niveau `INFO` et un format de message spécifiant l'heure, le niveau de log et le message.

- Définition des chemins de fichiers

```python
# Define file paths
#input_file_path = "histo_data_files"  # Adjust the path to your Parquet files
output_folder = "data_loaded"  # Dossier pour enregistrer les données transformées
```

- `output_folder` : spécifie le dossier où les données transformées seront enregistrées.

- Création du dossier de sortie s'il n'existe pas

```python
# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
```

- Vérifie si le dossier de sortie existe et le crée s'il n'existe pas.

- Liste des fichiers Parquet dans le dossier d'entrée

```python
# Liste des fichiers Parquet dans le dossier
parquet_files = [f for f in os.listdir("histo_data_files") if f.endswith('.parquet')]
```

- Liste tous les fichiers dans le dossier `histo_data_files` qui se terminent par `.parquet` et les stocke dans `parquet_files`.

- Boucle sur chaque fichier Parquet pour les étapes ETL

```python
# Pour chaque fichier Parquet
for file_name in parquet_files:
    # Extract data
    logging.info("Starting data extraction...")
    input_df = etl_functions.extract("histo_data_files/" + file_name)
    logging.info("Data extraction completed.")

    # Transform data
    logging.info("Starting data transformation...")
    transformed_df = etl_functions.transform(input_df)
    logging.info("Data transformation completed.")

    # Load data
    if transformed_df is not None:
        logging.info("Starting data loading...")
        output_file_path = output_folder + "/transformed_taxi_data.parquet"
        etl_functions.load(transformed_df, output_file_path)
        logging.info("Data loading completed.")
```

- Pour chaque fichier Parquet dans `parquet_files` :
  1. **Extraction des données** :
     - Journalise le début de l'extraction.
     - Appelle la fonction `extract` du module `etl_functions` pour lire le fichier Parquet et stocke le DataFrame résultant dans `input_df`.
     - Journalise la fin de l'extraction.
  2. **Transformation des données** :
     - Journalise le début de la transformation.
     - Appelle la fonction `transform` du module `etl_functions` pour transformer le DataFrame `input_df` et stocke le DataFrame transformé dans `transformed_df`.
     - Journalise la fin de la transformation.
  3. **Chargement des données** :
     - Si `transformed_df` n'est pas `None` (c'est-à-dire si la transformation a réussi), journalise le début du chargement.
     - Définit le chemin du fichier de sortie (`output_file_path`).
     - Appelle la fonction `load` du module `etl_functions` pour sauvegarder le DataFrame transformé dans un fichier Parquet.
     - Journalise la fin du chargement.

- Conclusion

Le script `etl_pipeline.py` est un orchestrateur simple et efficace pour exécuter les étapes d'extraction, de transformation et de chargement des données sur une série de fichiers Parquet. Chaque étape est journalisée, permettant un suivi facile des processus et un débogage en cas de problème. Les fonctions spécifiques à chaque étape (extraction, transformation, chargement) sont importées du script `etl_functions.py`, rendant le code modulaire et facile à maintenir.


Pour exécuter ce pipeline ETL, vous devez exécuter ces commandes dans l'ordre :

1. Téléchargement des données brutes

```bash
python3 download_data_files.py
``` 


2. Exécution du Pipeline

```bash
python3 etl_pipeline.py
``` 

Les principaux packages requis pour ce projet sont :

numpy==1.26.4

pandas==2.2.2

pyarrow==16.0.0

requests==2.31.0

pyspark==3.5.1