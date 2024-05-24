import streamlit as st
import requests
import markdown
from collections import defaultdict
from io import BytesIO
from PIL import Image
import re
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Maintenant, vous pouvez accéder aux variables d'environnement normalement comme ceci :
GITHUB_API_URL = os.getenv("GITHUB_API_URL")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
REPO_NAME = os.getenv("REPO_NAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_REPO_IMGS = os.getenv("GITHUB_REPO_IMGS")

headers = {
    "Authorization": f"token {GITHUB_TOKEN}"
}


# Fonction pour charger tous les projets à partir des fichiers Markdown depuis GitHub
def load_projects():
    projects = {}
    response = requests.get(GITHUB_REPO, headers=headers)
    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file['name'].endswith(".md"):
                file_path = file['path']
                file_url = file['download_url']
                
                sections = read_markdown_file(file_url)
                texte = read_md_file(file_url)
                
                # Parcourir les clés de sections
                for key in sections.keys():
                    # Vérifier si la clé contient la sous-chaîne "Projet :"
                    if 'Projet :' in key:
                        # Extraire le titre du projet en supprimant "Projet :" et les espaces avant et après
                        project_name = key.replace('Projet :', '').strip()
                        break  # Sortir de la boucle après avoir trouvé le titre du projet

                # Construire l'URL de l'image (C'est de la pure fabrication)
                image_file_name = sections["Image"].strip()
                https_link = re.search(r'\bhttps?://\S+\b', image_file_name).group()
                after_imgs = https_link.split('imgs/')[1]
                url_image = f"{GITHUB_API_URL}/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/imgs/{after_imgs}"
                last_slash_index = url_image.rfind('/') # Trouver l'index de la dernière occurrence du caractère '/'
                url_image_modified = url_image[:last_slash_index]  # Extraire la partie de la chaîne jusqu'à cet index
                response_img = requests.get(url_image_modified, headers=headers)
                response_json = response_img.json()
                file_url_img = response_json[0]['download_url']
                image = fetch_image_from_github(file_url_img)

                projects[project_name] = {
                    "description": sections["Description"].strip(),
                    "image": image,
                    "instructions": sections["Instructions"].strip().split('\n'),
                    "resources": sections["Resources"].strip().split('\n'),
                    "execution": extract_execution_section(texte)
                }
    else:
        print(f"Erreur {response.status_code} lors de la récupération des fichiers depuis GitHub")
    return projects



# Fonction pour récupérer un fichier depuis GitHub
def fetch_file_from_github(file_url):
    response = requests.get(file_url, headers=headers)
    if response.status_code == 200:
        return response.content.decode('utf-8')
    else:
        print(f"Erreur {response.status_code} lors de la récupération du fichier {file_url}")
        return ""

# Fonction pour récupérer une image depuis GitHub
def fetch_image_from_github(image_url):
    response = requests.get(image_url, headers={"Authorization": f"token {GITHUB_TOKEN}"})
    
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Erreur {response.status_code} lors de la récupération de l'image {image_url}")
        return None

# Fonction qui prend en argument un fichier .md et retourne le contenu complet de ce fichier .md sous forme de chaîne de caractères
def read_md_file(file_path):
    """
    Lit un fichier .md et retourne son contenu sous forme de chaîne de caractères.

    Args:
        file_path (str): Le chemin vers le fichier .md.

    Returns:
        str: Le contenu complet du fichier .md.
    """
    return fetch_file_from_github(file_path)

# Fonction pour lire un fichier Markdown et extraire les sections
def read_markdown_file(file_path):
    text = fetch_file_from_github(file_path)
    md = markdown.markdown(text)
    sections = defaultdict(str)
    current_section = None
    for line in text.split('\n'):
        if line.startswith('#'):
            current_section = line.strip('# ').strip()
        else:
            if current_section:
                sections[current_section] += line + '\n'
    return sections

def extract_execution_section(text):
    # Découpe le texte en lignes
    lines = text.split("\n")
    in_execution_section = False
    execution_content = []

    # Parcourt les lignes
    for line in lines:
        # Vérifie si la ligne débute la section "Execution du Projet"
        if line.strip() == "## Execution du Projet":
            in_execution_section = True
        # Si nous sommes dans la section, ajoute la ligne au contenu
        elif in_execution_section:
            # Vérifie si nous sommes sortis de la section
            if line.strip().startswith("##"):
                break
            execution_content.append(line)

    # Rejoindre les lignes pour obtenir le contenu complet
    return "\n".join(execution_content)


def get_image_download_url(file_api_url):
    response = requests.get(file_api_url, headers=headers)
    if response.status_code == 200:
        file_info = response.json()
        download_url = file_info['download_url']
        return download_url
    else:
        print(f"Erreur {response.status_code} lors de la récupération des informations de fichier depuis GitHub")
        return None


def fetch_gif_from_github(gif_url):
    response = requests.get(gif_url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Erreur {response.status_code} lors de la récupération du GIF depuis GitHub")
        return None
