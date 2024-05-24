import streamlit as st
#from st_paywall import add_auth
from PIL import Image
import os
#from dotenv import load_dotenv
from utils import load_projects, fetch_image_from_github, get_image_download_url, fetch_gif_from_github

# Charger les variables d'environnement depuis le fichier .env
#load_dotenv()

# Maintenant, vous pouvez accéder aux variables d'environnement normalement comme ceci :
#GITHUB_API_URL = os.getenv("GITHUB_API_URL")
#USERNAME = os.getenv("GITHUB_USERNAME")
#REPO_NAME = os.getenv("REPO_NAME")
#GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
#GITHUB_REPO = os.getenv("GITHUB_REPO")
#GITHUB_REPO_IMGS = os.getenv("GITHUB_REPO_IMGS")

GITHUB_API_URL = "https://api.github.com"
GITHUB_USERNAME = "joshafouda"
REPO_NAME = "DataPracticeHub-App-Resources"
GITHUB_TOKEN = "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"
GITHUB_REPO = "https://api.github.com/repos/joshafouda/DataPracticeHub-App-Resources/contents/projects"
GITHUB_REPO_IMGS = "https://api.github.com/repos/joshafouda/DataPracticeHub-App-Resources/contents/imgs"

headers = {"Authorization": f"token {GITHUB_TOKEN}"}

# Charger les projets depuis le répertoire 'projects'
projects = load_projects()

# Configuration de la page
st.set_page_config(
    page_title="DataPracticeHub",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS pour styliser les boutons
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Titre de la page
#st.title("DataPracticeHub\nMade by Josué AFOUDA")

# URLs des images dans le dépôt GitHub
logo_image_url = f"{GITHUB_REPO_IMGS}/logo.png"
logo_animation_url = f"{GITHUB_REPO_IMGS}/logo_animation.gif"

# Obtenez les URL de téléchargement des images
logo_image_download_url = get_image_download_url(logo_image_url)
logo_animation_download_url = get_image_download_url(logo_animation_url)

# Utilisation de colonnes pour afficher les images côte à côte
if logo_image_download_url and logo_animation_download_url:
    logo_image = fetch_image_from_github(logo_image_download_url)
    logo_animation = fetch_gif_from_github(logo_animation_download_url)

    if logo_image and logo_animation:
        col1, col2 = st.columns(2)
        with col1:
            st.image(logo_image, use_column_width=True)
        with col2:
            st.image(logo_animation, use_column_width=True)
    else:
        st.write("Erreur lors du chargement des images")
else:
    st.write("Les URL de téléchargement des images n'ont pas pu être obtenues")


st.header("Bienvenue sur DataPracticeHub")
st.write("DataPracticeHub est un répertoire de projets réels en Data Science pour vous aider à apprendre par la pratique.")

#add_auth(required=True)

# Sidebar pour la navigation
st.title("Menu")
pages = ["Accueil", "À propos"]
page = st.radio("Aller à", pages, horizontal=True)

# Fonction pour afficher les détails d'un projet
def show_project_details(project_name):
    project = projects[project_name]
    st.header(project_name)
    st.write(project["description"])
    st.image(project["image"])
    st.write("### Instructions")
    for instruction in project["instructions"]:
        st.write(instruction)
    st.write("### Ressources")
    for resource in project["resources"]:
        st.write(resource)
        

# Fonction pour afficher l'exécution du projet
def show_project_execution(project_name):
    project = projects[project_name]
    st.header(f"**{project_name}** : Exécution du Projet")
    #st.write(project["execution"])
    st.markdown(project["execution"], unsafe_allow_html=True)

if page == "Accueil":
    # st.header("Bienvenue sur DataPracticeHub")
    # st.write("DataPracticeHub est un répertoire de projets réels en Data Science pour vous aider à apprendre par la pratique.")
    st.write("Choisissez un projet ci-dessous pour commencer :")

    cols = st.columns(2)

    #for project_name, project in projects.items():
    for i, (project_name, project) in enumerate(projects.items()):
        with cols[i % 2]:
            with st.expander(f"Projet : {project_name}", expanded=False):
                st.subheader(f"**{project_name}**")
                if project["image"]:
                    st.image(project["image"])
                st.write(project["description"])

                if st.button("Guide", key=f"details_{project_name}"):
                    st.session_state.page = project_name
                    show_project_details(project_name)
                        
                if st.button("Solution", key=f"solution_{project_name}"):
                    st.session_state.page = f"solution_{project_name}"
                    show_project_execution(project_name)
            st.text("  ")
            st.text("  ")


elif page == "À propos":
    st.header("À propos de DataPracticeHub")
    st.write("DataPracticeHub est conçu pour aider les passionnés de Data Science à apprendre en réalisant des projets pratiques.")
    st.write("Pour toute question, contactez-nous à [j.a.datatech.consulting@gmail.com](mailto:j.a.datatech.consulting@gmail.com)")
