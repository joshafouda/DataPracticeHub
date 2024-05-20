import streamlit as st
#from st_paywall import add_auth
from PIL import Image
import os
from utils import load_projects

# Charger les projets depuis le répertoire 'projects'
projects = load_projects()

# Configuration de la page
st.set_page_config(
    page_title="DataPracticeHub",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Titre de la page
#st.title("DataPracticeHub\nMade by Josué AFOUDA")

# Vérifiez l'existence des fichiers
logo_path = "imgs/logo.png"
animation_path = "imgs/logo_animation.gif"

# Utilisation de colonnes pour afficher les images côte à côte
if os.path.exists(logo_path) and os.path.exists(animation_path):
    col1, col2 = st.columns(2)
    with col1:
        logo_image = Image.open(logo_path)
        st.image(logo_image, use_column_width=True)
    with col2:
        st.image(animation_path, use_column_width=True)

st.header("Bienvenue sur DataPracticeHub")
st.write("DataPracticeHub est un répertoire de projets réels en Data Science pour vous aider à apprendre par la pratique.")

#add_auth(required=True)

# Sidebar pour la navigation
st.sidebar.title("Navigation")
pages = ["Accueil", "Projets", "À propos"]
page = st.sidebar.radio("Aller à", pages)

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
    #st.header("Bienvenue sur DataPracticeHub")
    #st.write("DataPracticeHub est un répertoire de projets réels en Data Science pour vous aider à apprendre par la pratique.")
    st.write("Choisissez un projet ci-dessous pour commencer :")

    # Exemples de projets sur la page d'accueil
    cols = st.columns(2)

    for i, (project_name, project) in enumerate(projects.items()):
        with cols[i % 2]:
            st.subheader(f"**{project_name}**")
            if project["image"]:
                st.image(project["image"])
            st.write(project["description"])
            if st.button("Guide", key=f"details_{project_name}"):
                st.session_state.page = project_name
            if st.button("Solution", key=f"solution_{project_name}"):
                st.session_state.page = f"solution_{project_name}"

elif page == "Projets":
    st.header("Tous les Projets")
    st.write("Liste de tous les projets disponibles :")

    # Liste des projets
    for project_name, project in projects.items():
        st.subheader(f"**{project_name}**")
        if project["image"]:
            st.image(project["image"])
        st.write(project["description"])
        if st.button("Détails", key=f"details_{project_name}"):
            st.session_state.page = project_name
        if st.button("Solution", key=f"solution_{project_name}"):
            st.session_state.page = f"solution_{project_name}"

elif page == "À propos":
    st.header("À propos de DataPracticeHub")
    st.write("DataPracticeHub est conçu pour aider les passionnés de Data Science à apprendre en réalisant des projets pratiques.")
    st.write("Pour toute question, contactez-nous à [votre-email@example.com](mailto:votre-email@example.com)")

# Afficher le détail ou l'exécution du projet
if "page" in st.session_state:
    if st.session_state.page in projects:
        show_project_details(st.session_state.page)
    elif st.session_state.page.startswith("solution_"):
        project_name = st.session_state.page.replace("solution_", "")
        if project_name in projects:
            show_project_execution(project_name)