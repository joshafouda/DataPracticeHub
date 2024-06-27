import streamlit as st
from streamlit_player import st_player
from st_paywall import add_auth
from PIL import Image
import os
from utils import load_projects, read_description
import subprocess

from pathlib import Path

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer


# Récupérer le token d'accès personnel depuis les secrets de Streamlit
#github_pat = st.secrets["general"]["GITHUB_TOKEN"]
github_pat = st.secrets["GITHUB_TOKEN"]

# Définir le contenu du script shell
script_content = f"""#!/bin/bash
REPO_DIR="repository"

# Create the repository directory if it doesn't exist
mkdir -p "$REPO_DIR"

# Remove the repository directory if it exists
if [ -d "$REPO_DIR" ]; then
    echo "Removing existing repository directory."
    rm -rf "$REPO_DIR"
fi

# Clone the repository
echo "Cloning the repository."
git clone https://joshafouda:{github_pat}@github.com/joshafouda/DataPracticeHub-App-Resources.git "$REPO_DIR"

# Move the contents of the repository one level up and replace existing files
echo "Moving the contents of the repository."
mv -f "$REPO_DIR"/* .
mv -f "$REPO_DIR"/.[^.]* .

# Remove the repository directory
echo "Removing the repository directory."
rm -rf "$REPO_DIR"
"""

# Fonction pour exécuter le script shell
def run_shell_script(script_content):
    subprocess.run(['bash', '-c', script_content], capture_output=True, text=True)
    
    
# Exécuter le script pour cloner le dépôt, forçant le remplacement si nécessaire
run_shell_script(script_content)


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


# Sidebar pour la navigation
st.title("Menu")
pages = ["Accueil", "À propos", "Projets", "Livres"]
page = st.radio("Aller à :", pages, horizontal=True)

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
    # Sous-titre
    st.subheader("Découvrez notre plateforme avec cette vidéo de présentation")

    # URL de la vidéo YouTube
    html_video = '''<iframe width="560" height="315" src="https://www.youtube.com/embed/SpXPIb6Jkfo?si=u9tajJf7jeHJVj6m" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'''

    # Affichage de la vidéo
    st_player(html_video)


elif page == "À propos":
    st.header("À propos de DataPracticeHub")
    st.write("DataPracticeHub est conçu pour aider les passionnés de Data Science à apprendre en réalisant des projets pratiques.")
    st.write("Pour toute question, contactez-nous à [j.a.datatech.consulting@gmail.com](mailto:j.a.datatech.consulting@gmail.com)")


add_auth(required=True)

if page == "Projets":
    # st.header("Bienvenue sur DataPracticeHub")
    # st.write("DataPracticeHub est un répertoire de projets réels en Data Science pour vous aider à apprendre par la pratique.")
    st.write("Choisissez un projet ci-dessous pour commencer :")

    cols = st.columns(2)

    #for project_name, project in projects.items():
    for i, (project_name, project) in enumerate(projects.items()):
        with cols[i % 2]:
            with st.expander(f"Projet : {project_name}", expanded=True):
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

elif page == "Livres":

    # Répertoire contenant les livres
    books_directory = Path('books')

    # Liste des fichiers PDF et des images de couverture
    books = [book for book in books_directory.glob('*.pdf')]

    st.title('Bibliothèque PDF')

    # Parcourir tous les livres
    for book in books:
        # Obtenir le nom de base du livre sans extension
        book_name = book.stem

        # Trouver l'image de couverture correspondante
        cover_image = books_directory / f'{book_name}.png'
        description_file = books_directory / f'{book_name}.md'
        
        # Lire la description
        if description_file.exists():
            description = read_description(description_file)
        else:
            description = "Description non disponible."

        with st.expander(book_name, expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if cover_image.exists():
                    st.image(str(cover_image), use_column_width=True)
                else:
                    st.text("Image non disponible.")
            
            with col2:
                st.markdown(description)
                if st.button(f'Lire {book_name}'):
                    if 'pdf_ref' not in ss:
                        ss.pdf_ref = None
                    
                    with open(book, 'rb') as f:
                        ss.pdf_ref = f.read() # Lire le fichier PDF

                    # Afficher le PDF avec streamlit_pdf_viewer
                    if ss.pdf_ref:
                        pdf_viewer(input=ss.pdf_ref, width=1000)
                #if st.button(f'Acheter {book_name}'):
                    #st.write(f"[Lien d'achat](https://example.com/{book_name})")
