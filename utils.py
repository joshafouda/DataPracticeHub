import os
import markdown
from collections import defaultdict

# Fonction pour lire un fichier Markdown et extraire les sections
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
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

# Fonction pour charger tous les projets à partir des fichiers Markdown
def load_projects(directory="projects"):
    projects = {}
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            #project_name = filename[:-3]  # Remove the .md extension
            sections = read_markdown_file(os.path.join(directory, filename))

            # Parcourir les clés de sections
            for key in sections.keys():
                # Vérifier si la clé contient la sous-chaîne "Projet :"
                if 'Projet :' in key:
                    # Extraire le titre du projet en supprimant "Projet :" et les espaces avant et après
                    project_name = key.replace('Projet :', '').strip()
                    break  # Sortir de la boucle après avoir trouvé le titre du projet

            projects[project_name] = {
                "description": sections["Description"].strip(),
                "image": sections["Image"].strip(),
                "instructions": sections["Instructions"].strip().split('\n'),
                "resources": sections["Resources"].strip().split('\n'),
            }
    return projects