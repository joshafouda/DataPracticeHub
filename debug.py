import os
import markdown
from collections import defaultdict

directory = "projects"
filename = "project1.md"

# Fonction qui prend en argument un fichier .md et retourne le contenu complet de ce fichier .md sous forme de chaîne de caractères
def read_md_file(file_path):
    """
    Lit un fichier .md et retourne son contenu sous forme de chaîne de caractères.

    Args:
        file_path (str): Le chemin vers le fichier .md.

    Returns:
        str: Le contenu complet du fichier .md.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
        return md_content
    except FileNotFoundError:
        print(f"Le fichier {file_path} n'a pas été trouvé.")
        return ""
    except Exception as e:
        print(f"Une erreur s'est produite lors de la lecture du fichier : {e}")
        return ""

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

sections = read_markdown_file(os.path.join(directory, filename))
#print(type(sections))
#print('\n')
#print(sections)
#print('\n')

texte = read_md_file(os.path.join(directory, filename))
#print(type(texte))
#print('\n')
#print(texte)

print(extract_execution_section(texte))