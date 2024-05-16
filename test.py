import streamlit as st
import markdown

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

def main():
    st.title("Visionneuse de fichier Markdown")

    # Sélection du fichier .md
    uploaded_file = st.file_uploader("Uploader un fichier .md", type=["md"])

    if uploaded_file is not None:
        # Lecture du contenu du fichier .md
        text = uploaded_file.read().decode("utf-8")

        # Extraction du contenu de la section "Execution du Projet"
        execution_section_content = extract_execution_section(text)

        # Affichage du contenu Markdown de la section "Execution du Projet"
        st.markdown(execution_section_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
