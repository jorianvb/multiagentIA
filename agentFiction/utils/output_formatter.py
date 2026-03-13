# utils/output_formatter.py
# Sauvegarde et formatage des sorties pour l'auteur

import os
from pathlib import Path
from datetime import datetime


def save_response_to_file(
        response: str,
        project_name: str,
        session_id: str,
        output_dir: str = "./output"
) -> str:
    """
    Sauvegarde la réponse finale dans un fichier texte.

    Structure de sortie :
    output/
    └── {project_name}/
        └── sessions/
            └── {date}_{session_id}.txt

    Retourne le chemin du fichier créé.
    """
    # Création du répertoire de sortie
    output_path = Path(output_dir) / project_name / "sessions"
    output_path.mkdir(parents=True, exist_ok=True)

    # Nom de fichier avec date et ID de session
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{date_str}_{session_id[:8]}.txt"
    filepath = output_path / filename

    # Écriture du fichier
    header = (
        f"═══════════════════════════════════════\n"
        f"SYSTÈME MULTI-AGENT FICTION\n"
        f"Projet : {project_name}\n"
        f"Session : {session_id}\n"
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"═══════════════════════════════════════\n\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(response)

    print(f"\n💾 Réponse sauvegardée : {filepath}")
    return str(filepath)
