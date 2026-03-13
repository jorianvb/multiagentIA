# memory/narrative_bible.py
# Gestion de la bible narrative persistante
# La bible narrative est la source de vérité long terme du projet

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class NarrativeBible:
    """
    La Bible Narrative est le document persistant qui contient
    toutes les informations canoniques de l'histoire.

    Elle est mise à jour après chaque analyse et versionnée.

    Structure :
    - bible_narrative/
      ├── bible_v001.json    (version 1)
      ├── bible_v002.json    (version 2)
      ├── current.json       (version courante, symlink ou copie)
      └── changelog.md       (historique des modifications)
    """

    def __init__(self, project_name: str, base_dir: str = "./output"):
        self.project_name = project_name
        self.base_dir = Path(base_dir) / project_name / "bible"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self.base_dir / "current.json"
        self.changelog_file = self.base_dir / "changelog.md"
        self._bible = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Charge la bible existante ou en crée une nouvelle."""
        if self.current_file.exists():
            with open(self.current_file, "r", encoding="utf-8") as f:
                print(f"📚 Bible narrative chargée : {self.current_file}")
                return json.load(f)
        else:
            print("📚 Création d'une nouvelle bible narrative")
            return self._empty_bible()

    def _empty_bible(self) -> dict:
        """Structure vide d'une nouvelle bible narrative."""
        return {
            "meta": {
                "project_name": self.project_name,
                "created_at": datetime.now().isoformat(),
                "version": 1,
                "last_updated": datetime.now().isoformat()
            },
            "univers": {
                "genre": "",
                "epoque": "",
                "lieux_principaux": [],
                "regles_univers": []     # règles spécifiques à l'univers (magie, etc.)
            },
            "personnages": {},           # Dictionnaire des personnages canoniques
            "intrigues": {},             # Dictionnaire des intrigues
            "chronologie": [],           # Liste d'événements dans l'ordre
            "faits_canoniques": [],      # Faits établis et incontestables
            "questions_ouvertes": [],    # Mystères non résolus intentionnels
            "inconsistances_connues": [] # Incohérences connues et assumées
        }

    def update_from_analysis(self,
                             characters: dict,
                             plots: dict,
                             context: str,
                             session_id: str) -> None:
        """
        Met à jour la bible avec les résultats d'une nouvelle analyse.

        Stratégie de mise à jour :
        - Nouveaux personnages → ajoutés
        - Personnages existants → MERGE (on ne supprime jamais une info canonique)
        - Si conflit → on garde l'ancienne valeur et on note le conflit
        """
        changes = []

        # Mise à jour des personnages
        for nom, info in characters.items():
            if nom not in self._bible["personnages"]:
                # Nouveau personnage
                self._bible["personnages"][nom] = info
                changes.append(f"Nouveau personnage : {nom}")
            else:
                # Personnage existant : merge intelligent
                existing = self._bible["personnages"][nom]

                # Fusion des listes (traits, motivations, arcs)
                for field in ["traits", "motivations", "arcs"]:
                    if field in info:
                        existing_list = existing.get(field, [])
                        new_items = [x for x in info.get(field, [])
                                     if x not in existing_list]
                        if new_items:
                            existing[field] = existing_list + new_items
                            changes.append(f"{nom} - nouveaux {field}: {new_items}")

                # Mise à jour du statut actuel (toujours la version la plus récente)
                if info.get("statut_actuel"):
                    existing["statut_actuel"] = info["statut_actuel"]

                # Fusion des relations
                existing_relations = existing.get("relations", {})
                new_relations = info.get("relations", {})
                existing_relations.update(new_relations)
                existing["relations"] = existing_relations

                self._bible["personnages"][nom] = existing

        # Mise à jour des intrigues (logique similaire)
        for titre, intrigue in plots.items():
            if titre not in self._bible["intrigues"]:
                self._bible["intrigues"][titre] = intrigue
                changes.append(f"Nouvelle intrigue : {titre}")
            else:
                existing = self._bible["intrigues"][titre]
                # Mise à jour du statut
                if intrigue.get("statut"):
                    existing["statut"] = intrigue["statut"]
                # Fusion des fils non résolus
                existing_fils = set(existing.get("fils_non_resolus", []))
                new_fils = set(intrigue.get("fils_non_resolus", []))
                existing["fils_non_resolus"] = list(existing_fils | new_fils)
                self._bible["intrigues"][titre] = existing

        # Mise à jour des métadonnées
        self._bible["meta"]["last_updated"] = datetime.now().isoformat()
        self._bible["meta"]["version"] += 1

        # Sauvegarde
        self._save(changes, session_id)

    def _save(self, changes: list, session_id: str) -> None:
        """Sauvegarde la bible et crée une version archivée."""
        version = self._bible["meta"]["version"]

        # Sauvegarde de la version courante
        with open(self.current_file, "w", encoding="utf-8") as f:
            json.dump(self._bible, f, ensure_ascii=False, indent=2)

        # Archivage de la version
        archive_file = self.base_dir / f"bible_v{version:03d}.json"
        with open(archive_file, "w", encoding="utf-8") as f:
            json.dump(self._bible, f, ensure_ascii=False, indent=2)

        # Mise à jour du changelog
        self._update_changelog(version, changes, session_id)

        print(f"   💾 Bible sauvegardée : version {version} ({len(changes)} modification(s))")

    def _update_changelog(self, version: int, changes: list, session_id: str) -> None:
        """Met à jour le fichier changelog."""
        entry = f"\n## Version {version:03d} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        entry += f"\nSession : {session_id}\n"
        if changes:
            for change in changes:
                entry += f"- {change}\n"
        else:
            entry += "- Aucune modification détectée\n"

        with open(self.changelog_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def get_context_for_agents(self) -> str:
        """
        Retourne un résumé compressé de la bible pour injection dans les agents.
        Utilisé pour la mémoire long terme.
        """
        lines = ["=== BIBLE NARRATIVE (Mémoire Long Terme) ==="]
        lines.append(f"Projet : {self.project_name} | Version : {self._bible['meta']['version']}")

        # Personnages canoniques
        if self._bible["personnages"]:
            lines.append(f"\nPERSONNAGES CANONIQUES ({len(self._bible['personnages'])}) :")
            for nom, info in self._bible["personnages"].items():
                lines.append(f"  - {nom} ({info.get('role', '?')}) : {info.get('statut_actuel', '?')}")

        # Faits canoniques
        if self._bible["faits_canoniques"]:
            lines.append("\nFAITS ÉTABLIS :")
            for fait in self._bible["faits_canoniques"][-10:]:  # Les 10 derniers
                lines.append(f"  • {fait}")

        # Questions ouvertes
        if self._bible["questions_ouvertes"]:
            lines.append("\nQUESTIONS OUVERTES :")
            for question in self._bible["questions_ouvertes"]:
                lines.append(f"  ? {question}")

        return "\n".join(lines)

    @property
    def bible(self) -> dict:
        return self._bible
