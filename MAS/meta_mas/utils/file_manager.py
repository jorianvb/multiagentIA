"""
Gestionnaire de fichiers pour la création du MAS généré.

Crée les répertoires, écrit les fichiers, gère le rollback.
"""
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class FileManager:
    """
    Gère les opérations système de fichiers pour le MAS généré.

    Fonctionnalités :
    - Création de structures de répertoires
    - Écriture sûre de fichiers (avec création des parents)
    - Rollback (suppression des fichiers créés)
    - Affichage de l'arborescence
    - Suivi des fichiers créés
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()
        self._created_files: List[Path] = []
        self._created_dirs: List[Path] = []
        logger.info(f"FileManager: chemin de base → {self.base_path}")

    # ------------------------------------------------------------------
    # Création de répertoires
    # ------------------------------------------------------------------

    def ensure_base(self) -> None:
        """Crée le répertoire de base s'il n'existe pas."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        if self.base_path not in self._created_dirs:
            self._created_dirs.append(self.base_path)

    def create_directory(self, relative_path: str) -> Path:
        """Crée un répertoire (et ses parents)."""
        dir_path = self.base_path / relative_path
        dir_path.mkdir(parents=True, exist_ok=True)
        self._created_dirs.append(dir_path)
        logger.debug(f"FileManager: répertoire créé — {dir_path}")
        return dir_path

    def create_standard_structure(self) -> None:
        """Crée la structure standard pour un MAS généré."""
        dirs = ["config", "agents", "core", "tests"]
        for d in dirs:
            self.create_directory(d)

    # ------------------------------------------------------------------
    # Écriture de fichiers
    # ------------------------------------------------------------------

    def write_file(
        self,
        relative_path: str,
        content: str,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> Path:
        """
        Écrit le contenu dans un fichier.

        Args:
            relative_path: Chemin relatif au base_path.
            content: Contenu à écrire.
            overwrite: Si False, ne pas écraser un fichier existant.
            encoding: Encodage du fichier.

        Returns:
            Chemin absolu du fichier créé.
        """
        file_path = self.base_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and not overwrite:
            logger.warning(f"FileManager: fichier existant ignoré — {file_path}")
            return file_path

        file_path.write_text(content, encoding=encoding)
        self._created_files.append(file_path)
        logger.debug(
            f"FileManager: écrit {len(content)} chars → {relative_path}"
        )
        return file_path

    def write_files(self, files: Dict[str, str]) -> List[Path]:
        """
        Écrit plusieurs fichiers en une seule opération.

        Args:
            files: Dict {chemin_relatif: contenu}.

        Returns:
            Liste des chemins absolus créés.
        """
        written: List[Path] = []
        for rel_path, content in files.items():
            written.append(self.write_file(rel_path, content))
        logger.info(f"FileManager: {len(written)} fichier(s) écrits")
        return written

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def read_file(
        self, relative_path: str, encoding: str = "utf-8"
    ) -> Optional[str]:
        """Lit le contenu d'un fichier."""
        fp = self.base_path / relative_path
        if not fp.exists():
            return None
        return fp.read_text(encoding=encoding)

    def file_exists(self, relative_path: str) -> bool:
        """Vérifie si un fichier existe."""
        return (self.base_path / relative_path).exists()

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def list_files(self, pattern: str = "**/*") -> List[Path]:
        """Liste les fichiers correspondant au pattern."""
        return [p for p in self.base_path.glob(pattern) if p.is_file()]

    def get_tree(self) -> str:
        """Retourne une représentation arborescente du répertoire."""
        lines: List[str] = [str(self.base_path)]
        self._tree_recursive(self.base_path, lines, "")
        return "\n".join(lines)

    def _tree_recursive(
        self, path: Path, lines: List[str], prefix: str
    ) -> None:
        """Construit récursivement l'arbre."""
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{item.name}")
            if item.is_dir():
                extension = "    " if is_last else "│   "
                self._tree_recursive(item, lines, prefix + extension)

    def get_created_files(self) -> List[Path]:
        """Retourne la liste des fichiers créés."""
        return self._created_files.copy()

    def get_created_dirs(self) -> List[Path]:
        """Retourne la liste des répertoires créés."""
        return self._created_dirs.copy()

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> None:
        """Supprime tous les fichiers et répertoires créés (rollback)."""
        for fp in reversed(self._created_files):
            if fp.exists():
                fp.unlink()
                logger.debug(f"FileManager: rollback — supprimé {fp}")

        for dp in reversed(self._created_dirs):
            if dp.exists():
                try:
                    dp.rmdir()  # Ne supprime que si vide
                    logger.debug(f"FileManager: rollback — supprimé {dp}")
                except OSError:
                    pass  # Répertoire non vide, on ignore

    def delete_all(self) -> None:
        """Supprime le répertoire de base et tout son contenu."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            logger.warning(f"FileManager: supprimé — {self.base_path}")

