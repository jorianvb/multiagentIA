"""
Mémoire partagée et contexte global du Meta-MAS.

Thread-safe via asyncio.Lock. Stocke toutes les données du pipeline.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


class SharedMemory:
    """
    Mémoire partagée accessible par tous les agents du Meta-MAS.

    Clés standard du pipeline :
    - user_request         : demande brute de l'utilisateur
    - target_directory     : dossier cible pour le MAS généré
    - ollama_model         : modèle LLM sélectionné
    - analysis_result      : résultat de l'agent Analyste (dict)
    - architecture_result  : résultat de l'agent Architecte (dict)
    - generated_agents     : liste des agents générés (code + métadonnées)
    - validation_results   : résultats des validations
    - deployment_status    : statut du déploiement
    - pipeline_stage       : étape courante
    - errors               : liste des erreurs rencontrées
    """

    def __init__(self):
        self._store: Dict[str, Any] = {
            "user_request": None,
            "target_directory": None,
            "ollama_model": None,
            "analysis_result": None,
            "architecture_result": None,
            "generated_agents": [],
            "generated_files": {},
            "validation_results": [],
            "deployment_status": None,
            "errors": [],
            "pipeline_stage": "INIT",
        }
        self._history: List[Dict] = []
        self._lock = asyncio.Lock()
        logger.info("SharedMemory initialisée")

    # ------------------------------------------------------------------
    # Opérations CRUD
    # ------------------------------------------------------------------

    async def set(self, key: str, value: Any) -> None:
        """Définit une valeur (thread-safe)."""
        async with self._lock:
            old = self._store.get(key)
            self._store[key] = value
            self._history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "op": "SET",
                    "key": key,
                    "old": str(old)[:80] if old is not None else None,
                    "new": str(value)[:80],
                }
            )
            logger.debug(f"Memory SET: {key}")

    async def get(self, key: str, default: Any = None) -> Any:
        """Lit une valeur (thread-safe)."""
        async with self._lock:
            return self._store.get(key, default)

    async def append(self, key: str, value: Any) -> None:
        """Ajoute un élément à une liste existante."""
        async with self._lock:
            if key not in self._store:
                self._store[key] = []
            if not isinstance(self._store[key], list):
                raise TypeError(f"La clé '{key}' n'est pas une liste")
            self._store[key].append(value)
            logger.debug(f"Memory APPEND: {key}")

    async def update_dict(self, key: str, updates: Dict) -> None:
        """Met à jour un dictionnaire existant."""
        async with self._lock:
            if key not in self._store:
                self._store[key] = {}
            if not isinstance(self._store[key], dict):
                raise TypeError(f"La clé '{key}' n'est pas un dict")
            self._store[key].update(updates)
            logger.debug(f"Memory UPDATE: {key}")

    async def delete(self, key: str) -> None:
        """Supprime une clé."""
        async with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug(f"Memory DELETE: {key}")

    async def get_all(self) -> Dict[str, Any]:
        """Retourne un snapshot de toute la mémoire."""
        async with self._lock:
            return self._store.copy()

    # ------------------------------------------------------------------
    # Helpers pipeline
    # ------------------------------------------------------------------

    async def set_pipeline_stage(self, stage: str) -> None:
        """Met à jour l'étape courante du pipeline."""
        await self.set("pipeline_stage", stage)
        logger.info(f"🔄 Pipeline stage: [{stage}]")

    async def add_error(self, error: str, context: str = "") -> None:
        """Enregistre une erreur dans la mémoire."""
        await self.append(
            "errors",
            {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "error": error,
            },
        )
        logger.error(f"Memory: erreur enregistrée — {error[:120]}")

    async def add_generated_file(self, filename: str, content: str) -> None:
        """Ajoute un fichier généré à la mémoire."""
        async with self._lock:
            if "generated_files" not in self._store:
                self._store["generated_files"] = {}
            self._store["generated_files"][filename] = content
        logger.debug(f"Memory: fichier généré ajouté — {filename}")

    def get_history(self) -> List[Dict]:
        """Retourne l'historique des opérations mémoire."""
        return self._history.copy()

    def snapshot(self) -> Dict[str, Any]:
        """Snapshot synchrone (lecture seule, sans lock)."""
        return self._store.copy()

    def __repr__(self) -> str:
        return (
            f"<SharedMemory keys={list(self._store.keys())} "
            f"history={len(self._history)}>"
        )

