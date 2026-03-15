"""
Registre centralisé des agents actifs du Meta-MAS.
"""
from typing import Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from core.base_agent import BaseAgent, AgentStatus


class AgentRegistry:
    """
    Registre singleton pour gérer tous les agents actifs.

    Fonctionnalités :
    - Enregistrement / désenregistrement d'agents
    - Recherche par nom ou par rôle
    - Monitoring du statut
    - Itérabilité
    """

    _instance: Optional["AgentRegistry"] = None

    def __init__(self):
        self._agents: Dict[str, "BaseAgent"] = {}
        logger.info("AgentRegistry initialisé")

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Accès singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Réinitialise le singleton (utile pour les tests)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, agent: "BaseAgent") -> None:
        """Enregistre un agent. Écrase l'existant si même nom."""
        if agent.name in self._agents:
            logger.warning(
                f"Registry: agent '{agent.name}' déjà enregistré, remplacement"
            )
        self._agents[agent.name] = agent
        logger.info(f"Registry: ✅ '{agent.name}' [{agent.role}] enregistré")

    def unregister(self, agent_name: str) -> None:
        """Supprime un agent du registre."""
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info(f"Registry: '{agent_name}' retiré")
        else:
            logger.warning(f"Registry: agent '{agent_name}' introuvable")

    def get(self, name: str) -> Optional["BaseAgent"]:
        """Récupère un agent par son nom."""
        return self._agents.get(name)

    def get_by_role(self, role: str) -> List["BaseAgent"]:
        """Récupère tous les agents ayant un rôle donné."""
        return [a for a in self._agents.values() if a.role == role]

    def get_all(self) -> List["BaseAgent"]:
        """Retourne tous les agents enregistrés."""
        return list(self._agents.values())

    def get_status_all(self) -> Dict[str, Dict]:
        """Retourne le statut de tous les agents."""
        return {name: agent.get_status() for name, agent in self._agents.items()}

    def get_idle_agents(self) -> List["BaseAgent"]:
        """Retourne les agents actuellement IDLE."""
        from core.base_agent import AgentStatus
        return [a for a in self._agents.values() if a.status == AgentStatus.IDLE]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def count(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"<AgentRegistry agents={list(self._agents.keys())}>"

