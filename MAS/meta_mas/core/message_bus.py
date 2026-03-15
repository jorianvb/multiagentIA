"""
Bus de messages asynchrone pour la communication inter-agents.

Supporte unicast, broadcast et multicast via asyncio.Queue.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from core.base_agent import Message, MessageType


class MessageBus:
    """
    Bus de messages centralisé pour l'orchestration des agents.

    Patterns supportés :
    - Unicast   : message à un agent précis (receiver = agent_name)
    - Broadcast : message à tous les agents (receiver = "broadcast")
    - Multicast : message à un groupe (receiver = "agent1,agent2,...")

    Fonctionnalités :
    - Historique configurable
    - Middleware chainable (logging, filtrage, transformation)
    - Statistiques en temps réel
    """

    def __init__(self, max_history: int = 1000):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._history: List[Message] = []
        self._max_history = max_history
        self._middleware: List[Callable] = []
        logger.info("MessageBus initialisé")

    # ------------------------------------------------------------------
    # Gestion des agents
    # ------------------------------------------------------------------

    def register_agent(self, agent_name: str) -> None:
        """Enregistre un agent pour recevoir des messages."""
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()
            logger.debug(f"MessageBus: agent '{agent_name}' enregistré")

    def unregister_agent(self, agent_name: str) -> None:
        """Désenregistre un agent."""
        if agent_name in self._queues:
            del self._queues[agent_name]
            logger.debug(f"MessageBus: agent '{agent_name}' désenregistré")

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    async def publish(self, message: Message) -> None:
        """
        Publie un message sur le bus.
        Applique les middlewares puis route vers la/les files appropriées.
        """
        # Middlewares
        for mw in self._middleware:
            message = await mw(message)
            if message is None:
                return

        # Historique
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Routage
        if message.receiver == "broadcast":
            await self._broadcast(message)
        elif "," in message.receiver:
            receivers = [r.strip() for r in message.receiver.split(",")]
            await self._multicast(message, receivers)
        else:
            await self._unicast(message)

    async def _unicast(self, message: Message) -> None:
        """Envoie à un seul agent."""
        receiver = message.receiver
        if receiver not in self._queues:
            logger.error(f"MessageBus: agent inconnu '{receiver}'")
            raise KeyError(f"Agent '{receiver}' introuvable dans le bus")
        await self._queues[receiver].put(message)
        logger.debug(
            f"MessageBus: unicast {message.sender} → {receiver} [{message.type}]"
        )

    async def _broadcast(self, message: Message) -> None:
        """Envoie à tous les agents sauf l'expéditeur."""
        count = 0
        for agent_name, queue in self._queues.items():
            if agent_name != message.sender:
                await queue.put(message)
                count += 1
        logger.debug(
            f"MessageBus: broadcast depuis {message.sender} → {count} agents"
        )

    async def _multicast(self, message: Message, receivers: List[str]) -> None:
        """Envoie à un groupe d'agents."""
        for receiver in receivers:
            if receiver in self._queues:
                msg_copy = message.model_copy(update={"receiver": receiver})
                await self._queues[receiver].put(msg_copy)
            else:
                logger.warning(f"MessageBus: cible multicast '{receiver}' inconnue")
        logger.debug(
            f"MessageBus: multicast depuis {message.sender} → {receivers}"
        )

    # ------------------------------------------------------------------
    # Réception
    # ------------------------------------------------------------------

    async def receive(
        self, agent_name: str, timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Reçoit le prochain message pour un agent.

        Args:
            agent_name: Nom de l'agent récepteur.
            timeout: Timeout optionnel en secondes.

        Returns:
            Message ou None si timeout.
        """
        if agent_name not in self._queues:
            raise KeyError(f"Agent '{agent_name}' non enregistré dans le bus")
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._queues[agent_name].get(), timeout=timeout
                )
            return await self._queues[agent_name].get()
        except asyncio.TimeoutError:
            return None

    # ------------------------------------------------------------------
    # Middleware & stats
    # ------------------------------------------------------------------

    def add_middleware(self, middleware: Callable) -> None:
        """Ajoute une fonction middleware de traitement des messages."""
        self._middleware.append(middleware)

    def get_history(
        self,
        agent_name: Optional[str] = None,
        msg_type: Optional[MessageType] = None,
    ) -> List[Message]:
        """Retourne l'historique filtré optionnellement."""
        history = self._history.copy()
        if agent_name:
            history = [
                m for m in history
                if m.sender == agent_name or m.receiver == agent_name
            ]
        if msg_type:
            history = [m for m in history if m.type == msg_type]
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du bus."""
        return {
            "registered_agents": list(self._queues.keys()),
            "history_size": len(self._history),
            "queue_sizes": {
                name: q.qsize() for name, q in self._queues.items()
            },
        }

    def clear_history(self) -> None:
        """Vide l'historique des messages."""
        self._history.clear()

