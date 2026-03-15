"""
BaseAgent abstrait pour le Meta-MAS.

Tous les agents héritent de cette classe et implémentent la méthode `act`.
"""
import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """États possibles d'un agent."""

    IDLE = "IDLE"
    THINKING = "THINKING"
    ACTING = "ACTING"
    WAITING = "WAITING"
    ERROR = "ERROR"


class MessageType(str, Enum):
    """Types de messages inter-agents."""

    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    BROADCAST = "BROADCAST"
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"


class Message(BaseModel):
    """Format standard d'un message inter-agents."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str  # "broadcast" pour diffusion globale
    type: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<Message id={self.id[:8]} "
            f"from={self.sender} to={self.receiver} "
            f"type={self.type} len={len(self.content)}>"
        )


class BaseAgent(ABC):
    """
    Classe abstraite de base pour tous les agents du Meta-MAS.

    Chaque agent possède :
    - Un nom unique et un rôle
    - Un modèle LLM via Ollama
    - Un system_prompt spécialisé
    - Une mémoire conversationnelle (historique)
    - Un statut (IDLE | THINKING | ACTING | WAITING | ERROR)
    - Une connexion au bus de messages

    Usage minimal :
        class MonAgent(BaseAgent):
            async def act(self, message: Message) -> Message:
                response = await self.think({"task": message.content})
                return Message(sender=self.name, receiver=message.sender,
                               type=MessageType.RESPONSE, content=response)
    """

    def __init__(
        self,
        name: str,
        role: str,
        model: str,
        system_prompt: str,
        message_bus=None,
        timeout: int = 120,
    ):
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt
        self.message_bus = message_bus
        self.timeout = timeout
        self.status: AgentStatus = AgentStatus.IDLE
        self.memory: List[Dict[str, str]] = []
        self._inbox: asyncio.Queue = asyncio.Queue()

        logger.info(f"[{self.name}] ✨ Initialisé | rôle={self.role} | modèle={self.model}")

    # ------------------------------------------------------------------
    # Méthodes principales
    # ------------------------------------------------------------------

    async def think(self, context: dict) -> str:
        """
        Appel LLM pour générer une réponse basée sur le contexte.

        Args:
            context: Dict décrivant la tâche en cours.

        Returns:
            Réponse textuelle du LLM.
        """
        from utils.ollama_client import OllamaClient

        self.status = AgentStatus.THINKING
        logger.info(f"[{self.name}] 🤔 En train de réfléchir...")

        # Construction du historique de conversation
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Ajouter les 10 derniers échanges en mémoire
        for exchange in self.memory[-10:]:
            messages.append(exchange)

        # Ajouter le contexte actuel
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        messages.append({"role": "user", "content": context_str})

        client = OllamaClient(model=self.model, timeout=self.timeout)
        try:
            response = await client.chat(messages)
        finally:
            await client.close()

        # Mémoriser l'échange
        self.memory.append({"role": "user", "content": context_str})
        self.memory.append({"role": "assistant", "content": response})

        self.status = AgentStatus.IDLE
        logger.info(f"[{self.name}] ✅ Réflexion terminée ({len(response)} chars)")
        return response

    @abstractmethod
    async def act(self, message: Message) -> Message:
        """
        Méthode d'action principale — traite un message et retourne une réponse.

        Args:
            message: Message entrant à traiter.

        Returns:
            Message de réponse.
        """
        ...

    async def communicate(
        self,
        target: str,
        content: str,
        msg_type: MessageType = MessageType.REQUEST,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Envoie un message à un autre agent via le bus de messages.

        Args:
            target: Nom de l'agent cible ou "broadcast".
            content: Contenu du message.
            msg_type: Type de message.
            metadata: Métadonnées optionnelles.
        """
        if self.message_bus is None:
            raise RuntimeError(f"[{self.name}] Aucun bus de messages connecté")

        message = Message(
            sender=self.name,
            receiver=target,
            type=msg_type,
            content=content,
            metadata=metadata or {},
        )

        self.status = AgentStatus.ACTING
        await self.message_bus.publish(message)
        logger.debug(f"[{self.name}] 📤 Envoyé {msg_type} → {target}")
        self.status = AgentStatus.IDLE

    async def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Reçoit le prochain message de la file d'attente.

        Args:
            timeout: Timeout optionnel en secondes.

        Returns:
            Message reçu ou None si timeout.
        """
        self.status = AgentStatus.WAITING
        try:
            if timeout is not None:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            else:
                msg = await self._inbox.get()
            logger.debug(f"[{self.name}] 📥 Reçu message de {msg.sender}")
            return msg
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] ⏰ Timeout en attente de message")
            return None
        finally:
            self.status = AgentStatus.IDLE

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def add_to_inbox(self, message: Message) -> None:
        """Ajoute un message directement dans la file d'attente."""
        self._inbox.put_nowait(message)

    def clear_memory(self) -> None:
        """Efface l'historique conversationnel."""
        self.memory.clear()
        logger.debug(f"[{self.name}] Mémoire effacée")

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état complet de l'agent."""
        return {
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "status": self.status.value,
            "memory_size": len(self.memory),
            "inbox_size": self._inbox.qsize(),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name} role={self.role} status={self.status}>"
        )

