# src/agents/base_agent.py
"""Classe de base pour tous les agents."""

from abc import ABC, abstractmethod
from typing import Any
import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.config import AgentConfig, get_settings
from src.models import AgentOutput, AgentRole, WorkflowState
from src.utils.logger import log_agent_start, log_agent_output, log_error

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Classe de base abstraite pour tous les agents.

    Chaque agent hérite de cette classe et implémente :
    - `build_prompt()` : construit le prompt spécifique
    - `process_response()` : traite la réponse du LLM
    - `run()` : point d'entrée principal
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.settings = get_settings()

        # Initialisation du LLM via Ollama
        self.llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=config.model,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            # Options avancées Ollama
            options={
                "num_ctx": 4096,        # Fenêtre de contexte
                "repeat_penalty": 1.1,  # Évite les répétitions
                "top_k": 40,
                "top_p": 0.9,
            }
        )

        logger.info(
            "Agent initialisé",
            agent=self.name,
            model=config.model,
            temperature=config.temperature
        )

    def _build_messages(self, human_input: str) -> list:
        """Construit la liste de messages pour le LLM."""
        messages = []

        # System prompt de l'agent
        if self.config.system_prompt:
            messages.append(SystemMessage(content=self.config.system_prompt))

        # Message utilisateur
        messages.append(HumanMessage(content=human_input))

        return messages

    def _call_llm(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        """
        Appel sécurisé au LLM avec gestion des erreurs.

        Args:
            prompt: Le prompt à envoyer
            context: Contexte optionnel pour le logging

        Returns:
            La réponse du LLM sous forme de string
        """
        try:
            messages = self._build_messages(prompt)
            response = self.llm.invoke(messages)

            # Extraire le contenu de la réponse
            if hasattr(response, 'content'):
                return response.content
            return str(response)

        except ConnectionError as e:
            error_msg = f"Impossible de se connecter à Ollama: {e}"
            log_error(self.name, error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = f"Erreur LLM inattendue: {type(e).__name__}: {e}"
            log_error(self.name, error_msg)
            raise RuntimeError(error_msg) from e

    @abstractmethod
    def build_prompt(self, state: WorkflowState) -> str:
        """Construit le prompt spécifique à l'agent."""
        ...

    @abstractmethod
    def process_response(self, response: str, state: WorkflowState) -> WorkflowState:
        """Traite et intègre la réponse dans l'état."""
        ...

    def run(self, state: WorkflowState) -> WorkflowState:
        """
        Point d'entrée principal de l'agent.

        C'est cette méthode qui est appelée par LangGraph
        lors de l'exécution d'un nœud.
        """
        log_agent_start(self.name, state.topic)

        try:
            # 1. Construire le prompt
            prompt = self.build_prompt(state)

            # 2. Appeler le LLM
            response = self._call_llm(prompt)

            # 3. Traiter la réponse
            updated_state = self.process_response(response, state)

            # 4. Logger la sortie
            log_agent_output(self.name, response[:300])

            return updated_state

        except RuntimeError as e:
            # Gestion gracieuse des erreurs
            state.error_message = str(e)
            log_error(self.name, str(e))
            return state
