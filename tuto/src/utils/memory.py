# src/utils/memory.py
"""Utilitaires pour la gestion de la mémoire et du contexte."""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langchain_ollama import ChatOllama


class ContextManager:
    """
    Gère le contexte et la fenêtre de tokens pour éviter
    de dépasser les limites du modèle.
    """

    def __init__(self, max_tokens: int = 3000, model: str = "llama3.2"):
        self.max_tokens = max_tokens
        self.model = model

    def trim_context(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Tronque les messages pour rester dans la fenêtre de contexte.

        Stratégie : garder le premier message (system) et
        les N derniers messages récents.
        """
        if not messages:
            return messages

        # Utiliser trim_messages de LangChain
        trimmed = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            strategy="last",           # Garder les plus récents
            token_counter=len,         # Approximation simple
            include_system=True,       # Toujours garder le system prompt
            allow_partial=False,
            start_on="human"           # Commencer sur un message humain
        )

        return trimmed

    def summarize_if_needed(
            self,
            content: str,
            max_chars: int = 2000,
            llm: ChatOllama | None = None
    ) -> str:
        """
        Résume le contenu s'il est trop long.

        Utile pour passer le contexte entre agents sans
        dépasser la fenêtre de tokens.
        """
        if len(content) <= max_chars:
            return content

        if llm is None:
            # Troncature simple si pas de LLM disponible
            return content[:max_chars] + "\n\n[... contenu tronqué ...]"

        # Résumé intelligent avec le LLM
        summary_prompt = f"""Résume ce texte en maximum 500 mots en préservant 
        les informations essentielles:
        
        {content}
        
        Résumé:"""

        response = llm.invoke([HumanMessage(content=summary_prompt)])
        return response.content


class ConversationMemory:
    """
    Mémoire de conversation avec fenêtre glissante.
    Utile pour les workflows interactifs.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._messages: list[dict] = []

    def add_message(self, role: str, content: str, agent: str = "") -> None:
        """Ajoute un message à la mémoire."""
        self._messages.append({
            "role": role,
            "content": content,
            "agent": agent
        })

        # Maintenir la fenêtre glissante
        if len(self._messages) > self.window_size * 2:
            # Garder toujours le premier message (context initial)
            self._messages = [self._messages[0]] + self._messages[-(self.window_size):]

    def get_context(self) -> str:
        """Retourne le contexte formaté pour les prompts."""
        if not self._messages:
            return ""

        lines = ["=== Historique de la conversation ==="]
        for msg in self._messages[-self.window_size:]:
            agent_label = f" [{msg['agent']}]" if msg['agent'] else ""
            lines.append(f"{msg['role'].upper()}{agent_label}: {msg['content'][:200]}...")

        return "\n".join(lines)

    def clear(self) -> None:
        """Efface la mémoire."""
        self._messages.clear()
