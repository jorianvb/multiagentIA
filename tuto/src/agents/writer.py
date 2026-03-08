# src/agents/writer.py
"""Agent spécialisé dans la rédaction de contenu."""

from src.agents.base_agent import BaseAgent
from src.config import get_settings
from src.models import AgentOutput, AgentRole, WorkflowState


WRITER_PROMPT_TEMPLATE = """
Tu dois rédiger un contenu de qualité sur le sujet suivant.

**Sujet :** {topic}

**Instructions :** {instructions}

**Recherches effectuées :**
{research_output}

{revision_context}

Rédige un contenu complet, engageant et bien structuré qui :
1. Commence par une introduction percutante
2. Développe chaque point clé avec clarté
3. Utilise des exemples concrets
4. Conclut de manière mémorable

Format : Utilise des titres (##), des listes et une mise en forme claire.
Longueur cible : 400-600 mots.
"""

REVISION_CONTEXT_TEMPLATE = """
**⚠️ RÉVISION DEMANDÉE (Révision #{revision_count})**

Contenu précédent à améliorer :
{previous_content}

Feedback du critique :
{feedback}

Instructions de révision :
{revision_instructions}

Corrige les problèmes identifiés tout en préservant les points forts.
"""


class WriterAgent(BaseAgent):
    """
    Agent Writer : crée du contenu à partir des recherches.

    Responsabilités :
    - Rédiger le contenu initial
    - Intégrer les révisions demandées par le Critic
    - Adapter le style selon les instructions
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(settings.writer_config)

    def build_prompt(self, state: WorkflowState) -> str:
        """Construit le prompt de rédaction, avec contexte de révision si nécessaire."""

        # Contexte de révision (si c'est une révision)
        revision_context = ""
        if state.revision_count > 0 and state.review:
            revision_context = REVISION_CONTEXT_TEMPLATE.format(
                revision_count=state.revision_count,
                previous_content=state.draft_content,
                feedback=state.review.feedback,
                revision_instructions=state.review.revision_instructions
            )

        return WRITER_PROMPT_TEMPLATE.format(
            topic=state.topic,
            instructions=state.instructions or "Rédige un article informatif",
            research_output=state.research_output,
            revision_context=revision_context
        )

    def process_response(self, response: str, state: WorkflowState) -> WorkflowState:
        """Stocke le brouillon et met à jour l'état."""
        state.draft_content = response
        state.current_agent = self.name
        state.next_agent = "critic"

        state.agent_outputs.append(AgentOutput(
            agent_name=self.name,
            role=AgentRole.WRITER,
            content=response,
            metadata={
                "revision_number": state.revision_count,
                "is_revision": state.revision_count > 0
            }
        ))

        return state
