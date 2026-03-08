# src/agents/researcher.py
"""Agent spécialisé dans la recherche et l'analyse."""

from src.agents.base_agent import BaseAgent
from src.config import get_settings
from src.models import AgentOutput, AgentRole, WorkflowState


RESEARCHER_PROMPT_TEMPLATE = """
Sujet à analyser : {topic}

Instructions supplémentaires : {instructions}

Tu dois produire une analyse structurée comprenant :

## 1. Vue d'ensemble
- Définition et contexte
- Importance du sujet

## 2. Points clés
- Liste des éléments essentiels à couvrir
- Faits importants et données

## 3. Sous-thèmes principaux
- Décompose le sujet en 3-5 sous-thèmes
- Pour chaque sous-thème : description brève + points importants

## 4. Angles d'approche recommandés
- Comment aborder ce sujet efficacement
- Exemples concrets à utiliser

## 5. Sources et références suggérées
- Types de sources pertinentes
- Experts ou références dans le domaine

Sois exhaustif mais concis. Fournis des informations factuelles et vérifiables.
"""


class ResearcherAgent(BaseAgent):
    """
    Agent Researcher : analyse et structure l'information.

    Responsabilités :
    - Analyser le sujet en profondeur
    - Identifier les points clés
    - Structurer l'information pour le Writer
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(settings.researcher_config)

    def build_prompt(self, state: WorkflowState) -> str:
        """Construit le prompt de recherche."""
        return RESEARCHER_PROMPT_TEMPLATE.format(
            topic=state.topic,
            instructions=state.instructions or "Analyse générale approfondie"
        )

    def process_response(self, response: str, state: WorkflowState) -> WorkflowState:
        """Stocke la recherche dans l'état."""
        state.research_output = response
        state.current_agent = self.name
        state.next_agent = "writer"

        # Enregistrer dans l'historique
        state.agent_outputs.append(AgentOutput(
            agent_name=self.name,
            role=AgentRole.RESEARCHER,
            content=response,
            metadata={"topic": state.topic}
        ))

        return state
