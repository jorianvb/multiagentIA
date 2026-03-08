# src/agents/critic.py
"""Agent spécialisé dans la critique et l'évaluation du contenu."""

import re
import json
from src.agents.base_agent import BaseAgent
from src.config import get_settings
from src.models import AgentOutput, AgentRole, ReviewScore, WorkflowState
from src.utils.logger import log_agent_output


CRITIC_PROMPT_TEMPLATE = """
Tu dois évaluer rigoureusement ce contenu et retourner une évaluation JSON.

**Sujet traité :** {topic}

**Contenu à évaluer :**
{draft_content}

**Contexte de la recherche :**
{research_summary}

Évalue selon ces critères et retourne UNIQUEMENT un JSON valide :

```json
{{
    "overall_score": <0-10>,
    "accuracy_score": <0-10>,
    "clarity_score": <0-10>,
    "completeness_score": <0-10>,
    "feedback": "<feedback détaillé en français>",
    "needs_revision": <true|false>,
    "revision_instructions": "<instructions précises si révision nécessaire, sinon vide>"
}}
Règles de scoring :

8-10 : Excellent, publication possible
6-7 : Bien mais améliorable
4-5 : Moyen, révision recommandée  
0-3 : Insuffisant, révision obligatoire

needs_revision = true si overall_score < 7.0
Retourne UNIQUEMENT le JSON, sans texte avant ou après.
"""
class CriticAgent(BaseAgent):

    def __init__(self):
        settings = get_settings()
        super().__init__(settings.critic_config)

    def build_prompt(self, state: WorkflowState) -> str:
        """Construit le prompt d'évaluation."""
        # Résumé de la recherche (premiers 500 chars)
        research_summary = (
            state.research_output[:500] + "..."
            if len(state.research_output) > 500
            else state.research_output
        )

        return CRITIC_PROMPT_TEMPLATE.format(
            topic=state.topic,
            draft_content=state.draft_content,
            research_summary=research_summary
        )

    def _parse_review(self, response: str) -> ReviewScore:
        """
        Parse la réponse JSON du critic.

        Gère les cas où le LLM inclut du texte avant/après le JSON.
        """
        # Stratégie 1 : Parser directement
        try:
            data = json.loads(response.strip())
            return ReviewScore(**data)
        except json.JSONDecodeError:
            pass

        # Stratégie 2 : Extraire le JSON avec regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if 'overall_score' in data:
                    return ReviewScore(**data)
            except (json.JSONDecodeError, ValueError):
                continue

        # Stratégie 3 : Fallback avec score par défaut
        # Si le LLM ne retourne pas de JSON valide
        return ReviewScore(
            overall_score=5.0,
            accuracy_score=5.0,
            clarity_score=5.0,
            completeness_score=5.0,
            feedback=f"Parsing échoué. Réponse brute: {response[:200]}",
            needs_revision=True,
            revision_instructions="Révision générale recommandée"
        )

    def process_response(self, response: str, state: WorkflowState) -> WorkflowState:
        """Parse et intègre le review dans l'état."""
        review = self._parse_review(response)
        state.review = review
        state.current_agent = self.name

        # Log avec le score
        log_agent_output(
            self.name,
            f"Score: {review.overall_score}/10\n{review.feedback}",
            score=review.overall_score
        )

        # Décider de la prochaine étape
        if review.needs_revision and state.revision_count < state.max_revisions:
            state.next_agent = "writer"  # Retour au writer pour révision
            state.revision_count += 1
        else:
            state.next_agent = "end"
            state.final_content = state.draft_content
            state.is_complete = True

        state.agent_outputs.append(AgentOutput(
            agent_name=self.name,
            role=AgentRole.CRITIC,
            content=response,
            metadata={
                "score": review.overall_score,
                "needs_revision": review.needs_revision
            }
        ))

        return state

