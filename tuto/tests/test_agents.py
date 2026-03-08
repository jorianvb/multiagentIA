# tests/test_agents.py
"""Tests pour les agents et le workflow."""

import pytest
from unittest.mock import MagicMock, patch

from src.models import WorkflowState, ReviewScore
from src.agents.critic import CriticAgent
from src.agents.researcher import ResearcherAgent


class TestCriticAgent:
    """Tests pour le CriticAgent, notamment le parsing JSON."""

    def setup_method(self):
        """Setup avant chaque test."""
        with patch('src.agents.base_agent.ChatOllama'):
            self.critic = CriticAgent()

    def test_parse_valid_json(self):
        """Test du parsing d'un JSON valide."""
        valid_json = '''
        {
            "overall_score": 8.5,
            "accuracy_score": 9.0,
            "clarity_score": 8.0,
            "completeness_score": 8.5,
            "feedback": "Excellent contenu",
            "needs_revision": false,
            "revision_instructions": ""
        }
        '''
        result = self.critic._parse_review(valid_json)

        assert result.overall_score == 8.5
        assert result.needs_revision == False
        assert result.feedback == "Excellent contenu"

    def test_parse_json_with_surrounding_text(self):
        """Test extraction JSON depuis du texte mixte."""
        mixed_response = '''
        Voici mon évaluation du contenu :
        
        {
            "overall_score": 6.0,
            "accuracy_score": 7.0,
            "clarity_score": 5.5,
            "completeness_score": 5.5,
            "feedback": "Contenu correct mais incomplet",
            "needs_revision": true,
            "revision_instructions": "Ajouter des exemples concrets"
        }
        
        J'espère que cette évaluation vous aide.
        '''
        result = self.critic._parse_review(mixed_response)

        assert result.overall_score == 6.0
        assert result.needs_revision == True

    def test_parse_invalid_json_fallback(self):
        """Test du fallback quand le JSON est invalide."""
        invalid_response = "Je pense que c'est bien écrit, environ 7/10."

        result = self.critic._parse_review(invalid_response)

        # Doit retourner un ReviewScore par défaut
        assert isinstance(result, ReviewScore)
        assert result.overall_score == 5.0
        assert result.needs_revision == True

    def test_route_logic_max_revisions(self):
        """Test que le routing respecte max_revisions."""
        state = WorkflowState(
            topic="Test",
            revision_count=3,
            max_revisions=3,
            review=ReviewScore(
                overall_score=5.0,
                accuracy_score=5.0,
                clarity_score=5.0,
                completeness_score=5.0,
                feedback="Révision nécessaire",
                needs_revision=True
            )
        )

        # Simuler le routage
        from src.graphs.advanced_graph import route_after_critic
        result = route_after_critic(state)

        # Doit finir malgré needs_revision=True (max atteint)
        assert result == "end"


class TestWorkflowState:
    """Tests pour le modèle d'état."""

    def test_state_initialization(self):
        """Test l'initialisation de l'état."""
        state = WorkflowState(
            topic="Test IA",
            instructions="Article court"
        )

        assert state.topic == "Test IA"
        assert state.revision_count == 0
        assert state.is_complete == False
        assert state.agent_outputs == []

    def test_state_accumulation(self):
        """Test l'accumulation des messages."""
        state = WorkflowState(topic="Test")
        # Les listes annotées avec operator.add s'accumulent
        assert isinstance(state.messages, list)


# Tests d'intégration (nécessitent Ollama)
@pytest.mark.integration
class TestIntegration:
    """Tests d'intégration (nécessitent Ollama lancé)."""

    @pytest.mark.slow
    def test_simple_workflow(self):
        """Test complet du workflow simple."""
        from src.graphs.simple_graph import run_simple_workflow

        result = run_simple_workflow(
            topic="Les avantages du café",
            instructions="Article court de 200 mots"
        )

        assert result.research_output != ""
        assert result.draft_content != ""

    @pytest.mark.slow
    def test_advanced_workflow_completes(self):
        """Test que le workflow avancé se termine."""
        from src.graphs.advanced_graph import run_advanced_workflow

        result = run_advanced_workflow(
            topic="Les bienfaits du sport",
            max_revisions=1,  # Limiter pour la rapidité
            thread_id="test-session"
        )

        assert result.is_complete == True
        assert len(result.agent_outputs) >= 3  # Au moins 3 agents
