# tests/test_agents.py
# Tests unitaires pour les agents
# Utilise des mocks pour éviter d'appeler Ollama dans les tests

import pytest
import json

from unittest.mock import patch, MagicMock
from state import StoryState
from agents.analyst     import run_analyst, _parse_json_safely
from agents.checker     import run_checker
from agents.ideator     import run_ideator
from agents.synthesizer import run_synthesizer


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> StoryState:
    """État de base pour les tests."""
    return {
            "existing_story"    : "Elara, une mage, découvrit que son mentor l'avait trahie.",
        "user_request"      : "Proposer une suite dramatique",
        "model_name"        : "llama3.1",
        "characters_summary": {},
        "plots_summary"     : {},
        "story_context"     : "",
        "consistency_report": {},
        "story_ideas"       : [],
        "final_response"    : "",
        "iteration_count"   : 0,
        "session_id"        : "test-session-001",
        "timestamp"         : "2024-01-01T00:00:00",
        "errors"            : []
    }


@pytest.fixture
def state_with_analysis(base_state) -> StoryState:
    """État après l'analyse (pour tester checker/ideator)."""
    return {
        **base_state,
        "story_context": "Elara fuit après la trahison d'Aldric.",
        "characters_summary": {
            "Elara": {
                "nom": "Elara", "role": "protagoniste",
                "traits": ["courageuse", "méfiante"],
                "motivations": ["survivre", "se venger"],
                "statut_actuel": "en fuite",
                "relations": {"Aldric": "mentor-traître"},
                "arcs": ["arc de vengeance"],
                "incertain": False
            }
        },
        "plots_summary": {
            "Trahison d'Aldric": {
                "titre": "Trahison d'Aldric",
                "type": "principale",
                "description": "Le mentor d'Elara l'a trahie",
                "statut": "en_cours",
                "personnages_impliques": ["Elara", "Aldric"],
                "fils_non_resolus": ["Pourquoi Aldric a-t-il trahi ?"],
                "incertaine": False
            }
        }
    }


# ── Tests de _parse_json_safely ────────────────────────────────────────────

def test_parse_json_valid():
    """Test avec un JSON valide."""
    content = '{"key": "value"}'
    result  = _parse_json_safely(content, "TEST")
    assert result == {"key": "value"}


def test_parse_json_with_surrounding_text():
    """Test avec du texte autour du JSON."""
    content = 'Voici le résultat : {"key": "value"} fin.'
    result  = _parse_json_safely(content, "TEST")
    assert result == {"key": "value"}


def test_parse_json_invalid():
    """Test avec un JSON invalide."""
    result = _parse_json_safely("ce n'est pas du json", "TEST")
    assert result is None


# ── Tests des agents avec mock Ollama ──────────────────────────────────────

@patch("agents.analyst.ChatOllama")
def test_analyst_success(mock_ollama, base_state):
    """Test que l'analyste parse correctement une réponse valide."""
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "contexte_actuel": "Elara fuit.",
        "ton_general": "dramatique",
        "univers": "fantasy",
        "dernier_evenement": "trahison",
        "personnages": {
            "Elara": {
                "nom": "Elara", "role": "protagoniste",
                "traits": ["courageuse"], "motivations": ["survivre"],
                "statut_actuel": "en fuite", "relations": {},
                "arcs": [], "incertain": False
            }
        },
        "intrigues": {}
    })
    mock_ollama.return_value.invoke.return_value = mock_response

    result = run_analyst(base_state)

    assert result["story_context"] == "Elara fuit."
    assert "Elara" in result["characters_summary"]
    assert len(result["errors"]) == 0


@patch("agents.synthesizer.ChatOllama")
def test_synthesizer_fallback(mock_ollama, state_with_analysis):
    """Test que le synthétiseur produit une réponse fallback si LLM échoue."""
    mock_ollama.return_value.invoke.side_effect = Exception("LLM indisponible")
    result = run_synthesizer(state_with_analysis)
    assert result["final_response"] != ""



# ── Tests de scoring ───────────────────────────────────────────────────────

def test_scoring_basic():
    """Test du calcul de score de base."""
    import json
    from utils.scoring import calculate_idea_score

    idea = {
        "titre": "Test",
        "type": "dramatique",
        "description": "test description",
        "avantages": [], "risques": [],
        "impact_personnages": {"Elara": "impact"},
        "impact_intrigues": {},
        "detail_score": {
            "coherence": 8.0,
            "potentiel_dramatique": 7.0,
            "respect_personnages": 9.0,
            "originalite": 6.0
        }
    }
    consistency = {"plot_holes": [], "arcs_abandonnes": []}
    result = calculate_idea_score(idea, consistency, {})
    assert 0 <= result["score"] <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
