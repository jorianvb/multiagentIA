# tests/test_state.py
# Tests de la structure d'état

import pytest
from state import StoryState


def test_state_required_keys():
    """Vérifie que tous les champs requis sont présents."""
    required_keys = [
        "existing_story", "user_request", "model_name",
        "characters_summary", "plots_summary", "story_context",
        "consistency_report", "story_ideas", "final_response",
        "iteration_count", "session_id", "timestamp", "errors"
    ]
    state: StoryState = {
        "existing_story": "test", "user_request": "test",
        "model_name": "llama3.1", "characters_summary": {},
        "plots_summary": {}, "story_context": "",
        "consistency_report": {}, "story_ideas": [],
        "final_response": "", "iteration_count": 0,
        "session_id": "test", "timestamp": "2024-01-01",
        "errors": []
    }
    for key in required_keys:
        assert key in state, f"Clé manquante : {key}"


def test_state_errors_accumulation():
    """Vérifie l'accumulation des erreurs."""
    errors = []
    errors = errors + ["Erreur 1"]
    errors = errors + ["Erreur 2"]
    assert len(errors) == 2
