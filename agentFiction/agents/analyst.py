# agents/analyst.py
# Agent analyste : extrait les données structurées du texte de l'auteur

import json
import re
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.analyst_prompt import ANALYST_SYSTEM_PROMPT, ANALYST_USER_TEMPLATE


def run_analyst(state: StoryState) -> StoryState:
    """
    Agent 1 : Analyse le texte existant et extrait les personnages et intrigues.

    Entrée : existing_story, user_request
    Sortie : characters_summary, plots_summary, story_context
    """
    print("\n🔍 [AGENT ANALYSTE] Analyse du texte en cours...")
    print(f"   Modèle utilisé : {state['model_name']}")
    print(f"   Longueur du texte : {len(state['existing_story'])} caractères")

    # Vérification que le texte existe
    if not state.get("existing_story", "").strip():
        erreur = "ANALYSTE: Aucun texte fourni à analyser."
        print(f"   ❌ Erreur : {erreur}")
        return {
            **state,
            "errors": state.get("errors", []) + [erreur],
            "characters_summary": {},
            "plots_summary": {},
            "story_context": "Aucun texte fourni."
        }

    try:
        # Initialisation du modèle Ollama
        llm = ChatOllama(
            model=state["model_name"],
            temperature=0.1,      # Faible température : on veut de la précision
            format="json"         # Force la sortie JSON
        )

        # Construction du message utilisateur
        user_message = ANALYST_USER_TEMPLATE.format(
            existing_story=state["existing_story"],
            user_request=state.get("user_request", "Analyse générale")
        )

        # Appel au LLM
        print("   📡 Envoi au LLM...")
        response = llm.invoke([
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        # Parsing du JSON retourné
        raw_content = response.content
        analysis = _parse_json_safely(raw_content, "ANALYSTE")

        if not analysis:
            raise ValueError("Impossible de parser la réponse JSON de l'analyste")

        # Extraction des données
        characters = analysis.get("personnages", {})
        intrigues = analysis.get("intrigues", {})
        context = analysis.get("contexte_actuel", "Contexte non déterminé")

        print(f"   ✅ {len(characters)} personnage(s) identifié(s)")
        print(f"   ✅ {len(intrigues)} intrigue(s) identifiée(s)")
        print(f"   ✅ Contexte : {context[:80]}...")

        return {
            **state,
            "characters_summary": characters,
            "plots_summary": intrigues,
            "story_context": context,
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    except ConnectionError:
        erreur = "ANALYSTE: Impossible de se connecter à Ollama. Vérifiez qu'il est lancé (ollama serve)."
        print(f"   ❌ {erreur}")
        return {**state, "errors": state.get("errors", []) + [erreur]}

    except Exception as e:
        erreur = f"ANALYSTE: Erreur inattendue : {str(e)}"
        print(f"   ❌ {erreur}")
        return {**state, "errors": state.get("errors", []) + [erreur],
                "characters_summary": {}, "plots_summary": {}, "story_context": "Erreur d'analyse"}


def _parse_json_safely(content: str, agent_name: str) -> dict | None:
    """
    Tente de parser un JSON même si le LLM a ajouté du texte autour.
    Stratégie : cherche le premier { et le dernier }
    """
    try:
        # Tentative directe
        return json.loads(content)
    except json.JSONDecodeError:
        # Tentative avec extraction du bloc JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                print(f"   ⚠️  [{agent_name}] JSON malformé, impossible de parser")
                return None
    return None
