# agents/checker.py
# Agent de cohérence : vérifie les incohérences dans l'histoire

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.checker_prompt import CHECKER_SYSTEM_PROMPT, CHECKER_USER_TEMPLATE
from agents.analyst import _parse_json_safely


def run_checker(state: StoryState) -> StoryState:
    """
    Agent 2 : Vérifie la cohérence de l'histoire analysée.

    Entrée : characters_summary, plots_summary, existing_story
    Sortie : consistency_report
    """
    print("\n🔎 [AGENT COHÉRENCE] Vérification de la cohérence...")

    # Vérification des prérequis
    if not state.get("characters_summary") and not state.get("plots_summary"):
        erreur = "CHECKER: Aucune analyse disponible depuis l'agent Analyste."
        print(f"   ❌ {erreur}")
        return {
            **state,
            "errors": state.get("errors", []) + [erreur],
            "consistency_report": {"score_coherence_global": 0, "warnings": [], "plot_holes": []}
        }

    try:
        llm = ChatOllama(
            model=state["model_name"],
            temperature=0.1,
            format="json"
        )

        # Préparation des données pour le checker
        user_message = CHECKER_USER_TEMPLATE.format(
            story_context=state.get("story_context", "Non disponible"),
            characters_json=json.dumps(state.get("characters_summary", {}),
                                       ensure_ascii=False, indent=2),
            plots_json=json.dumps(state.get("plots_summary", {}),
                                  ensure_ascii=False, indent=2),
            existing_story=state["existing_story"]
        )

        print("   📡 Analyse de cohérence en cours...")
        response = llm.invoke([
            SystemMessage(content=CHECKER_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        report = _parse_json_safely(response.content, "CHECKER")

        if not report:
            raise ValueError("Impossible de parser le rapport de cohérence")

        # Logs de résultat
        score = report.get("score_coherence_global", "N/A")
        nb_warnings = len(report.get("warnings", []))
        nb_critiques = sum(1 for w in report.get("warnings", [])
                           if w.get("severite") == "critique")

        print(f"   ✅ Score de cohérence : {score}/10")
        print(f"   ✅ {nb_warnings} warning(s) dont {nb_critiques} critique(s)")
        print(f"   ✅ {len(report.get('plot_holes', []))} plot hole(s) détecté(s)")

        return {
            **state,
            "consistency_report": report
        }

    except ConnectionError:
        erreur = "CHECKER: Connexion Ollama impossible."
        print(f"   ❌ {erreur}")
        rapport_vide = {
            "score_coherence_global": 5,
            "points_coherents": [],
            "warnings": [],
            "plot_holes": [],
            "suggestions_correction": [],
            "points_vigilance_futurs": []
        }
        return {**state, "errors": state.get("errors", []) + [erreur],
                "consistency_report": rapport_vide}

    except Exception as e:
        erreur = f"CHECKER: Erreur : {str(e)}"
        print(f"   ❌ {erreur}")
        return {**state, "errors": state.get("errors", []) + [erreur],
                "consistency_report": {}}
