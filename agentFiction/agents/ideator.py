# agents/ideator.py
# Agent créatif : génère des idées de suite pour l'histoire

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.ideator_prompt import IDEATOR_SYSTEM_PROMPT, IDEATOR_USER_TEMPLATE
from agents.analyst import _parse_json_safely


def run_ideator(state: StoryState) -> StoryState:
    """
    Agent 3 : Génère des idées créatives pour la suite de l'histoire.

    Entrée : characters_summary, plots_summary, consistency_report, user_request
    Sortie : story_ideas (liste d'idées scorées)
    """
    print("\n💡 [AGENT CRÉATIF] Génération des idées de suite...")

    try:
        llm = ChatOllama(
            model=state["model_name"],
            temperature=0.7,    # Température plus haute : on veut de la créativité
            format="json"
        )

        # Extraction des données de cohérence pour informer la créativité
        consistency = state.get("consistency_report", {})
        coherence_points = consistency.get("points_coherents", [])
        warnings = consistency.get("warnings", [])
        vigilance = consistency.get("points_vigilance_futurs", [])

        user_message = IDEATOR_USER_TEMPLATE.format(
            story_context=state.get("story_context", "Non disponible"),
            characters_json=json.dumps(state.get("characters_summary", {}),
                                       ensure_ascii=False, indent=2),
            plots_json=json.dumps(state.get("plots_summary", {}),
                                  ensure_ascii=False, indent=2),
            coherence_points=json.dumps(coherence_points, ensure_ascii=False),
            warnings=json.dumps(warnings, ensure_ascii=False),
            vigilance_points=json.dumps(vigilance, ensure_ascii=False),
            user_request=state.get("user_request", "Proposer une suite intéressante")
        )

        print("   📡 Génération créative en cours...")
        response = llm.invoke([
            SystemMessage(content=IDEATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        ideas_data = _parse_json_safely(response.content, "IDEATOR")

        if not ideas_data:
            raise ValueError("Impossible de parser les idées créatives")

        idees = ideas_data.get("idees", [])

        # Tri par score décroissant (sécurité si le LLM ne l'a pas fait)
        idees_triees = sorted(idees, key=lambda x: x.get("score", 0), reverse=True)

        print(f"   ✅ {len(idees_triees)} idée(s) générée(s)")
        for i, idee in enumerate(idees_triees):
            print(f"   {'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '  '} "
                  f"[{idee.get('score', 0):.1f}/10] {idee.get('titre', 'Sans titre')}")

        return {
            **state,
            "story_ideas": idees_triees
        }

    except ConnectionError:
        erreur = "IDEATOR: Connexion Ollama impossible."
        print(f"   ❌ {erreur}")
        return {**state, "errors": state.get("errors", []) + [erreur], "story_ideas": []}

    except Exception as e:
        erreur = f"IDEATOR: Erreur : {str(e)}"
        print(f"   ❌ {erreur}")
        return {**state, "errors": state.get("errors", []) + [erreur], "story_ideas": []}
