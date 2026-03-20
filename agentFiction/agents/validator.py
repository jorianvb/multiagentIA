# agents/validator.py
# Agent validateur : vérifie la qualité et la cohérence de la suite écrite
# avant de la présenter à l'utilisateur

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.validator_prompt import VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_TEMPLATE
from agents.analyst import _parse_json_safely


def run_validator(state: StoryState) -> StoryState:
    """
    Agent validateur : évalue la suite écrite par l'agent writer.

    Entrée  : written_continuation, existing_story, characters_summary,
              plots_summary, user_request
    Sortie  : validation_report
    """
    print("\n🔬 [AGENT VALIDATEUR] Vérification de la suite écrite...")

    written = state.get("written_continuation")
    if not written or not written.get("suite_ecrite", "").strip():
        print("   ℹ️  Aucune suite à valider (chemin 'ideas' ou writer inactif)")
        return {
            **state,
            "validation_report": {
                "score_global": 10.0,
                "is_valid": True,
                "verdict": "APPROUVÉ",
                "points_forts": [],
                "problemes": [],
                "suggestions_amelioration": []
            }
        }

    suite_ecrite = written["suite_ecrite"]

    try:
        llm = ChatOllama(
            model=state["model_name"],
            temperature=0.1,
            format="json"
        )

        user_message = VALIDATOR_USER_TEMPLATE.format(
            existing_story   = state.get("existing_story", "")[:3000],
            characters_json  = json.dumps(state.get("characters_summary", {}),
                                          ensure_ascii=False, indent=2),
            plots_json       = json.dumps(state.get("plots_summary", {}),
                                          ensure_ascii=False, indent=2),
            user_request     = state.get("user_request", ""),
            suite_ecrite     = suite_ecrite
        )

        print("   📡 Validation en cours...")
        response = llm.invoke([
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        report = _parse_json_safely(response.content, "VALIDATEUR")

        if not report:
            raise ValueError("Impossible de parser le rapport de validation")

        score   = report.get("score_global", 7.0)
        verdict = report.get("verdict", "APPROUVÉ")
        nb_pb   = len(report.get("problemes", []))
        nb_crit = sum(1 for p in report.get("problemes", [])
                      if p.get("severite") == "critique")

        # Émoji selon verdict
        icon = {"APPROUVÉ": "✅", "À AMÉLIORER": "⚠️ ", "REJETÉ": "❌"}.get(verdict, "❓")
        print(f"   {icon} Verdict : {verdict} — Score : {score:.1f}/10")
        print(f"   📊 {nb_pb} problème(s) dont {nb_crit} critique(s)")

        for pf in report.get("points_forts", [])[:2]:
            print(f"   💚 {pf}")
        for pb in report.get("problemes", [])[:2]:
            sev_icon = {"critique": "🔴", "important": "🟡", "mineur": "🟢"}.get(
                pb.get("severite", "mineur"), "⚪")
            print(f"   {sev_icon} {pb.get('description', '')}")

        return {**state, "validation_report": report}

    except Exception as e:
        erreur = f"VALIDATEUR: Erreur : {str(e)}"
        print(f"   ❌ {erreur}")
        # En cas d'erreur, on approuve quand même pour ne pas bloquer
        return {
            **state,
            "validation_report": {
                "score_global": 7.0,
                "is_valid": True,
                "verdict": "APPROUVÉ",
                "points_forts": [],
                "problemes": [],
                "suggestions_amelioration": [f"Validation automatique (erreur: {str(e)})"]
            },
            "errors": state.get("errors", []) + [erreur]
        }

