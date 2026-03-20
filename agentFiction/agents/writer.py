# agents/writer.py
# Agent writer : rédige la suite de l'histoire en s'appuyant sur le contexte complet

import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.writer_prompt import WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def _parse_writer_response(raw: str) -> dict:
    """Parse la réponse JSON de l'agent writer."""
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        print (raw)
        print (" la suite en brute")
        print (raw[start:end])
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"Writer JSON parse error: {e}")

    # Fallback : retourne le texte brut comme suite
    return {
        "suite_ecrite":          raw.strip(),
        "personnages_impliques": [],
        "evenements_cles":       [],
        "ton_narratif":          "inconnu",
        "point_de_fin":          "",
        "avertissements":        ["JSON invalide, texte brut retourné"],
    }


def run_writer(state: StoryState) -> StoryState:
    """
    Agent writer : rédige la suite de l'histoire.

    Activé uniquement si routing_decision == "write".
    Prend en compte les corrections utilisateur via writer_correction.
    Utilise le contexte complet de l'analyste.
    """
    # ── Guard : on n'écrit que si l'orchestrateur l'a décidé ─────────────
    if state.get("routing_decision") != "write":
        logger.info("Writer: chemin 'ideas' détecté, agent ignoré.")
        return {**state, "written_continuation": None}

    existing_story = state.get("existing_story", "").strip()
    if not existing_story:
        return {
            **state,
            "written_continuation": None,
            "errors": state.get("errors", []) + ["Writer: histoire vide, impossible d'écrire la suite."],
        }

    iteration = state.get("writer_iteration", 0) + 1
    correction = state.get("writer_correction", "").strip()

    print(f"\n✍️  [AGENT WRITER] Rédaction de la suite... (itération {iteration})")
    if correction:
        print(f"   📝 Correction demandée : {correction[:80]}...")

    # ── Enrichissement du prompt avec la correction si fournie ───────────
    user_request = state.get("user_request", "").strip()
    if correction:
        user_request = f"{user_request}\n\n[CORRECTION DEMANDÉE PAR L'AUTEUR]: {correction}"

    # ── Préparation du contexte complet (issu de l'analyste) ─────────────
    user_message = WRITER_USER_TEMPLATE.format(
        existing_story    = existing_story,
        characters_json   = json.dumps(state.get("characters_summary", {}),
                                       ensure_ascii=False, indent=2),
        plots_json        = json.dumps(state.get("plots_summary", {}),
                                       ensure_ascii=False, indent=2),
        consistency_json  = json.dumps(state.get("consistency_report", {}),
                                       ensure_ascii=False, indent=2),
        story_ideas_json  = json.dumps(state.get("story_ideas", []),
                                       ensure_ascii=False, indent=2),
        user_request      = user_request,
    )

    # ── Appel LLM ────────────────────────────────────────────────────────
    try:
        llm = ChatOllama(
            model=state["model_name"],
            temperature=0.75,     # Créativité modérée pour cohérence
        )
        print("   📡 Rédaction en cours...")
        response = llm.invoke([
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        parsed = _parse_writer_response(response.content)

        nb_chars = len(parsed.get("suite_ecrite", ""))
        print(f"   ✅ Suite écrite : {nb_chars} caractères")
        print(f"   🎭 Ton narratif : {parsed.get('ton_narratif', 'N/A')}")
        print(f"   📍 Point de fin : {parsed.get('point_de_fin', 'N/A')[:80]}")

        return {
            **state,
            "written_continuation": parsed,
            "writer_iteration": iteration,
        }

    except Exception as e:
        logger.error(f"Writer error: {e}")
        return {
            **state,
            "written_continuation": None,
            "writer_iteration": iteration,
            "errors": state.get("errors", []) + [f"Writer: {str(e)}"],
        }
