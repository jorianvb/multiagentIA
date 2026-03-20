# agents/orchestrator.py
# Agent orchestrateur : analyse l'intention de l'utilisateur et route le workflow

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT, ORCHESTRATOR_USER_TEMPLATE


# Mots-clés de déclenchement rapide pour éviter un appel LLM si évident
_WRITE_TRIGGERS = {
    "écris", "ecris", "rédige", "redige", "continue", "continues",
    "développe", "developpe", "raconte", "montre", "suite",
    "passage", "scène", "scene", "réécris", "reecris",
}
_IDEAS_TRIGGERS = {
    "idée", "idee", "idées", "idees", "analyse", "analyser",
    "propose", "suggestion", "suggestions", "piste", "pistes",
    "direction", "directions", "réfléchir", "reflechir",
    "incohérence", "incoherence", "vérif", "verif",
}


def _fast_detect(user_request: str) -> str | None:
    """
    Détection rapide par mots-clés avant d'appeler le LLM.
    Retourne 'write', 'ideas' ou None si ambigu.
    """
    words = set(re.findall(r'\w+', user_request.lower()))
    has_write  = bool(words & _WRITE_TRIGGERS)
    has_ideas  = bool(words & _IDEAS_TRIGGERS)

    if has_write and not has_ideas:
        return "write"
    if has_ideas and not has_write:
        return "ideas"
    return None  # ambigu → LLM


def run_orchestrator(state: StoryState) -> StoryState:
    """
    Agent orchestrateur : détermine l'intention de l'utilisateur.

    Sortie : routing_decision = "write" | "ideas"
    """
    print("\n🎯 [AGENT ORCHESTRATEUR] Analyse de l'intention...")

    user_request = state.get("user_request", "").strip()
    if not user_request:
        print("   ⚠️  Aucune demande détectée → chemin 'ideas' par défaut")
        return {
            **state,
            "routing_decision": "ideas",
            "orchestrator_reasoning": "Aucune demande fournie, analyse générale par défaut."
        }

    # ── Tentative de détection rapide ────────────────────────────────────
    fast_result = _fast_detect(user_request)
    if fast_result:
        label = "✍️  ÉCRITURE" if fast_result == "write" else "💡 IDÉES"
        print(f"   ✅ Détection rapide : {label}")
        return {
            **state,
            "routing_decision": fast_result,
            "orchestrator_reasoning": f"Détection par mots-clés : {fast_result}"
        }

    # ── Appel LLM pour les cas ambigus ───────────────────────────────────
    print("   📡 Demande ambiguë → analyse LLM...")
    try:
        llm = ChatOllama(
            model=state["model_name"],
            temperature=0.0,
            format="json"
        )

        story_preview = state.get("existing_story", "")[:200].strip()

        user_message = ORCHESTRATOR_USER_TEMPLATE.format(
            user_request=user_request,
            story_preview=story_preview
        )

        response = llm.invoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        # Parse JSON
        raw = response.content
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            data  = json.loads(raw[start:end]) if start != -1 else {}
        except json.JSONDecodeError:
            data = {}

        decision   = data.get("routing_decision", "ideas")
        reasoning  = data.get("raisonnement", "")
        intention  = data.get("intention_detectee", "")
        confiance  = data.get("confiance", 0.5)

        # Sécurité : force "ideas" si la valeur est inconnue
        if decision not in ("write", "ideas"):
            decision = "ideas"

        label = "✍️  ÉCRITURE" if decision == "write" else "💡 IDÉES"
        print(f"   ✅ Décision LLM : {label} (confiance {confiance:.0%})")
        print(f"   📋 Intention : {intention}")

        return {
            **state,
            "routing_decision": decision,
            "orchestrator_reasoning": reasoning
        }

    except Exception as e:
        erreur = f"ORCHESTRATEUR: Erreur LLM : {str(e)} → fallback 'ideas'"
        print(f"   ❌ {erreur}")
        return {
            **state,
            "routing_decision": "ideas",
            "orchestrator_reasoning": erreur,
            "errors": state.get("errors", []) + [erreur]
        }

