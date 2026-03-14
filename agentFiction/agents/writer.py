import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.writer_prompt import WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE


logger = logging.getLogger(__name__)
# Mots-clés qui déclenchent l'agent writer
WRITE_TRIGGERS = [
    "écris", "ecris", "rédige", "redige", "continue",
    "développe", "developpe", "raconte", "montre",
    "suite", "passage", "scène", "scene",
]


def should_write(user_request: str) -> bool:
    """Détermine si l'utilisateur veut qu'on écrive la suite."""
    request_lower = user_request.lower()
    return any(trigger in request_lower for trigger in WRITE_TRIGGERS)


def _parse_writer_response(raw: str) -> dict:
    """Parse la réponse JSON de l'agent writer."""
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
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
    """Agent qui écrit la suite de l'histoire si l'utilisateur le demande."""

    user_request = state.get("user_request", "").strip()

    # ── Guard : on n'écrit que si demandé ────────────────────────────────
    if not should_write(user_request):
        logger.info("Writer: pas de demande d'écriture détectée, passage ignoré.")
        return {**state, "written_continuation": None}

    existing_story = state.get("existing_story", "").strip()
    if not existing_story:
        return {
            **state,
            "written_continuation": None,
            "errors": state.get("errors", []) + ["Writer: histoire vide, impossible d'écrire la suite."],
        }

    # ── Préparation du contexte ───────────────────────────────────────────
    user_message = WRITER_USER_TEMPLATE.format(
        existing_story    = existing_story,
        characters_json   = json.dumps(state.get("characters_summary", {}), ensure_ascii=False, indent=2),
        plots_json        = json.dumps(state.get("plots_summary",      {}), ensure_ascii=False, indent=2),
        consistency_json  = json.dumps(state.get("consistency_report", {}), ensure_ascii=False, indent=2),
        story_ideas_json  = json.dumps(state.get("story_ideas",        []), ensure_ascii=False, indent=2),
        user_request      = user_request,
    )

    # ── Appel LLM ────────────────────────────────────────────────────────
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.8)
        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        parsed   = _parse_writer_response(response.content)

        logger.info(f"Writer: suite écrite ({len(parsed.get('suite_ecrite',''))} chars)")

        return {
            **state,
            "written_continuation": parsed,
            "errors": state.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Writer error: {e}")
        return {
            **state,
            "written_continuation": None,
            "errors": state.get("errors", []) + [f"Writer: {str(e)}"],
        }
