# agents/synthesizer.py
# Agent de synthèse : produit la réponse finale formatée pour l'auteur

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from state import StoryState
from prompts.synthesizer_prompt import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE


def run_synthesizer(state: StoryState) -> StoryState:
    """
    Agent 4 : Synthétise toutes les analyses et produit la réponse finale.

    Entrée : tous les outputs des agents précédents
    Sortie : final_response (texte formaté pour l'auteur)
    """
    print("\n📝 [AGENT SYNTHÈSE] Production de la réponse finale...")

    try:
        llm = ChatOllama(
            model=state["model_name"],
            temperature=0.3,    # Équilibre entre précision et fluidité
        )

        user_message = SYNTHESIZER_USER_TEMPLATE.format(
            story_context=state.get("story_context", "Non disponible"),
            characters_json=json.dumps(state.get("characters_summary", {}),
                                       ensure_ascii=False, indent=2),
            plots_json=json.dumps(state.get("plots_summary", {}),
                                  ensure_ascii=False, indent=2),
            consistency_json=json.dumps(state.get("consistency_report", {}),
                                        ensure_ascii=False, indent=2),
            ideas_json=json.dumps(state.get("story_ideas", []),
                                  ensure_ascii=False, indent=2),
            user_request=state.get("user_request", "Analyse générale")
        )

        print("   📡 Synthèse en cours...")
        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        final_text = response.content

        # Ajout des erreurs accumulées si nécessaire
        errors = state.get("errors", [])
        if errors:
            final_text += "\n\n═══════════════════════════════════════\n"
            final_text += "⚠️  ERREURS SYSTÈME DÉTECTÉES\n"
            final_text += "═══════════════════════════════════════\n"
            for err in errors:
                final_text += f"  • {err}\n"

        if state.get("written_continuation"):
            wc = state["written_continuation"]
            sections.append(SUITE_ECRITE_SECTION.format(
                suite_ecrite  = wc.get("suite_ecrite", ""),
                point_de_fin  = wc.get("point_de_fin", ""),
            ))
        print("   ✅ Réponse finale générée")
        print(f"   ✅ Longueur : {len(final_text)} caractères")

        return {
            **state,
            "final_response": final_text
        }

    except Exception as e:
        # En cas d'échec du synthétiseur, on produit une réponse de secours
        erreur = f"SYNTHÉTISEUR: Erreur : {str(e)}"
        print(f"   ❌ {erreur}")

        fallback = _generate_fallback_response(state)
        return {**state, "final_response": fallback,
                "errors": state.get("errors", []) + [erreur]}


def _generate_fallback_response(state: StoryState) -> str:
    """
    Génère une réponse de secours formatée manuellement
    si l'agent synthèse échoue.
    """
    lines = ["═══════════════════════════════════════"]
    lines.append("📖 SITUATION ACTUELLE")
    lines.append("═══════════════════════════════════════")
    lines.append(state.get("story_context", "Non disponible"))

    lines.append("\n═══════════════════════════════════════")
    lines.append("👥 PERSONNAGES IDENTIFIÉS")
    lines.append("═══════════════════════════════════════")
    for nom, info in state.get("characters_summary", {}).items():
        lines.append(f"\n{nom} - {info.get('role', 'rôle inconnu')}")
        lines.append(f"  • Statut : {info.get('statut_actuel', 'inconnu')}")

    lines.append("\n═══════════════════════════════════════")
    lines.append("💡 IDÉES GÉNÉRÉES")
    lines.append("═══════════════════════════════════════")
    for idee in state.get("story_ideas", []):
        lines.append(f"\n[{idee.get('score', '?')}/10] {idee.get('titre', 'Sans titre')}")
        lines.append(f"  {idee.get('description', '')}")

    return "\n".join(lines)
