# graph.py
# Définition du graphe LangGraph orchestrant tous les agents
#
# Workflow complet :
#
#   orchestrator
#        │
#        ├─► [write]  analyst → writer → validator → synthesizer → END
#        │
#        └─► [ideas]  analyst ──┬──► checker → ideator ──► synthesizer → END
#                               │
#                               └──► synthesizer (si erreur critique)

from langgraph.graph import StateGraph, END
from state import StoryState
from agents import (
    run_analyst, run_checker, run_ideator,
    run_synthesizer, run_writer, run_orchestrator, run_validator
)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de routage conditionnel
# ─────────────────────────────────────────────────────────────────────────────

def route_after_orchestrator(state: StoryState) -> str:
    """
    Après l'orchestrateur : toujours passer par l'analyste.
    L'analyste est nécessaire dans les deux chemins pour le contexte.
    """
    return "analyst"


def route_after_analyst(state: StoryState) -> str:
    """
    Après l'analyste : erreur critique → synthesizer (mode dégradé)
    Sinon, selon la décision de l'orchestrateur.
    """
    errors = state.get("errors", [])
    critical = [e for e in errors if "ANALYSTE" in e]

    if critical and not state.get("characters_summary"):
        print("   ⚠️  Erreur critique analyste → mode dégradé (synthesizer)")
        return "synthesizer"

    decision = state.get("routing_decision", "ideas")
    if decision == "write":
        print("   🔀 Routage → chemin ÉCRITURE (writer)")
        return "writer"
    else:
        print("   🔀 Routage → chemin IDÉES (checker)")
        return "checker"


def route_after_validator(state: StoryState) -> str:
    """
    Après le validateur : toujours vers le synthesizer.
    La boucle de correction utilisateur est gérée dans main.py,
    pas dans le graphe (évite la complexité des checkpoints).
    """
    return "synthesizer"


# ─────────────────────────────────────────────────────────────────────────────
# Construction du graphe
# ─────────────────────────────────────────────────────────────────────────────

def build_story_graph() -> StateGraph:
    """
    Construit et compile le graphe LangGraph.

    Chemin "write"  : orchestrator → analyst → writer → validator → synthesizer → END
    Chemin "ideas"  : orchestrator → analyst → checker → ideator → synthesizer → END
    Mode dégradé    : orchestrator → analyst → synthesizer → END
    """
    graph = StateGraph(StoryState)

    # ── Nœuds ────────────────────────────────────────────────────────────
    graph.add_node("orchestrator", run_orchestrator)
    graph.add_node("analyst",      run_analyst)
    graph.add_node("checker",      run_checker)
    graph.add_node("ideator",      run_ideator)
    graph.add_node("writer",       run_writer)
    graph.add_node("validator",    run_validator)
    graph.add_node("synthesizer",  run_synthesizer)

    # ── Point d'entrée : orchestrateur ───────────────────────────────────
    graph.set_entry_point("orchestrator")

    # orchestrator → analyst (toujours)
    graph.add_edge("orchestrator", "analyst")


    # analyst → writer | checker | synthesizer (conditionnel)
    graph.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "writer":      "writer",
            "checker":     "checker",
            "synthesizer": "synthesizer"
        }
    )

    # ── Chemin ÉCRITURE ───────────────────────────────────────────────────
    # writer → validator → synthesizer
    graph.add_edge("writer",    "validator")
    graph.add_edge("validator", "synthesizer")

    # ── Chemin IDÉES ─────────────────────────────────────────────────────
    # checker → ideator → synthesizer
    graph.add_edge("checker",  "ideator")
    graph.add_edge("ideator",  "synthesizer")

    # ── Fin ───────────────────────────────────────────────────────────────
    graph.add_edge("synthesizer", END)

    compiled = graph.compile()

    print("✅ Graphe LangGraph compilé avec succès")
    print("   Flux write : orchestrator → analyst → writer → validator → synthesizer → END")
    print("   Flux ideas : orchestrator → analyst → checker → ideator → synthesizer → END")

    return compiled


# Instance globale du graphe (singleton)
story_graph = build_story_graph()
