# graph.py
# Définition du graphe LangGraph orchestrant tous les agents

from langgraph.graph import StateGraph, END
from state import StoryState
from agents import run_analyst, run_checker, run_ideator, run_synthesizer


def should_continue_after_analyst(state: StoryState) -> str:
    """
    Condition de routage après l'agent analyste.
    Si erreur critique → fin, sinon → checker
    """
    errors = state.get("errors", [])
    critical_errors = [e for e in errors if "ANALYSTE" in e]

    if critical_errors and not state.get("characters_summary"):
        print("   ⚠️  Erreur critique détectée, passage en mode dégradé")
        return "synthesizer"  # Saute au synthétiseur avec les données disponibles
    return "checker"


def build_story_graph() -> StateGraph:
    """
    Construit et compile le graphe LangGraph.

    Flux : analyst → checker → ideator → synthesizer → END

    En cas d'erreur critique de l'analyste,
    on saute directement au synthétiseur.
    """
    # Initialisation du graphe avec notre état
    graph = StateGraph(StoryState)

    # Ajout des nœuds (chaque agent est un nœud)
    graph.add_node("analyst", run_analyst)
    graph.add_node("checker", run_checker)
    graph.add_node("ideator", run_ideator)
    graph.add_node("synthesizer", run_synthesizer)

    # Point d'entrée
    graph.set_entry_point("analyst")

    # Edges conditionnels depuis l'analyste
    graph.add_conditional_edges(
        "analyst",
        should_continue_after_analyst,
        {
            "checker": "checker",
            "synthesizer": "synthesizer"
        }
    )

    # Edges fixes pour la suite du pipeline
    graph.add_edge("checker", "ideator")
    graph.add_edge("ideator", "synthesizer")
    graph.add_edge("synthesizer", END)

    # Compilation du graphe
    compiled = graph.compile()

    print("✅ Graphe LangGraph compilé avec succès")
    print("   Flux : analyst → checker → ideator → synthesizer → END")

    return compiled


# Instance globale du graphe (singleton)
story_graph = build_story_graph()
