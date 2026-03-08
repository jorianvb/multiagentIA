"""
Graphe simple : Researcher → Writer → END
Démonstration des concepts de base de LangGraph.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.researcher import ResearcherAgent
from src.agents.writer import WriterAgent
from src.models import WorkflowState
from src.utils.logger import setup_logging, log_transition


def create_simple_graph():
    """
    Crée un graphe simple à 2 agents.

    Flux : START → researcher → writer → END
    """
    # Initialiser les agents
    researcher = ResearcherAgent()
    writer = WriterAgent()

    # Créer le graphe avec notre état typé
    graph = StateGraph(WorkflowState)

    # ── Ajouter les nœuds ──────────────────────────────────
    # Chaque nœud est une fonction qui prend et retourne l'état
    graph.add_node("researcher", researcher.run)
    graph.add_node("writer", writer.run)

    # ── Définir les transitions ────────────────────────────
    # Point d'entrée
    graph.set_entry_point("researcher")

    # Transitions fixes (pas de conditions)
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)

    # ── Compiler le graphe ─────────────────────────────────
    # MemorySaver permet de sauvegarder l'état entre les nœuds
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    return compiled_graph


def run_simple_workflow(topic: str, instructions: str = "") -> WorkflowState:
    """Exécute le workflow simple."""
    setup_logging()

    graph = create_simple_graph()

    # État initial
    initial_state = WorkflowState(
        topic=topic,
        instructions=instructions
    )

    # Configuration de la session (pour la mémoire)
    config = {"configurable": {"thread_id": "simple-workflow-1"}}

    # Exécuter le graphe
    final_state = graph.invoke(initial_state, config=config)

    return WorkflowState(**final_state)
