# src/graphs/advanced_graph.py
"""
Graphe avancé : Researcher → Writer → Critic → (Writer | END)
Inclut une boucle de révision conditionnelle.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from src.agents.researcher import ResearcherAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from src.models import WorkflowState
from src.utils.logger import (
    setup_logging,
    log_transition,
    log_workflow_complete
)

console = Console()


# ── Fonctions de routage (conditional edges) ─────────────────────────

def route_after_critic(state: WorkflowState) -> str:
    """
    Fonction de routage après le Critic.

    LangGraph appelle cette fonction pour décider
    vers quel nœud transitionner.

    Returns:
        "writer" : si révision nécessaire
        "end"    : si contenu approuvé
    """

    # Sécurité : erreur lors du critique
    if state.error_message:
        console.print(f"[red]Erreur détectée, fin du workflow: {state.error_message}[/red]")
        return "end"

    # Nombre max de révisions atteint
    if state.revision_count >= state.max_revisions:
        console.print(f"[yellow]⚠️ Nombre max de révisions atteint ({state.max_revisions})[/yellow]")
        log_transition("critic", "END", f"Max révisions ({state.max_revisions}) atteint")
        return "end"

    # Vérifier si révision nécessaire
    if state.review and state.review.needs_revision:
        score = state.review.overall_score
        log_transition(
            "critic", "writer",
            f"Score insuffisant: {score}/10 - Révision #{state.revision_count}"
        )
        return "writer"

    # Contenu approuvé
    score = state.review.overall_score if state.review else "N/A"
    log_transition("critic", "END", f"Contenu approuvé - Score: {score}/10")
    return "end"


def route_after_writer(state: WorkflowState) -> str:
    """
    Routage après le Writer.
    Normalement toujours vers critic, sauf erreur.
    """
    if state.error_message:
        return "end"
    return "critic"


# ── Nœuds de contrôle ────────────────────────────────────────────────

def finalize_output(state: WorkflowState) -> WorkflowState:
    """
    Nœud final : finalise le contenu et affiche les statistiques.

    Ce nœud s'exécute avant END pour nettoyer l'état.
    """
    # S'assurer que final_content est défini
    if not state.final_content and state.draft_content:
        state.final_content = state.draft_content

    # Afficher les statistiques finales
    console.print("\n[bold cyan]📊 Statistiques du Workflow:[/bold cyan]")
    console.print(f"  • Révisions effectuées : {state.revision_count}")
    console.print(f"  • Agents exécutés : {len(state.agent_outputs)}")

    if state.review:
        console.print(f"  • Score final : {state.review.overall_score}/10")
        console.print(f"  • Précision : {state.review.accuracy_score}/10")
        console.print(f"  • Clarté : {state.review.clarity_score}/10")
        console.print(f"  • Complétude : {state.review.completeness_score}/10")

    state.is_complete = True
    return state


# ── Création du graphe ───────────────────────────────────────────────

def create_advanced_graph():
    """
    Crée le graphe avancé avec boucle de révision.

    Topologie :
    ┌─────────────┐
    │    START    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Researcher │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐ ◄──────────────┐
    │    Writer   │                │
    └──────┬──────┘                │
           │                       │ (si révision)
           ▼                       │
    ┌─────────────┐                │
    │    Critic   │ ───────────────┘
    └──────┬──────┘
           │ (si approuvé)
           ▼
    ┌─────────────┐
    │  Finalize   │
    └──────┬──────┘
           │
           ▼
          END
    """
    # Initialiser les agents
    researcher = ResearcherAgent()
    writer = WriterAgent()
    critic = CriticAgent()

    # Créer le graphe
    graph = StateGraph(WorkflowState)

    # ── Nœuds ────────────────────────────────────────────
    graph.add_node("researcher", researcher.run)
    graph.add_node("writer", writer.run)
    graph.add_node("critic", critic.run)
    graph.add_node("finalize", finalize_output)

    # ── Transitions ───────────────────────────────────────
    # Point d'entrée
    graph.set_entry_point("researcher")

    # Researcher → Writer (toujours)
    graph.add_edge("researcher", "writer")

    # Writer → Critic ou END (conditionnel)
    graph.add_conditional_edges(
        "writer",
        route_after_writer,
        {
            "critic": "critic",
            "end": "finalize"
        }
    )

    # Critic → Writer (révision) ou Finalize (approuvé)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "writer": "writer",
            "end": "finalize"
        }
    )

    # Finalize → END
    graph.add_edge("finalize", END)

    # ── Compiler ──────────────────────────────────────────
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_advanced_workflow(
        topic: str,
        instructions: str = "",
        max_revisions: int = 3,
        thread_id: str = "advanced-1"
) -> WorkflowState:
    """
    Exécute le workflow avancé complet.

    Args:
        topic: Le sujet à traiter
        instructions: Instructions supplémentaires
        max_revisions: Nombre max de révisions (défaut: 3)
        thread_id: ID de session unique (pour la mémoire)

    Returns:
        L'état final du workflow
    """
    setup_logging()

    console.print(f"\n[bold green]🚀 Démarrage du Workflow Multi-Agent[/bold green]")
    console.print(f"[cyan]Sujet:[/cyan] {topic}")
    console.print(f"[cyan]Max révisions:[/cyan] {max_revisions}\n")

    graph = create_advanced_graph()

    # État initial
    initial_state = WorkflowState(
        topic=topic,
        instructions=instructions,
        max_revisions=max_revisions
    )

    # Config pour la persistance de session
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,  # Protection contre les boucles infinies
    }

    try:
        # Exécuter le workflow
        final_state_dict = graph.invoke(initial_state, config=config)
        final_state = WorkflowState(**final_state_dict)

        # Afficher le résultat final
        if final_state.final_content:
            log_workflow_complete(
                final_state.final_content,
                len(final_state.agent_outputs)
            )

        return final_state

    except Exception as e:
        console.print(f"[bold red]❌ Erreur fatale du workflow: {e}[/bold red]")
        raise
