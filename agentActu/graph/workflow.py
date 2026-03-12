# graph/workflow.py
# Construction et compilation du graph LangGraph

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.search_agent import search_agent
from agents.summary_agent import summary_agent
from agents.validation_agent import validation_agent
from config.settings import logger


def router_apres_recherche(
        state: AgentState
) -> Literal["summary_agent", "end"]:
    """
    Fonction de routage après le SearchAgent.

    Décide si le workflow continue vers le SummaryAgent
    ou s'arrête en cas d'erreur critique.

    Args:
        state: L'état actuel après l'exécution du SearchAgent

    Returns:
        Nom du prochain nœud à exécuter
    """
    current_step = state.get("current_step", "")
    raw_results = state.get("raw_results", [])

    # Si erreur critique ou aucun résultat : arrêt du workflow
    if current_step == "error" or not raw_results:
        logger.warning(
            "⚠️ [Router] Arrêt après SearchAgent - "
            f"step={current_step}, résultats={len(raw_results)}"
        )
        return "end"

    logger.info("➡️ [Router] SearchAgent → SummaryAgent")
    return "summary_agent"


def router_apres_resume(
        state: AgentState
) -> Literal["validation_agent", "end"]:
    """
    Fonction de routage après le SummaryAgent.

    Vérifie que le résumé a bien été généré avant de
    lancer la validation.

    Args:
        state: L'état actuel après l'exécution du SummaryAgent

    Returns:
        Nom du prochain nœud à exécuter
    """
    current_step = state.get("current_step", "")
    summary = state.get("summary", "")

    # Si erreur ou résumé vide : arrêt du workflow
    if current_step == "error" or not summary:
        logger.warning(
            "⚠️ [Router] Arrêt après SummaryAgent - "
            f"step={current_step}, résumé vide={not summary}"
        )
        return "end"

    logger.info("➡️ [Router] SummaryAgent → ValidationAgent")
    return "validation_agent"


def construire_workflow() -> StateGraph:
    """
    Construit et compile le graph LangGraph du système de veille.

    Architecture du graph :
    START → search_agent → [router] → summary_agent → [router] → validation_agent → END

    Les routeurs permettent un arrêt propre en cas d'erreur
    à chaque étape critique.

    Returns:
        Le graph compilé prêt à être exécuté

    Example:
        >>> workflow = construire_workflow()
        >>> result = workflow.invoke({"query": "IA 2024", ...})
    """
    logger.info("🔨 Construction du workflow LangGraph...")

    # ----------------------------------------
    # ÉTAPE 1 : Créer le StateGraph
    # ----------------------------------------
    # StateGraph est le conteneur principal de LangGraph.
    # On lui passe notre TypedDict AgentState pour qu'il sache
    # comment gérer et valider l'état partagé.
    graph = StateGraph(AgentState)

    # ----------------------------------------
    # ÉTAPE 2 : Ajouter les nœuds (agents)
    # ----------------------------------------
    # Chaque nœud est une fonction Python qui reçoit le state
    # et retourne un dict avec les mises à jour du state.

    # Nœud 1 : SearchAgent (Chercheur)
    graph.add_node("search_agent", search_agent)
    logger.info("   ✅ Nœud 'search_agent' ajouté")

    # Nœud 2 : SummaryAgent (Condensateur)
    graph.add_node("summary_agent", summary_agent)
    logger.info("   ✅ Nœud 'summary_agent' ajouté")

    # Nœud 3 : ValidationAgent (Vérificateur)
    graph.add_node("validation_agent", validation_agent)
    logger.info("   ✅ Nœud 'validation_agent' ajouté")

    # ----------------------------------------
    # ÉTAPE 3 : Définir les arêtes (transitions)
    # ----------------------------------------

    # Arête d'entrée : START → search_agent
    # Le workflow commence TOUJOURS par le SearchAgent
    graph.add_edge(START, "search_agent")
    logger.info("   ✅ Arête START → search_agent ajoutée")

    # Arête conditionnelle : search_agent → [router] → suite
    # Selon le résultat du SearchAgent, on continue ou on s'arrête
    graph.add_conditional_edges(
        source="search_agent",
        path=router_apres_recherche,
        path_map={
            "summary_agent": "summary_agent",
            "end": END
        }
    )
    logger.info("   ✅ Arête conditionnelle search_agent → [router] ajoutée")

    # Arête conditionnelle : summary_agent → [router] → suite
    graph.add_conditional_edges(
        source="summary_agent",
        path=router_apres_resume,
        path_map={
            "validation_agent": "validation_agent",
            "end": END
        }
    )
    logger.info("   ✅ Arête conditionnelle summary_agent → [router] ajoutée")

    # Arête de sortie : validation_agent → END
    # Le workflow se termine TOUJOURS après le ValidationAgent
    graph.add_edge("validation_agent", END)
    logger.info("   ✅ Arête validation_agent → END ajoutée")

    # ----------------------------------------
    # ÉTAPE 4 : Compiler le graph
    # ----------------------------------------
    # La compilation valide la structure du graph,
    # vérifie qu'il n'y a pas de nœuds isolés,
    # et optimise les transitions.
    workflow_compile = graph.compile()

    logger.info("🎉 Workflow LangGraph compilé avec succès !")
    logger.info(
        "   Flux : START → SearchAgent → SummaryAgent → ValidationAgent → END"
    )

    return workflow_compile


def visualiser_workflow(workflow) -> None:
    """
    Affiche une représentation ASCII du workflow dans le terminal.

    Tente d'utiliser la visualisation Mermaid de LangGraph si disponible,
    sinon affiche une représentation ASCII simplifiée.

    Args:
        workflow: Le workflow compilé LangGraph
    """
    print("\n" + "="*60)
    print("📊 STRUCTURE DU WORKFLOW LANGGRAPH")
    print("="*60)

    try:
        # Tenter d'obtenir la représentation Mermaid
        mermaid = workflow.get_graph().draw_mermaid()
        print("\n🔗 Diagramme Mermaid (collez sur mermaid.live) :")
        print(mermaid)
    except Exception:
        pass

    # Représentation ASCII de secours (toujours affichée)
    print("""
┌─────────────────────────────────────────┐
│           FLUX DU WORKFLOW              │
│                                         │
│  ┌─────────┐                            │
│  │  START  │                            │
│  └────┬────┘                            │
│       │                                 │
│       ▼                                 │
│  ┌────────────┐   ❌ Erreur             │
│  │SearchAgent │──────────────► END      │
│  └────┬───────┘                         │
│       │ ✅ OK                            │
│       ▼                                 │
│  ┌─────────────┐  ❌ Erreur             │
│  │SummaryAgent │─────────────► END      │
│  └──────┬──────┘                        │
│         │ ✅ OK                          │
│         ▼                               │
│  ┌────────────────┐                     │
│  │ValidationAgent │                     │
│  └───────┬────────┘                     │
│          │                              │
│          ▼                              │
│        END                              │
└─────────────────────────────────────────┘
""")
    print("="*60 + "\n")
