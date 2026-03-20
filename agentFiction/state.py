# state.py
# Définition de l'état partagé entre tous les agents du graphe LangGraph
# Cet état est la "mémoire de travail" du système pour une session d'analyse

from typing import TypedDict, Optional, List, Dict, Any


class PersonnageInfo(TypedDict):
    """Structure d'un personnage extrait de l'histoire"""
    nom: str
    role: str                    # protagoniste, antagoniste, secondaire, etc.
    traits: List[str]            # traits de caractère
    motivations: List[str]       # ce que veut le personnage
    statut_actuel: str           # où en est le personnage dans l'histoire
    relations: Dict[str, str]    # {nom_personnage: type_relation}
    arcs: List[str]              # arcs narratifs en cours
    incertain: bool              # True si certaines infos sont supposées


class IntrigueInfo(TypedDict):
    """Structure d'une intrigue extraite de l'histoire"""
    titre: str
    type: str                    # principale, secondaire, romantique, etc.
    description: str
    statut: str                  # en cours, résolue, abandonnée
    personnages_impliques: List[str]
    fils_non_resolus: List[str]
    incertaine: bool


class IdeeSuite(TypedDict):
    """Structure d'une idée de suite générée"""
    titre: str
    type: str                    # dramatique, twist, légère, développement
    description: str
    avantages: List[str]
    risques: List[str]
    impact_personnages: Dict[str, str]
    impact_intrigues: Dict[str, str]
    score: float                 # score de pertinence 0-10
    justification_score: str


class StoryState(TypedDict):
    """
    État global partagé entre tous les agents.
    Chaque agent lit et enrichit cet état.
    """
    # === INPUTS ===
    existing_story: str          # Le texte déjà écrit par l'auteur (SOURCE DE VÉRITÉ)
    user_request: str            # La demande spécifique de l'auteur
    model_name: str              # Modèle Ollama à utiliser

    # === ORCHESTRATEUR ===
    routing_decision: str                            # "write" | "ideas"
    orchestrator_reasoning: str                      # Raisonnement de l'orchestrateur

    # === OUTPUTS DES AGENTS ===
    characters_summary: Dict[str, PersonnageInfo]   # Agent Analyste
    plots_summary: Dict[str, IntrigueInfo]           # Agent Analyste
    story_context: str                               # Résumé de situation actuelle
    consistency_report: Dict[str, Any]               # Agent Cohérence
    story_ideas: List[IdeeSuite]                     # Agent Créatif
    written_continuation: Optional[Dict[str, Any]]  # Agent Writer
    validation_report: Dict[str, Any]               # Agent Validateur
    final_response: str                              # Agent Synthèse

    # === BOUCLE DE CORRECTION ===
    writer_correction: str       # Correction demandée par l'utilisateur (si relance)
    writer_iteration: int        # Nombre d'itérations du writer

    # === MÉTADONNÉES ===
    iteration_count: int
    session_id: str
    timestamp: str
    errors: List[str]            # Erreurs non bloquantes accumulées

