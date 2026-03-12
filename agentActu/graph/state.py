# graph/state.py
# Définition de l'état partagé entre tous les agents LangGraph

from typing import TypedDict, List, Dict, Any, Optional


class ArticleResult(TypedDict):
    """
    Structure d'un article collecté par le SearchAgent.

    Chaque article doit contenir au minimum un titre et une URL.
    Les autres champs peuvent être vides si non disponibles.
    """
    titre: str
    url: str
    source: str
    date: str
    contenu: str
    score_pertinence: float


class ValidationResult(TypedDict):
    """
    Structure du résultat de validation produit par le ValidationAgent.

    Contient le score de fiabilité, la décision finale et les détails
    de l'analyse de cohérence.
    """
    score_fiabilite: int
    decision: str
    scores_detail: Dict[str, int]
    points_forts: List[str]
    points_douteux: List[str]
    contradictions: List[str]
    justification: str
    recommandations: List[str]


class AgentState(TypedDict):
    """
    État global partagé entre tous les agents du workflow.

    C'est le "tableau blanc" sur lequel chaque agent lit
    et écrit ses données. LangGraph gère automatiquement
    la transmission de cet état entre les nœuds.

    Analogie : C'est comme un dossier de travail partagé
    que chaque agent enrichit à son tour.

    Champs :
        query           : La requête de recherche initiale
        raw_results     : Articles bruts collectés par SearchAgent
        search_metadata : Métadonnées de la recherche
        summary         : Résumé Markdown généré par SummaryAgent
        summary_metadata: Métadonnées du résumé
        validation_result: Résultat de validation du ValidationAgent
        final_report    : Rapport final combinant résumé et validation
        current_step    : Étape actuelle du workflow
        errors          : Liste des erreurs rencontrées
        timestamps      : Horodatages de chaque étape
    """
    query: str
    raw_results: List[ArticleResult]
    search_metadata: Dict[str, Any]
    summary: str
    summary_metadata: Dict[str, Any]
    validation_result: Optional[ValidationResult]
    final_report: str
    current_step: str
    errors: List[str]
    timestamps: Dict[str, str]
