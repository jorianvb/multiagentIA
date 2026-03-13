# utils/scoring.py
# Stratégie de scoring des idées de suite

from typing import List, Dict, Any


# Critères et leurs poids dans le score final
SCORING_WEIGHTS = {
    "coherence": 0.30,          # Cohérence avec l'histoire existante
    "potentiel_dramatique": 0.30,  # Intérêt narratif et tension
    "respect_personnages": 0.20,   # Fidélité aux arcs des personnages
    "originalite": 0.20            # Fraîcheur et originalité
}

# Bonus/malus automatiques
BONUS_RESOLVES_PLOT_HOLE = 0.5      # Résout un plot hole identifié
BONUS_DEVELOPPE_SECONDAIRE = 0.3   # Développe un personnage secondaire
MALUS_CONTRADICTION = -1.0          # Crée une contradiction
MALUS_ARC_ABANDONNE = -0.5         # Abandonne un arc en cours


def calculate_idea_score(
        idea: Dict[str, Any],
        consistency_report: Dict[str, Any],
        plots: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calcule le score final d'une idée en tenant compte du contexte.

    Formule :
    score_final = Σ(critère × poids) + bonus - malus

    Score max théorique : 10 + bonus potentiels
    Score normalisé : min(10, score_final)
    """
    # Score de base depuis le LLM (0-10)
    detail = idea.get("detail_score", {})
    base_score = (
                         detail.get("coherence", 5) * SCORING_WEIGHTS["coherence"] +
                         detail.get("potentiel_dramatique", 5) * SCORING_WEIGHTS["potentiel_dramatique"] +
                         detail.get("respect_personnages", 5) * SCORING_WEIGHTS["respect_personnages"] +
                         detail.get("originalite", 5) * SCORING_WEIGHTS["originalite"]
                 ) * (10 / 1)  # Normalisation

    # Calcul des bonus/malus contextuels
    bonus = 0.0
    malus = 0.0
    bonus_details = []
    malus_details = []

    # Bonus : résolution de plot holes
    plot_holes = consistency_report.get("plot_holes", [])
    description = idea.get("description", "").lower()
    for hole in plot_holes:
        if any(word in description for word in hole.get("description", "").lower().split()):
            bonus += BONUS_RESOLVES_PLOT_HOLE
            bonus_details.append(f"+{BONUS_RESOLVES_PLOT_HOLE} résolution plot hole")
            break

    # Bonus : développement personnages secondaires
    impact_perso = idea.get("impact_personnages", {})
    if len(impact_perso) > 2:  # Impact sur plus de 2 personnages
        bonus += BONUS_DEVELOPPE_SECONDAIRE
        bonus_details.append(f"+{BONUS_DEVELOPPE_SECONDAIRE} développement multi-personnages")

    # Malus : arcs abandonnés
    arcs_abandonnes = consistency_report.get("arcs_abandonnes", [])
    for arc in arcs_abandonnes:
        if arc.lower() in description:
            malus += abs(MALUS_ARC_ABANDONNE)
            malus_details.append(f"{MALUS_ARC_ABANDONNE} arc potentiellement abandonné")

    # Score final normalisé
    final_score = min(10.0, max(0.0, base_score + bonus - malus))

    return {
        **idea,
        "score": round(final_score, 2),
        "score_detail": {
            "base": round(base_score, 2),
            "bonus": round(bonus, 2),
            "malus": round(malus, 2),
            "bonus_details": bonus_details,
            "malus_details": malus_details
        }
    }


def rank_ideas(
        ideas: List[Dict],
        consistency_report: Dict,
        plots: Dict
) -> List[Dict]:
    """Recalcule et re-trie les idées par score décroissant."""
    scored = [calculate_idea_score(idea, consistency_report, plots)
              for idea in ideas]
    return sorted(scored, key=lambda x: x["score"], reverse=True)
