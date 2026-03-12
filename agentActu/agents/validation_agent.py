# agents/validation_agent.py
# Agent 3 : Validation, cohérence et scoring de fiabilité

import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState, ValidationResult
from config.settings import ollama_config, logger


# Prompt système du ValidationAgent
VALIDATION_AGENT_SYSTEM_PROMPT = """Tu es un expert en fact-checking, vérification d'information 
et évaluation de la fiabilité des contenus journalistiques.

Ton rôle est d'analyser un résumé d'actualités et d'évaluer sa qualité et sa cohérence.

CRITÈRES D'ÉVALUATION :

1. COHÉRENCE LOGIQUE (25 points)
   - Les informations sont-elles logiquement cohérentes entre elles ?
   - Y a-t-il des contradictions internes ?
   - Le raisonnement est-il solide ?

2. QUALITÉ DES SOURCES (25 points)
   - Les sources citées sont-elles reconnues et fiables ?
   - Y a-t-il une diversité de sources ?
   - Les sources sont-elles récentes ?

3. OBJECTIVITÉ (25 points)
   - Le contenu est-il neutre et factuel ?
   - Y a-t-il des biais évidents ?
   - Les affirmations sont-elles étayées ?

4. COMPLÉTUDE (25 points)
   - Le sujet est-il traité de façon complète ?
   - Les points importants sont-ils couverts ?
   - Y a-t-il des lacunes évidentes ?

SCORE FINAL = Somme des 4 critères (0-100)

Interprétation :
- 80-100 : VALIDÉ ✅ (contenu fiable)
- 60-79  : VALIDÉ AVEC RÉSERVES ⚠️ (quelques points à vérifier)
- 40-59  : REJETÉ ❌ (trop d'incohérences)
- 0-39   : REJETÉ ❌ (contenu non fiable)

Tu dois répondre UNIQUEMENT en JSON valide avec cette structure exacte :
{
    "score_fiabilite": <nombre entier 0-100>,
    "decision": "<VALIDÉ|VALIDÉ AVEC RÉSERVES|REJETÉ>",
    "scores_detail": {
        "coherence_logique": <0-25>,
        "qualite_sources": <0-25>,
        "objectivite": <0-25>,
        "completude": <0-25>
    },
    "points_forts": [
        "<point fort 1>",
        "<point fort 2>"
    ],
    "points_douteux": [
        "<point douteux 1>"
    ],
    "contradictions": [
        "<contradiction détectée 1>"
    ],
    "justification": "<Explication détaillée de 3-5 phrases>",
    "recommandations": [
        "<recommandation 1>",
        "<recommandation 2>"
    ]
}

Réponds UNIQUEMENT avec du JSON valide."""


def validation_agent(state: AgentState) -> Dict[str, Any]:
    """
    Nœud LangGraph : Agent de validation et fact-checking.

    Analyse le résumé généré par le SummaryAgent pour détecter
    les incohérences, évaluer la fiabilité et produire un rapport
    de validation avec un score.

    Args:
        state: L'état actuel contenant le summary du SummaryAgent

    Returns:
        Dictionnaire avec les champs mis à jour :
        - validation_result: Résultat complet de la validation
        - final_report: Rapport final combinant résumé et validation
        - current_step: 'done' si succès, 'error' si échec critique
        - errors: Erreurs éventuelles
        - timestamps: Horodatages mis à jour
    """
    logger.info("🔍 [ValidationAgent] Démarrage de la validation...")

    # Récupérer les données depuis le state
    summary: str = state.get("summary", "")
    query: str = state.get("query", "")
    raw_results = state.get("raw_results", [])
    summary_metadata = state.get("summary_metadata", {})
    errors: List[str] = state.get("errors", [])
    timestamps: Dict = state.get("timestamps", {})

    # Enregistrer le début de l'étape
    timestamps["validation_start"] = datetime.now().isoformat()

    # Vérification des données d'entrée
    if not summary:
        erreur = "❌ Aucun résumé disponible pour la validation"
        logger.error(erreur)
        errors.append(erreur)
        return {
            "validation_result": _creer_validation_erreur(erreur),
            "final_report": f"# ❌ Erreur de validation\n\n{erreur}",
            "current_step": "error",
            "errors": errors,
            "timestamps": timestamps
        }

    logger.info(f"📋 Validation du résumé pour : '{query}'")

    # ----------------------------------------
    # PHASE 1 : Analyse via LLM
    # ----------------------------------------
    validation_result: Optional[ValidationResult] = None

    try:
        # Préparer le contexte de validation
        contexte_sources = _preparer_contexte_sources(raw_results)

        message_utilisateur = f"""Analyse et valide ce résumé d'actualités sur le sujet : "{query}"

INFORMATIONS SUR LA RECHERCHE :
- Nombre d'articles sources : {len(raw_results)}
- Sources utilisées : {', '.join(summary_metadata.get('sources_utilisees', [])[:10])}
- Thématiques identifiées : {summary_metadata.get('nombre_thematiques', 0)}

RÉSUMÉ À VALIDER :
{summary}

ÉCHANTILLON DES SOURCES BRUTES POUR VÉRIFICATION :
{contexte_sources}

Évalue maintenant ce résumé selon les critères définis et retourne ton analyse en JSON."""

        # Créer le LLM avec température très basse pour une évaluation rigoureuse
        llm = ollama_config.creer_llm(temperature=0.05)

        messages = [
            SystemMessage(content=VALIDATION_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=message_utilisateur)
        ]

        logger.info("🤖 Invocation du LLM pour la validation...")
        response = llm.invoke(messages)

        # Parser la réponse JSON
        validation_result = _parser_reponse_validation(response.content)

        if validation_result:
            logger.info(
                f"✅ Validation terminée - Score : {validation_result['score_fiabilite']}/100 "
                f"- Décision : {validation_result['decision']}"
            )
        else:
            raise ValueError("Impossible de parser la réponse de validation")

    except Exception as e:
        erreur = f"⚠️ Erreur validation LLM : {str(e)}"
        logger.warning(erreur)
        errors.append(erreur)

        # Créer une validation de fallback
        validation_result = _creer_validation_fallback(summary, raw_results)
        logger.info("⚠️ Validation de fallback appliquée")

    # ----------------------------------------
    # PHASE 2 : Génération du rapport final
    # ----------------------------------------
    final_report = _generer_rapport_final(
        query=query,
        summary=summary,
        validation_result=validation_result,
        summary_metadata=summary_metadata,
        timestamps=timestamps
    )

    # Enregistrer la fin de l'étape
    timestamps["validation_end"] = datetime.now().isoformat()
    timestamps["workflow_end"] = datetime.now().isoformat()

    logger.info("🏁 [ValidationAgent] Terminé - Rapport final généré")

    return {
        "validation_result": validation_result,
        "final_report": final_report,
        "current_step": "done",
        "errors": errors,
        "timestamps": timestamps
    }


def _parser_reponse_validation(contenu: str) -> Optional[ValidationResult]:
    """
    Parse la réponse JSON du LLM de validation.

    Gère les cas où le LLM ajoute du texte autour du JSON
    ou retourne un JSON malformé.

    Args:
        contenu: La réponse brute du LLM

    Returns:
        ValidationResult parsé ou None si impossible
    """
    try:
        contenu_nettoye = contenu.strip()

        # Chercher le bloc JSON dans la réponse
        debut_json = contenu_nettoye.find("{")
        fin_json = contenu_nettoye.rfind("}") + 1

        if debut_json < 0 or fin_json <= debut_json:
            logger.warning("⚠️ Pas de JSON trouvé dans la réponse de validation")
            return None

        json_str = contenu_nettoye[debut_json:fin_json]
        data = json.loads(json_str)

        # Valider et normaliser les champs obligatoires
        score = int(data.get("score_fiabilite", 50))
        score = max(0, min(100, score))  # Clamper entre 0 et 100

        decision = data.get("decision", "VALIDÉ AVEC RÉSERVES")

        # Normaliser la décision
        if score >= 80:
            decision = "VALIDÉ"
        elif score >= 60:
            decision = "VALIDÉ AVEC RÉSERVES"
        else:
            decision = "REJETÉ"

        validation: ValidationResult = {
            "score_fiabilite": score,
            "decision": decision,
            "points_forts": data.get("points_forts", []),
            "points_douteux": data.get("points_douteux", []),
            "contradictions": data.get("contradictions", []),
            "justification": data.get(
                "justification",
                "Validation automatique effectuée."
            ),
            "recommandations": data.get("recommandations", [])
        }

        return validation

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"⚠️ Erreur parsing validation JSON : {str(e)}")
        return None


def _creer_validation_fallback(
        summary: str,
        raw_results: list
) -> ValidationResult:
    """
    Crée une validation heuristique sans LLM.

    Utilise des règles simples pour évaluer la qualité du résumé
    quand le LLM n'est pas disponible.

    Args:
        summary: Le résumé à évaluer
        raw_results: Les articles sources

    Returns:
        ValidationResult basé sur des heuristiques simples
    """
    score = 50  # Score de base
    points_forts = []
    points_douteux = []

    # Heuristique 1 : Longueur du résumé
    if len(summary) > 500:
        score += 10
        points_forts.append("Résumé suffisamment détaillé")
    else:
        points_douteux.append("Résumé trop court")

    # Heuristique 2 : Présence de sources citées
    if "(Source:" in summary or "| " in summary:
        score += 10
        points_forts.append("Sources citées dans le résumé")
    else:
        points_douteux.append("Sources non citées explicitement")

    # Heuristique 3 : Nombre de sources brutes
    if len(raw_results) >= 5:
        score += 10
        points_forts.append(f"{len(raw_results)} sources collectées")
    elif len(raw_results) >= 3:
        score += 5
    else:
        points_douteux.append("Nombre de sources insuffisant")

    # Heuristique 4 : Structure Markdown
    if summary.count("##") >= 3:
        score += 5
        points_forts.append("Bonne structure thématique")

    # Clamper le score
    score = max(0, min(100, score))

    if score >= 80:
        decision = "VALIDÉ"
    elif score >= 60:
        decision = "VALIDÉ AVEC RÉSERVES"
    else:
        decision = "REJETÉ"

    return ValidationResult({
        "score_fiabilite": score,
        "decision": decision,
        "points_forts": points_forts,
        "points_douteux": points_douteux,
        "contradictions": [],
        "justification": (
            f"Validation heuristique (LLM indisponible). "
            f"Score calculé sur {len(raw_results)} sources. "
            f"Résumé de {len(summary)} caractères analysé."
        ),
        "recommandations": [
            "Relancer avec le LLM disponible pour une validation complète",
            "Vérifier manuellement les sources citées"
        ]
    })


def _creer_validation_erreur(message: str) -> ValidationResult:
    """
    Crée une ValidationResult d'erreur pour les cas bloquants.

    Args:
        message: Message d'erreur à inclure

    Returns:
        ValidationResult indiquant une erreur critique
    """
    return ValidationResult({
        "score_fiabilite": 0,
        "decision": "REJETÉ",
        "points_forts": [],
        "points_douteux": ["Données d'entrée manquantes"],
        "contradictions": [],
        "justification": f"Validation impossible : {message}",
        "recommandations": ["Vérifier que le SearchAgent et le SummaryAgent ont bien fonctionné"]
    })


def _preparer_contexte_sources(
        raw_results: list,
        max_articles: int = 4
) -> str:
    """
    Prépare un extrait des sources brutes pour le contexte de validation.

    Args:
        raw_results: Liste des articles bruts
        max_articles: Nombre maximum d'articles à inclure

    Returns:
        Texte formaté avec les extraits des sources
    """
    if not raw_results:
        return "Aucune source brute disponible"

    lignes = []
    for i, article in enumerate(raw_results[:max_articles], 1):
        lignes.append(
            f"SOURCE {i} | {article.get('source', '?')} | "
            f"{article.get('date', '?')}\n"
            f"Titre : {article.get('titre', '')[:150]}\n"
            f"Extrait : {article.get('contenu', '')[:250]}\n"
        )

    return "\n---\n".join(lignes)


def _generer_rapport_final(
        query: str,
        summary: str,
        validation_result: ValidationResult,
        summary_metadata: Dict,
        timestamps: Dict
) -> str:
    """
    Génère le rapport final complet combinant résumé et validation.

    Ce rapport est le document final présenté à l'utilisateur,
    incluant le résumé structuré et le badge de validation.

    Args:
        query: La requête de recherche originale
        summary: Le résumé Markdown généré
        validation_result: Le résultat de la validation
        summary_metadata: Les métadonnées du résumé
        timestamps: Les horodatages du workflow

    Returns:
        Rapport final complet en Markdown
    """
    score = validation_result["score_fiabilite"]
    decision = validation_result["decision"]

    # Choisir l'emoji selon le score
    if score >= 80:
        badge_score = f"🟢 {score}/100"
        badge_decision = "✅ VALIDÉ"
    elif score >= 60:
        badge_score = f"🟡 {score}/100"
        badge_decision = "⚠️ VALIDÉ AVEC RÉSERVES"
    else:
        badge_score = f"🔴 {score}/100"
        badge_decision = "❌ REJETÉ"

    # Calculer la durée du workflow
    duree = _calculer_duree(timestamps)

    # Construire le rapport section par section
    rapport_lignes = [
        f"# 🤖 Rapport de Veille Informationnelle",
        f"",
        f"**Sujet** : {query}",
        f"**Généré le** : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}",
        f"**Durée d'exécution** : {duree}",
        f"",
        f"---",
        f"",
        f"## 🏅 Badge de Validation",
        f"",
        f"| Critère | Résultat |",
        f"|---------|---------|",
        f"| Score de fiabilité | {badge_score} |",
        f"| Décision | {badge_decision} |",
        f"| Sources analysées | {summary_metadata.get('nombre_articles_traites', 0)} |",
        f"| Sources uniques | {summary_metadata.get('nombre_sources_uniques', 0)} |",
        f"| Thèmes couverts | {summary_metadata.get('nombre_thematiques', 0)} |",
        f"",
        f"---",
        f"",
        f"{summary}",
        f"",
        f"---",
        f"",
        f"## 🔍 Rapport de Validation Détaillé",
        f"",
        f"### ✅ Points Forts",
    ]

    # Ajouter les points forts
    if validation_result["points_forts"]:
        for point in validation_result["points_forts"]:
            rapport_lignes.append(f"- {point}")
    else:
        rapport_lignes.append("- Aucun point fort identifié")

    rapport_lignes.extend([
        f"",
        f"### ⚠️ Points à Vérifier",
    ])

    # Ajouter les points douteux
    if validation_result["points_douteux"]:
        for point in validation_result["points_douteux"]:
            rapport_lignes.append(f"- {point}")
    else:
        rapport_lignes.append("- Aucun point douteux identifié ✅")

    rapport_lignes.extend([
        f"",
        f"### 🔄 Contradictions Détectées",
    ])

    # Ajouter les contradictions
    if validation_result["contradictions"]:
        for contradiction in validation_result["contradictions"]:
            rapport_lignes.append(f"- ⚡ {contradiction}")
    else:
        rapport_lignes.append("- Aucune contradiction détectée ✅")

    rapport_lignes.extend([
        f"",
        f"### 📝 Justification",
        f"",
        f"{validation_result['justification']}",
        f"",
        f"### 💡 Recommandations",
        f""
    ])

    # Ajouter les recommandations
    if validation_result["recommandations"]:
        for reco in validation_result["recommandations"]:
            rapport_lignes.append(f"- {reco}")
    else:
        rapport_lignes.append("- Aucune recommandation particulière")

    rapport_lignes.extend([
        f"",
        f"---",
        f"",
        f"## ⏱️ Chronologie du Workflow",
        f"",
        f"| Étape | Début | Fin |",
        f"|-------|-------|-----|",
        f"| 🔍 Recherche | {timestamps.get('search_start', 'N/A')[:19]} | "
        f"{timestamps.get('search_end', 'N/A')[:19]} |",
        f"| 📝 Résumé | {timestamps.get('summary_start', 'N/A')[:19]} | "
        f"{timestamps.get('summary_end', 'N/A')[:19]} |",
        f"| ✅ Validation | {timestamps.get('validation_start', 'N/A')[:19]} | "
        f"{timestamps.get('validation_end', 'N/A')[:19]} |",
        f"",
        f"---",
        f"",
        f"*Rapport généré automatiquement par le système de veille multi-agents*"
    ])

    return "\n".join(rapport_lignes)


def _calculer_duree(timestamps: Dict) -> str:
    """
    Calcule la durée totale du workflow depuis les timestamps.

    Args:
        timestamps: Dictionnaire des horodatages du workflow

    Returns:
        Durée formatée en secondes ou minutes
    """
    try:
        debut = datetime.fromisoformat(
            timestamps.get("search_start", datetime.now().isoformat())
        )
        fin = datetime.fromisoformat(
            timestamps.get("validation_end", datetime.now().isoformat())
        )

        duree_secondes = (fin - debut).total_seconds()

        if duree_secondes < 60:
            return f"{duree_secondes:.1f} secondes"
        else:
            minutes = int(duree_secondes // 60)
            secondes = int(duree_secondes % 60)
            return f"{minutes}m {secondes}s"

    except Exception:
        return "Durée non calculable"
