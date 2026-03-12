# agents/search_agent.py
# Agent 1 : Recherche web et collecte des actualités

import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState, ArticleResult
from tools.search_tools import creer_outil_recherche
from config.settings import ollama_config, logger


# Prompt système du SearchAgent
# Ce prompt guide le LLM pour analyser et enrichir les résultats de recherche
SEARCH_AGENT_SYSTEM_PROMPT = """Tu es un agent expert en recherche d'informations et en veille informationnelle.

Ton rôle est d'analyser les résultats de recherche bruts fournis et de les enrichir en :
1. Identifiant les informations les plus pertinentes et récentes
2. Évaluant la qualité et la fiabilité de chaque source
3. Organisant les résultats par ordre de pertinence
4. Identifiant les thèmes principaux qui émergent des résultats
5. Signalant les sources potentiellement peu fiables

Tu dois répondre en JSON valide avec la structure suivante :
{
    "articles_analyses": [
        {
            "titre": "string",
            "url": "string", 
            "date": "string",
            "contenu": "string",
            "source": "string",
            "score_pertinence": 0-10,
            "fiabilite_source": "haute|moyenne|faible",
            "themes": ["theme1", "theme2"]
        }
    ],
    "themes_principaux": ["theme1", "theme2", "theme3"],
    "nombre_sources": number,
    "periode_couverte": "string",
    "observations": "string"
}

Réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après."""


def search_agent(state: AgentState) -> Dict[str, Any]:
    """
    Nœud LangGraph : Agent de recherche web.

    Cet agent effectue une recherche web sur le sujet demandé,
    collecte les articles bruts, puis utilise le LLM pour analyser
    et enrichir les résultats.

    Args:
        state: L'état actuel du workflow contenant la requête de recherche

    Returns:
        Dictionnaire avec les champs du state à mettre à jour :
        - raw_results: Liste des articles collectés
        - search_metadata: Métadonnées sur la recherche
        - current_step: Étape suivante
        - errors: Liste des erreurs éventuelles
        - timestamps: Horodatage de l'étape
    """
    logger.info("🔍 [SearchAgent] Démarrage de la recherche...")

    # Récupérer la requête depuis le state
    query = state.get("query", "")
    errors = state.get("errors", [])
    timestamps = state.get("timestamps", {})

    # Enregistrer le début de l'étape
    timestamps["search_start"] = datetime.now().isoformat()

    if not query:
        erreur = "❌ Aucune requête fournie dans le state"
        logger.error(erreur)
        errors.append(erreur)
        return {
            "raw_results": [],
            "search_metadata": {"erreur": erreur},
            "current_step": "error",
            "errors": errors,
            "timestamps": timestamps
        }

    logger.info(f"🎯 Requête reçue : '{query}'")

    # -------------------------
    # PHASE 1 : Recherche web
    # -------------------------
    raw_results: List[ArticleResult] = []

    try:
        # Créer l'outil de recherche selon la configuration
        outil_recherche = creer_outil_recherche()

        # Effectuer la recherche d'actualités en priorité
        logger.info("📡 Lancement de la recherche d'actualités...")

        if hasattr(outil_recherche, 'rechercher_actualites'):
            # DuckDuckGo : essayer la recherche news en premier
            raw_results = outil_recherche.rechercher_actualites(query)

            # Si pas assez de résultats, compléter avec une recherche classique
            if len(raw_results) < 3:
                logger.info(
                    f"⚠️ Seulement {len(raw_results)} résultats news, "
                    f"complétion avec recherche classique..."
                )
                resultats_supplementaires = outil_recherche.rechercher(query)

                # Éviter les doublons par URL
                urls_existantes = {r["url"] for r in raw_results}
                for resultat in resultats_supplementaires:
                    if resultat["url"] not in urls_existantes:
                        raw_results.append(resultat)
                        urls_existantes.add(resultat["url"])
        else:
            # Tavily : recherche directe
            raw_results = outil_recherche.rechercher(query)

        logger.info(
            f"✅ Collecte terminée : {len(raw_results)} articles trouvés"
        )

    except Exception as e:
        erreur = f"❌ Erreur lors de la recherche web : {str(e)}"
        logger.error(erreur)
        errors.append(erreur)

        # Si aucun résultat, on ne peut pas continuer
        if not raw_results:
            return {
                "raw_results": [],
                "search_metadata": {"erreur": erreur, "query": query},
                "current_step": "error",
                "errors": errors,
                "timestamps": timestamps
            }

    # -------------------------
    # PHASE 2 : Analyse LLM
    # -------------------------
    logger.info("🤖 Analyse LLM des résultats bruts...")

    try:
        # Préparer le contexte pour le LLM
        # Limiter le contenu pour éviter de dépasser le contexte du LLM
        articles_pour_llm = []
        for article in raw_results[:8]:  # Maximum 8 articles pour le LLM
            articles_pour_llm.append({
                "titre": article["titre"][:200],
                "url": article["url"],
                "date": article["date"],
                "contenu": article["contenu"][:500],  # Limiter le contenu
                "source": article["source"]
            })

        # Construire le message pour le LLM
        message_utilisateur = f"""Analyse ces résultats de recherche pour la requête : "{query}"

RÉSULTATS BRUTS :
{json.dumps(articles_pour_llm, ensure_ascii=False, indent=2)}

Analyse et enrichis ces résultats selon les instructions du prompt système.
Identifie les thèmes principaux et évalue la fiabilité des sources."""

        # Créer le LLM avec une température basse pour des analyses cohérentes
        llm = ollama_config.creer_llm(temperature=0.1)

        # Invoquer le LLM avec les messages système et utilisateur
        messages = [
            SystemMessage(content=SEARCH_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=message_utilisateur)
        ]

        response = llm.invoke(messages)

        # Parser la réponse JSON du LLM
        try:
            # Nettoyer la réponse (parfois le LLM ajoute du texte autour du JSON)
            contenu_response = response.content.strip()

            # Chercher le JSON dans la réponse
            debut_json = contenu_response.find("{")
            fin_json = contenu_response.rfind("}") + 1

            if debut_json >= 0 and fin_json > debut_json:
                json_str = contenu_response[debut_json:fin_json]
                analyse_llm = json.loads(json_str)

                # Enrichir les résultats bruts avec l'analyse LLM
                articles_enrichis = analyse_llm.get("articles_analyses", [])

                if articles_enrichis:
                    # Mettre à jour raw_results avec les données enrichies
                    # Convertir au format ArticleResult standard
                    raw_results_enrichis: List[ArticleResult] = []
                    for article_enrichi in articles_enrichis:
                        article_normalise: ArticleResult = {
                            "titre": article_enrichi.get("titre", ""),
                            "url": article_enrichi.get("url", ""),
                            "date": article_enrichi.get("date", ""),
                            "contenu": article_enrichi.get("contenu", ""),
                            "source": article_enrichi.get("source", "")
                        }
                        raw_results_enrichis.append(article_normalise)

                    raw_results = raw_results_enrichis

                    # Préparer les métadonnées enrichies
                    metadata = {
                        "query": query,
                        "nombre_resultats": len(raw_results),
                        "moteur_recherche": app_config_info(),
                        "themes_principaux": analyse_llm.get(
                            "themes_principaux", []
                        ),
                        "periode_couverte": analyse_llm.get(
                            "periode_couverte", "Non déterminée"
                        ),
                        "observations": analyse_llm.get("observations", ""),
                        "timestamp_recherche": datetime.now().isoformat()
                    }

                    logger.info(
                        f"✅ Analyse LLM réussie - Thèmes : "
                        f"{', '.join(metadata['themes_principaux'])}"
                    )
                else:
                    # L'analyse LLM n'a pas retourné d'articles, garder les bruts
                    logger.warning(
                        "⚠️ L'analyse LLM n'a pas retourné d'articles enrichis. "
                        "Utilisation des résultats bruts."
                    )
                    metadata = _creer_metadata_basique(query, raw_results)
            else:
                logger.warning("⚠️ Pas de JSON valide dans la réponse LLM")
                metadata = _creer_metadata_basique(query, raw_results)

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Erreur parsing JSON LLM : {str(e)}")
            metadata = _creer_metadata_basique(query, raw_results)

    except Exception as e:
        erreur = f"⚠️ Analyse LLM échouée (utilisation résultats bruts) : {str(e)}"
        logger.warning(erreur)
        errors.append(erreur)
        metadata = _creer_metadata_basique(query, raw_results)

    # Enregistrer la fin de l'étape
    timestamps["search_end"] = datetime.now().isoformat()

    logger.info(
        f"🏁 [SearchAgent] Terminé - {len(raw_results)} articles collectés"
    )

    # Retourner les mises à jour du state
    return {
        "raw_results": raw_results,
        "search_metadata": metadata,
        "current_step": "summary",
        "errors": errors,
        "timestamps": timestamps
    }


def _creer_metadata_basique(
        query: str,
        raw_results: List[ArticleResult]
) -> Dict[str, Any]:
    """
    Crée des métadonnées basiques sans analyse LLM.

    Utilisé comme fallback si l'analyse LLM échoue.

    Args:
        query: La requête de recherche
        raw_results: Les résultats bruts collectés

    Returns:
        Dictionnaire de métadonnées basiques
    """
    sources_uniques = list({r["source"] for r in raw_results if r["source"]})

    return {
        "query": query,
        "nombre_resultats": len(raw_results),
        "moteur_recherche": "automatique",
        "themes_principaux": [],
        "periode_couverte": "Non analysée",
        "observations": "Métadonnées basiques (analyse LLM non disponible)",
        "sources_uniques": sources_uniques,
        "timestamp_recherche": datetime.now().isoformat()
    }


def app_config_info() -> str:
    """
    Retourne le nom du moteur de recherche utilisé depuis la config.

    Returns:
        Nom du moteur de recherche en cours d'utilisation
    """
    from config.settings import app_config
    return app_config.search_engine
