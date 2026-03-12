# agents/summary_agent.py
# Agent 2 : Condensation et résumé structuré des résultats

import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState, ArticleResult
from config.settings import ollama_config, logger


# Prompt système du SummaryAgent
# Ce prompt est précis et structuré pour obtenir un markdown cohérent
SUMMARY_AGENT_SYSTEM_PROMPT = """Tu es un expert en synthèse d'information et en veille stratégique.

Ton rôle est de transformer des résultats de recherche bruts en un résumé 
structuré, clair et exploitable en français.

INSTRUCTIONS STRICTES :
1. Regroupe les informations par THÉMATIQUES (maximum 5 thèmes)
2. Élimine TOUTES les redondances et répétitions
3. Conserve UNIQUEMENT les faits importants et vérifiables
4. Cite les sources entre parenthèses sous forme (Source: nom_du_site)
5. Utilise un langage professionnel et neutre
6. Mets en évidence les informations les plus récentes
7. Signale clairement si une information semble contradictoire avec [CONFLIT]

FORMAT DE SORTIE OBLIGATOIRE en Markdown :

# 📰 Résumé : [TITRE DU SUJET]

## 📅 Date du résumé : [DATE]
## 🔢 Sources analysées : [NOMBRE]

---

## 🎯 Points Clés
- Point clé 1
- Point clé 2  
- Point clé 3

---

## 📌 [Thème 1]
Contenu synthétisé du thème 1...
(Source: nom_site1, nom_site2)

## 📌 [Thème 2]
Contenu synthétisé du thème 2...
(Source: nom_site3)

[Continuer pour chaque thème...]

---

## 🔗 Sources Consultées
| # | Titre | Source | Date |
|---|-------|--------|------|
| 1 | Titre article | source.com | date |

---

## ⚡ Conclusion
Synthèse finale en 2-3 phrases maximum.

Réponds UNIQUEMENT avec le contenu Markdown, sans JSON, sans texte avant ou après."""


def summary_agent(state: AgentState) -> Dict[str, Any]:
    """
    Nœud LangGraph : Agent de synthèse et condensation.

    Reçoit les résultats bruts du SearchAgent et génère un résumé
    structuré en Markdown via le LLM Ollama.

    Args:
        state: L'état actuel contenant raw_results du SearchAgent

    Returns:
        Dictionnaire avec les champs mis à jour :
        - summary: Le résumé formaté en Markdown
        - summary_metadata: Métadonnées sur le résumé généré
        - current_step: Étape suivante ('validation')
        - errors: Erreurs éventuelles ajoutées
        - timestamps: Horodatages mis à jour
    """
    logger.info("📝 [SummaryAgent] Démarrage de la synthèse...")

    # Récupérer les données depuis le state
    raw_results: List[ArticleResult] = state.get("raw_results", [])
    query: str = state.get("query", "")
    search_metadata: Dict = state.get("search_metadata", {})
    errors: List[str] = state.get("errors", [])
    timestamps: Dict = state.get("timestamps", {})

    # Enregistrer le début de l'étape
    timestamps["summary_start"] = datetime.now().isoformat()

    # Vérification des données d'entrée
    if not raw_results:
        erreur = "❌ Aucun résultat brut disponible pour la synthèse"
        logger.error(erreur)
        errors.append(erreur)
        return {
            "summary": "# ❌ Erreur\nAucune donnée disponible pour générer un résumé.",
            "summary_metadata": {"erreur": erreur},
            "current_step": "error",
            "errors": errors,
            "timestamps": timestamps
        }

    logger.info(
        f"📊 {len(raw_results)} articles à synthétiser pour : '{query}'"
    )

    # ----------------------------------------
    # PHASE 1 : Préparation du contexte LLM
    # ----------------------------------------
    # Préparer les articles de façon optimisée pour le contexte LLM
    # On limite la taille pour ne pas dépasser la fenêtre de contexte d'Ollama
    articles_formates = _formater_articles_pour_llm(raw_results)

    # Construire le prompt avec toutes les informations
    message_utilisateur = f"""Synthétise les informations suivantes sur le sujet : "{query}"

MÉTADONNÉES DE RECHERCHE :
- Nombre de sources : {len(raw_results)}
- Thèmes identifiés : {', '.join(search_metadata.get('themes_principaux', ['Non définis']))}
- Période couverte : {search_metadata.get('periode_couverte', 'Non définie')}

ARTICLES À SYNTHÉTISER :
{articles_formates}

Génère maintenant un résumé complet, structuré et en français selon le format demandé.
Date du jour : {datetime.now().strftime('%d/%m/%Y')}"""

    # ----------------------------------------
    # PHASE 2 : Génération du résumé via LLM
    # ----------------------------------------
    summary = ""

    try:
        logger.info("🤖 Invocation du LLM pour la synthèse...")

        # Température légèrement plus haute pour un résumé plus naturel
        llm = ollama_config.creer_llm(temperature=0.3)

        messages = [
            SystemMessage(content=SUMMARY_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=message_utilisateur)
        ]

        # Invoquer le LLM
        response = llm.invoke(messages)
        summary = response.content.strip()

        # Vérifier que le résumé n'est pas vide
        if not summary:
            raise ValueError("Le LLM a retourné un résumé vide")

        # Vérifier que c'est bien du Markdown (doit commencer par #)
        if not summary.startswith("#"):
            logger.warning(
                "⚠️ Le résumé ne commence pas par un titre Markdown. "
                "Ajout d'un en-tête..."
            )
            summary = f"# 📰 Résumé : {query}\n\n{summary}"

        logger.info(
            f"✅ Résumé généré - {len(summary)} caractères, "
            f"{summary.count(chr(10))} lignes"
        )

    except Exception as e:
        erreur = f"❌ Erreur lors de la génération du résumé : {str(e)}"
        logger.error(erreur)
        errors.append(erreur)

        # Générer un résumé de fallback à partir des données brutes
        summary = _generer_resume_fallback(query, raw_results)
        logger.info("⚠️ Résumé de fallback généré")

    # ----------------------------------------
    # PHASE 3 : Construction des métadonnées
    # ----------------------------------------
    sources_utilisees = list({
        article["source"]
        for article in raw_results
        if article.get("source")
    })

    # Compter les thématiques dans le résumé généré
    nombre_thematiques = summary.count("## 📌")

    summary_metadata = {
        "query": query,
        "nombre_articles_traites": len(raw_results),
        "nombre_sources_uniques": len(sources_utilisees),
        "sources_utilisees": sources_utilisees,
        "nombre_thematiques": nombre_thematiques,
        "longueur_resume_chars": len(summary),
        "timestamp_resume": datetime.now().isoformat()
    }

    # Enregistrer la fin de l'étape
    timestamps["summary_end"] = datetime.now().isoformat()

    logger.info(
        f"🏁 [SummaryAgent] Terminé - "
        f"{nombre_thematiques} thèmes, "
        f"{len(sources_utilisees)} sources uniques"
    )

    return {
        "summary": summary,
        "summary_metadata": summary_metadata,
        "current_step": "validation",
        "errors": errors,
        "timestamps": timestamps
    }


def _formater_articles_pour_llm(
        articles: List[ArticleResult],
        max_articles: int = 8,
        max_contenu_par_article: int = 600
) -> str:
    """
    Formate les articles en texte structuré pour le LLM.

    Limite intelligemment la taille pour rester dans la fenêtre
    de contexte du modèle Ollama.

    Args:
        articles: Liste des articles à formater
        max_articles: Nombre maximum d'articles à inclure
        max_contenu_par_article: Longueur max du contenu par article

    Returns:
        Chaîne de caractères formatée pour le LLM
    """
    lignes = []

    for i, article in enumerate(articles[:max_articles], 1):
        titre = article.get("titre", "Sans titre")[:200]
        source = article.get("source", "Source inconnue")
        date = article.get("date", "Date inconnue")
        contenu = article.get("contenu", "")[:max_contenu_par_article]
        url = article.get("url", "")

        lignes.append(f"""
--- ARTICLE {i} ---
Titre    : {titre}
Source   : {source}
Date     : {date}
URL      : {url}
Contenu  : {contenu}
""")

    if len(articles) > max_articles:
        lignes.append(
            f"\n[... {len(articles) - max_articles} articles supplémentaires "
            f"non inclus pour des raisons de contexte ...]"
        )

    return "\n".join(lignes)


def _generer_resume_fallback(
        query: str,
        articles: List[ArticleResult]
) -> str:
    """
    Génère un résumé basique sans LLM en cas d'erreur.

    Ce fallback garantit qu'un résumé est toujours disponible
    même si le LLM est indisponible ou retourne une erreur.

    Args:
        query: La requête de recherche
        articles: Les articles bruts à résumer

    Returns:
        Résumé formaté en Markdown basique
    """
    date_actuelle = datetime.now().strftime("%d/%m/%Y à %H:%M")
    sources_uniques = list({a["source"] for a in articles if a.get("source")})

    # Construire le résumé ligne par ligne
    lignes = [
        f"# 📰 Résumé : {query}",
        f"",
        f"## 📅 Date du résumé : {date_actuelle}",
        f"## 🔢 Sources analysées : {len(articles)}",
        f"",
        f"---",
        f"",
        f"> ⚠️ **Note** : Ce résumé a été généré en mode dégradé "
        f"(LLM indisponible).",
        f"",
        f"## 📌 Articles Collectés",
        f""
    ]

    for i, article in enumerate(articles[:8], 1):
        titre = article.get("titre", "Sans titre")
        source = article.get("source", "Source inconnue")
        date = article.get("date", "Date inconnue")
        contenu = article.get("contenu", "")[:300]

        lignes.extend([
            f"### {i}. {titre}",
            f"**Source** : {source} | **Date** : {date}",
            f"",
            f"{contenu}",
            f""
        ])

    lignes.extend([
        f"---",
        f"",
        f"## 🔗 Sources Consultées",
        f"",
        f"| # | Source |",
        f"|---|--------|"
    ])

    for i, source in enumerate(sources_uniques, 1):
        lignes.append(f"| {i} | {source} |")

    lignes.extend([
        f"",
        f"---",
        f"",
        f"## ⚡ Conclusion",
        f"",
        f"{len(articles)} articles collectés sur le sujet \"{query}\". "
        f"Résumé automatique généré sans analyse LLM."
    ])

    return "\n".join(lignes)
