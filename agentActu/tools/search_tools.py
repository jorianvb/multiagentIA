# tools/search_tools.py
# Outils de recherche web : DuckDuckGo et Tavily

import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from graph.state import ArticleResult
from config.settings import app_config, logger


class DuckDuckGoTool:
    """
    Outil de recherche utilisant DuckDuckGo Search.

    DuckDuckGo ne nécessite pas de clé API et est idéal pour commencer.
    Limitation : peut être rate-limité si trop de requêtes en peu de temps.

    Avantages:
    - Gratuit, sans clé API
    - Protège la vie privée
    - Résultats récents et variés

    Inconvénients:
    - Rate limiting possible
    - Moins de contrôle sur les résultats
    - Pas de garantie de fraîcheur des données
    """

    def __init__(self, max_results: int = 8, timeout: int = 10):
        """
        Initialise l'outil DuckDuckGo.

        Args:
            max_results: Nombre maximum de résultats à retourner
            timeout: Timeout en secondes pour chaque requête
        """
        self.max_results = max_results
        self.timeout = timeout
        logger.info(f"🦆 DuckDuckGo initialisé (max_results={max_results})")

    def rechercher(
            self,
            query: str,
            region: str = "fr-fr",
            periode: str = "w"
    ) -> List[ArticleResult]:
        """
        Effectue une recherche web avec DuckDuckGo.

        Args:
            query: La requête de recherche
            region: Code région pour les résultats (fr-fr = France)
            periode: Période des résultats ('d'=jour, 'w'=semaine, 'm'=mois, 'y'=année)

        Returns:
            Liste d'ArticleResult normalisés et prêts à être traités

        Raises:
            Exception: Si la recherche échoue après plusieurs tentatives
        """
        logger.info(f"🔍 Recherche DuckDuckGo : '{query}'")

        resultats: List[ArticleResult] = []
        tentatives = 0
        max_tentatives = 3

        while tentatives < max_tentatives:
            try:
                # Utiliser DDGS comme gestionnaire de contexte
                with DDGS() as ddgs:
                    # Effectuer la recherche textuelle
                    resultats_bruts = list(
                        ddgs.text(
                            keywords=query,
                            region=region,
                            timelimit=periode,
                            max_results=self.max_results
                        )
                    )

                # Normaliser chaque résultat au format ArticleResult
                for item in resultats_bruts:
                    article: ArticleResult = {
                        "titre": item.get("title", "Titre non disponible"),
                        "url": item.get("href", ""),
                        "date": item.get("published", datetime.now().isoformat()),
                        "contenu": item.get("body", "Contenu non disponible"),
                        "source": self._extraire_domaine(item.get("href", ""))
                    }
                    resultats.append(article)

                logger.info(
                    f"✅ DuckDuckGo : {len(resultats)} résultats pour '{query}'"
                )
                return resultats

            except DuckDuckGoSearchException as e:
                tentatives += 1
                logger.warning(
                    f"⚠️ Tentative {tentatives}/{max_tentatives} échouée : {str(e)}"
                )
                if tentatives < max_tentatives:
                    # Attendre avant de réessayer (backoff exponentiel)
                    delai = 2 ** tentatives
                    logger.info(f"⏳ Attente de {delai} secondes avant retry...")
                    time.sleep(delai)

            except Exception as e:
                logger.error(f"❌ Erreur inattendue DuckDuckGo : {str(e)}")
                tentatives += 1
                if tentatives >= max_tentatives:
                    raise Exception(
                        f"Échec de la recherche DuckDuckGo après "
                        f"{max_tentatives} tentatives : {str(e)}"
                    )
                time.sleep(2)

        return resultats

    def rechercher_actualites(
            self,
            query: str,
            region: str = "fr-fr"
    ) -> List[ArticleResult]:
        """
        Effectue une recherche spécifique aux actualités avec DuckDuckGo News.

        Utilise l'endpoint News de DuckDuckGo pour obtenir des résultats
        plus récents et pertinents pour la veille informationnelle.

        Args:
            query: La requête de recherche
            region: Code région pour les résultats

        Returns:
            Liste d'ArticleResult depuis les actualités DuckDuckGo
        """
        logger.info(f"📰 Recherche actualités DuckDuckGo : '{query}'")

        resultats: List[ArticleResult] = []

        try:
            with DDGS() as ddgs:
                # Utiliser l'endpoint news pour des actualités plus fraîches
                resultats_news = list(
                    ddgs.news(
                        keywords=query,
                        region=region,
                        safesearch="moderate",
                        timelimit="w",  # Dernière semaine
                        max_results=self.max_results
                    )
                )

            for item in resultats_news:
                article: ArticleResult = {
                    "titre": item.get("title", "Titre non disponible"),
                    "url": item.get("url", ""),
                    "date": item.get("date", datetime.now().isoformat()),
                    "contenu": item.get("body", item.get("excerpt",
                                                         "Contenu non disponible")),
                    "source": item.get("source",
                                       self._extraire_domaine(item.get("url", "")))
                }
                resultats.append(article)

            logger.info(
                f"✅ Actualités DuckDuckGo : {len(resultats)} résultats"
            )
            return resultats

        except Exception as e:
            logger.warning(
                f"⚠️ Échec recherche news, fallback sur recherche classique : {str(e)}"
            )
            # Fallback sur la recherche classique en cas d'échec
            return self.rechercher(query, region)

    def _extraire_domaine(self, url: str) -> str:
        """
        Extrait le nom de domaine d'une URL pour identifier la source.

        Args:
            url: L'URL complète de l'article

        Returns:
            Le nom de domaine extrait ou 'Source inconnue'
        """
        if not url:
            return "Source inconnue"

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            # Retirer le 'www.' pour un nom plus propre
            domaine = parsed.netloc.replace("www.", "")
            return domaine if domaine else "Source inconnue"
        except Exception:
            return "Source inconnue"


class TavilyTool:
    """
    Outil de recherche utilisant l'API Tavily.

    Tavily est une API de recherche spécialement conçue pour les LLM.
    Elle retourne des résultats plus structurés et avec plus de contexte
    que DuckDuckGo, mais nécessite une clé API.

    Avantages:
    - Résultats de haute qualité et récents
    - Contenu plus long et plus riche
    - Spécialement optimisé pour les LLM
    - Filtrage des publicités et contenus non pertinents

    Inconvénients:
    - Clé API requise (1000 req/mois gratuites)
    - Moins de contrôle sur les sources
    """

    def __init__(self, api_key: str = None, max_results: int = 8):
        """
        Initialise l'outil Tavily.

        Args:
            api_key: Clé API Tavily. Si None, utilise TAVILY_API_KEY du .env
            max_results: Nombre maximum de résultats

        Raises:
            ValueError: Si aucune clé API n'est fournie
        """
        self.api_key = api_key or app_config.tavily_api_key
        self.max_results = max_results

        if not self.api_key:
            raise ValueError(
                "Clé API Tavily manquante. "
                "Ajoutez TAVILY_API_KEY dans votre fichier .env "
                "ou obtenez une clé sur https://tavily.com"
            )

        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
            logger.info("🔎 Tavily initialisé avec succès")
        except ImportError:
            raise ImportError(
                "Package tavily-python non installé. "
                "Lancez : pip install tavily-python"
            )

    def rechercher(self, query: str) -> List[ArticleResult]:
        """
        Effectue une recherche avec l'API Tavily.

        Args:
            query: La requête de recherche

        Returns:
            Liste d'ArticleResult normalisés

        Raises:
            Exception: Si l'API Tavily retourne une erreur
        """
        logger.info(f"🔍 Recherche Tavily : '{query}'")

        try:
            # Effectuer la recherche via l'API Tavily
            response = self.client.search(
                query=query,
                search_depth="advanced",  # Recherche approfondie
                max_results=self.max_results,
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                # Filtrer sur les actualités récentes
                topic="news"
            )

            resultats: List[ArticleResult] = []

            # Normaliser les résultats Tavily
            for item in response.get("results", []):
                article: ArticleResult = {
                    "titre": item.get("title", "Titre non disponible"),
                    "url": item.get("url", ""),
                    "date": item.get("published_date",
                                     datetime.now().isoformat()),
                    "contenu": item.get("content", "Contenu non disponible"),
                    "source": self._extraire_domaine(item.get("url", ""))
                }
                resultats.append(article)

            logger.info(f"✅ Tavily : {len(resultats)} résultats pour '{query}'")
            return resultats

        except Exception as e:
            logger.error(f"❌ Erreur Tavily : {str(e)}")
            raise Exception(f"Échec de la recherche Tavily : {str(e)}")

    def _extraire_domaine(self, url: str) -> str:
        """
        Extrait le nom de domaine d'une URL.

        Args:
            url: L'URL complète

        Returns:
            Le nom de domaine sans 'www.'
        """
        if not url:
            return "Source inconnue"
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "") or "Source inconnue"
        except Exception:
            return "Source inconnue"


def creer_outil_recherche(moteur: str = None) -> DuckDuckGoTool | TavilyTool:
    """
    Factory function qui crée et retourne l'outil de recherche approprié.

    Cette fonction centralise la création des outils de recherche et
    choisit automatiquement le bon outil selon la configuration.

    Args:
        moteur: 'duckduckgo' ou 'tavily'. Si None, utilise la config .env

    Returns:
        Instance de l'outil de recherche configuré

    Example:
        >>> outil = creer_outil_recherche()
        >>> resultats = outil.rechercher("IA 2024")
    """
    moteur_choisi = moteur or app_config.search_engine
    max_results = app_config.search_max_results

    if moteur_choisi.lower() == "tavily" and app_config.tavily_api_key:
        logger.info("🔎 Utilisation de Tavily comme moteur de recherche")
        return TavilyTool(
            api_key=app_config.tavily_api_key,
            max_results=max_results
        )
    else:
        if moteur_choisi.lower() == "tavily" and not app_config.tavily_api_key:
            logger.warning(
                "⚠️ Tavily sélectionné mais TAVILY_API_KEY manquante. "
                "Fallback sur DuckDuckGo."
            )
        logger.info("🦆 Utilisation de DuckDuckGo comme moteur de recherche")
        return DuckDuckGoTool(max_results=max_results)
