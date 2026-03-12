# config/settings.py
# Configuration centralisée de l'application

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from rich.logging import RichHandler

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


def configurer_logging() -> logging.Logger:
    """
    Configure le système de logging avec Rich pour un affichage coloré en console
    et sauvegarde dans un fichier de log.

    Returns:
        Logger configuré prêt à l'emploi
    """
    # Créer le répertoire de logs s'il n'existe pas
    Path("logs").mkdir(exist_ok=True)

    # Récupérer le niveau de log depuis les variables d'environnement
    niveau_log = os.getenv("LOG_LEVEL", "INFO")
    fichier_log = os.getenv("LOG_FILE", "logs/veille.log")

    # Configurer les handlers (console + fichier)
    logging.basicConfig(
        level=getattr(logging, niveau_log),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            # Handler console avec Rich pour un affichage coloré et lisible
            RichHandler(rich_tracebacks=True, markup=True),
            # Handler fichier pour conserver les logs
            logging.FileHandler(fichier_log, encoding="utf-8")
        ]
    )

    logger = logging.getLogger("veille_informationnelle")
    return logger


# Logger global de l'application
logger = configurer_logging()


class OllamaConfig:
    """
    Configuration et initialisation du modèle Ollama.

    Cette classe centralise tous les paramètres de connexion à Ollama
    et fournit des méthodes pour créer des instances de LLM configurées.
    """

    def __init__(self):
        """Initialise la configuration depuis les variables d'environnement."""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))

        logger.info(f"🤖 Configuration Ollama : modèle={self.model}, "
                    f"temperature={self.temperature}")

    def creer_llm(
            self,
            temperature: float = None,
            max_tokens: int = None
    ) -> ChatOllama:
        """
        Crée et retourne une instance configurée de ChatOllama.

        Args:
            temperature: Température du modèle (0=déterministe, 1=créatif).
                        Si None, utilise la valeur par défaut de la config.
            max_tokens: Nombre maximum de tokens en sortie.
                       Si None, utilise la valeur par défaut de la config.

        Returns:
            Instance ChatOllama configurée et prête à l'emploi

        Raises:
            ConnectionError: Si le serveur Ollama n'est pas accessible
        """
        # Utiliser les valeurs par défaut si non spécifiées
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            # Créer l'instance ChatOllama
            llm = ChatOllama(
                base_url=self.base_url,
                model=self.model,
                temperature=temp,
                num_predict=tokens,
                # num_ctx définit la fenêtre de contexte (mémoire du modèle)
                num_ctx=8192,
            )

            logger.debug(f"✅ LLM créé : {self.model} (temp={temp}, "
                         f"max_tokens={tokens})")
            return llm

        except Exception as e:
            logger.error(f"❌ Impossible de créer le LLM Ollama : {str(e)}")
            raise ConnectionError(
                f"Connexion à Ollama impossible. "
                f"Vérifiez que 'ollama serve' est en cours d'exécution. "
                f"Erreur : {str(e)}"
            )

    def tester_connexion(self) -> bool:
        """
        Teste la connexion au serveur Ollama et la disponibilité du modèle.

        Returns:
            True si la connexion est opérationnelle, False sinon
        """
        try:
            import httpx

            # Tester que le serveur répond
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5.0
            )

            if response.status_code != 200:
                logger.error(f"❌ Serveur Ollama répond avec code {response.status_code}")
                return False

            # Vérifier que le modèle est disponible
            modeles_disponibles = [
                m["name"] for m in response.json().get("models", [])
            ]

            # Vérifier si le modèle (avec ou sans tag) est disponible
            modele_disponible = any(
                self.model in modele or modele.startswith(self.model)
                for modele in modeles_disponibles
            )

            if not modele_disponible:
                logger.warning(
                    f"⚠️ Modèle '{self.model}' non trouvé. "
                    f"Disponibles : {modeles_disponibles}. "
                    f"Lancez : ollama pull {self.model}"
                )
                return False

            logger.info(f"✅ Connexion Ollama OK - Modèle '{self.model}' disponible")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur de connexion Ollama : {str(e)}")
            return False


class AppConfig:
    """
    Configuration générale de l'application.

    Centralise tous les paramètres de l'application en dehors de la configuration LLM.
    """

    def __init__(self):
        """Initialise tous les paramètres de configuration."""
        # Configuration de la recherche
        self.search_max_results = int(os.getenv("SEARCH_MAX_RESULTS", "8"))
        self.search_engine = os.getenv("SEARCH_ENGINE", "duckduckgo")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

        # Configuration des sorties
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
        self.save_results = os.getenv("SAVE_RESULTS", "true").lower() == "true"

        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir.mkdir(exist_ok=True)

        logger.info(
            f"⚙️ Config App : moteur={self.search_engine}, "
            f"max_résultats={self.search_max_results}"
        )


# Instances globales accessibles dans tout le projet
ollama_config = OllamaConfig()
app_config = AppConfig()
