"""
Configuration globale du Meta-MAS via pydantic-settings.
Lit les variables depuis .env ou l'environnement.
"""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Paramètres centraux du Meta-MAS."""

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_default_model: str = Field(default="mistral", alias="OLLAMA_DEFAULT_MODEL")
    ollama_timeout: int = Field(default=120, alias="OLLAMA_TIMEOUT")

    # Modèle spécifique par agent (override possible)
    ollama_model: str = "mistral"

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")

    # Pipeline
    max_validation_retries: int = Field(default=3, alias="MAX_VALIDATION_RETRIES")
    max_generation_retries: int = Field(default=3, alias="MAX_GENERATION_RETRIES")
    llm_timeout: int = 120

    # Répertoire cible du MAS généré
    target_directory: str = "./generated_mas"

    # Répertoire racine du projet
    project_root: Path = Path(__file__).parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Instance globale
_settings: Optional[Settings] = None


def get_settings(**overrides) -> Settings:
    """Retourne l'instance de Settings (singleton avec overrides)."""
    global _settings
    if _settings is None or overrides:
        _settings = Settings(**overrides)
    return _settings

