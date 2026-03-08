# src/config.py
"""Configuration centrale du système multi-agent."""

from functools import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AgentConfig(BaseModel):
    """Configuration d'un agent individuel."""
    name: str
    model: str = "llama3.2"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    system_prompt: str = ""


class Settings(BaseSettings):
    """Configuration globale de l'application."""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    fast_model: str = "mistral"

    # Système
    max_iterations: int = 10
    debug: bool = False
    log_level: str = "INFO"

    # Agents par défaut
    researcher_config: AgentConfig = AgentConfig(
        name="Researcher",
        model="llama3.2",
        temperature=0.3,  # Plus déterministe pour la recherche
        system_prompt="""Tu es un expert en recherche et analyse.
        Ton rôle est d'analyser les sujets en profondeur,
        de structurer l'information et d'identifier les points clés.
        Sois factuel, précis et exhaustif."""
    )

    writer_config: AgentConfig = AgentConfig(
        name="Writer",
        model="llama3.2",
        temperature=0.8,  # Plus créatif pour l'écriture
        system_prompt="""Tu es un rédacteur expert.
        Tu crées du contenu clair, engageant et bien structuré
        à partir des recherches fournies.
        Adapte ton style au contexte demandé."""
    )

    critic_config: AgentConfig = AgentConfig(
        name="Critic",
        model="mistral",
        temperature=0.4,
        system_prompt="""Tu es un critique constructif et rigoureux.
        Tu évalues le contenu selon ces critères :
        - Exactitude factuelle
        - Clarté et structure
        - Complétude
        - Style et lisibilité
        Donne un score /10 et des suggestions précises d'amélioration."""
    )

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


@lru_cache()
def get_settings() -> Settings:
    """Retourne les paramètres (cached singleton)."""
    return Settings()