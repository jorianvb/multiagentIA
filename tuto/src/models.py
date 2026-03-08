# src/models.py
"""Modèles Pydantic pour l'état et les messages du système."""

from typing import Annotated, Any
from enum import Enum
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AgentRole(str, Enum):
    """Rôles disponibles dans le système."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic"
    SUPERVISOR = "supervisor"


class ReviewScore(BaseModel):
    """Score de review de l'agent critique."""
    overall_score: float = Field(ge=0, le=10)
    accuracy_score: float = Field(ge=0, le=10)
    clarity_score: float = Field(ge=0, le=10)
    completeness_score: float = Field(ge=0, le=10)
    feedback: str
    needs_revision: bool
    revision_instructions: str = ""


class AgentOutput(BaseModel):
    """Sortie standardisée d'un agent."""
    agent_name: str
    role: AgentRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None


class WorkflowState(BaseModel):
    """
    État global partagé entre tous les agents.

    Utilise Annotated avec operator.add pour les listes
    afin de permettre l'accumulation (pattern LangGraph).
    """
    # Input
    topic: str = ""
    instructions: str = ""

    # Messages (accumulés via operator.add)
    messages: Annotated[list[BaseMessage], operator.add] = Field(
        default_factory=list
    )

    # Outputs des agents
    research_output: str = ""
    draft_content: str = ""
    final_content: str = ""

    # Review
    review: ReviewScore | None = None
    revision_count: int = 0
    max_revisions: int = 3

    # Contrôle du flux
    current_agent: str = ""
    next_agent: str = ""
    is_complete: bool = False
    error_message: str = ""

    # Historique pour debug
    agent_outputs: list[AgentOutput] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
