# src/utils/logger.py
"""Logging structuré avec Rich pour une belle sortie console."""

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from datetime import datetime

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog avec un renderer lisible."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )


def log_agent_start(agent_name: str, task: str) -> None:
    """Affiche le démarrage d'un agent."""
    console.print(Panel(
        f"[bold cyan]🤖 Agent:[/bold cyan] {agent_name}\n"
        f"[bold yellow]📋 Tâche:[/bold yellow] {task[:100]}...",
        title="[bold green]▶ Agent Démarré[/bold green]",
        border_style="green"
    ))


def log_agent_output(agent_name: str, output: str, score: float | None = None) -> None:
    """Affiche la sortie d'un agent."""
    score_str = f" | Score: [bold]{score}/10[/bold]" if score else ""
    console.print(Panel(
        output[:500] + "..." if len(output) > 500 else output,
        title=f"[bold blue]✅ {agent_name}{score_str}[/bold blue]",
        border_style="blue"
    ))


def log_transition(from_agent: str, to_agent: str, reason: str = "") -> None:
    """Affiche une transition entre agents."""
    console.print(
        f"\n[bold magenta]🔄 Transition:[/bold magenta] "
        f"[cyan]{from_agent}[/cyan] → [cyan]{to_agent}[/cyan]"
        + (f" | [dim]{reason}[/dim]" if reason else "")
    )


def log_error(agent_name: str, error: str) -> None:
    """Affiche une erreur."""
    console.print(Panel(
        f"[red]{error}[/red]",
        title=f"[bold red]❌ Erreur dans {agent_name}[/bold red]",
        border_style="red"
    ))


def log_workflow_complete(final_content: str, iterations: int) -> None:
    """Affiche la complétion du workflow."""
    console.print(Panel(
        f"[bold green]Workflow terminé en {iterations} itération(s)[/bold green]\n\n"
        f"{final_content}",
        title="[bold green]🎉 Résultat Final[/bold green]",
        border_style="gold1"
    ))
