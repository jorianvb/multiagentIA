"""
Logger structuré pour le Meta-MAS.

Utilise loguru pour le logging et Rich pour l'affichage CLI.
"""
import sys
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Console Rich globale
console = Console()


def setup_logger(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure loguru avec formatage colorisé.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR).
        log_file: Chemin optionnel vers le fichier de log.
    """
    logger.remove()

    # Handler console avec couleurs
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # Handler fichier (rotation 10 MB)
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week",
            encoding="utf-8",
        )


def print_banner() -> None:
    """Affiche la bannière du Meta-MAS."""
    banner = Text()
    banner.append("\n")
    banner.append("  ╔══════════════════════════════════════════════════╗\n", style="bold cyan")
    banner.append("  ║              🤖  Meta-MAS  v0.1.0               ║\n", style="bold cyan")
    banner.append("  ║    Multi-Agent System Generator & Orchestrator  ║\n", style="cyan")
    banner.append("  ║           Powered by Ollama (local LLM)         ║\n", style="cyan")
    banner.append("  ╚══════════════════════════════════════════════════╝\n", style="bold cyan")
    console.print(banner)


def print_stage(stage_name: str, description: str) -> None:
    """Affiche un en-tête d'étape du pipeline."""
    console.print()
    console.print(
        Panel(
            f"[bold white]{description}[/bold white]",
            title=f"[bold cyan]🔄 {stage_name}[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def print_agent_action(agent_name: str, action: str, detail: str = "") -> None:
    """Affiche une action d'agent formatée."""
    detail_str = f" [dim]{detail}[/dim]" if detail else ""
    console.print(
        f"  [bold cyan]▶ {agent_name}[/bold cyan] "
        f"[yellow]{action}[/yellow]{detail_str}"
    )


def print_success(message: str) -> None:
    """Affiche un message de succès."""
    console.print(f"[bold green]✅ {message}[/bold green]")


def print_error(message: str) -> None:
    """Affiche un message d'erreur."""
    console.print(f"[bold red]❌ {message}[/bold red]")


def print_warning(message: str) -> None:
    """Affiche un avertissement."""
    console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")


def print_info(message: str) -> None:
    """Affiche une information."""
    console.print(f"[bold blue]ℹ️  {message}[/bold blue]")


def print_agents_table(agents_info: list) -> None:
    """
    Affiche un tableau des agents avec leurs statuts.

    Args:
        agents_info: Liste de dicts {name, role, status}.
    """
    table = Table(
        title="[bold cyan]Agents du MAS généré[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Nom", style="white")
    table.add_column("Rôle", style="cyan")
    table.add_column("Responsabilités", style="dim")

    for agent in agents_info:
        responsibilities = "\n".join(
            f"• {r}" for r in agent.get("responsibilities", [])[:3]
        )
        table.add_row(
            agent.get("name", "?"),
            agent.get("role", "?"),
            responsibilities,
        )
    console.print(table)

