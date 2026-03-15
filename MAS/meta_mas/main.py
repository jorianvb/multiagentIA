"""
Meta-MAS — Point d'entrée principal.

Lance le système multi-agents méta qui génère et déploie
d'autres systèmes multi-agents dans un dossier cible.

Usage :
    uv run python main.py
    uv run python main.py --graph          # Afficher uniquement le graphe
    uv run python main.py --demo           # Mode démo (sans LLM)
"""
import asyncio
import sys
from pathlib import Path

# ── Ajouter le répertoire racine au sys.path ──────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from loguru import logger

from config.settings import Settings
from core.agent_registry import AgentRegistry
from core.memory import SharedMemory
from core.message_bus import MessageBus
from agents.orchestrator import OrchestratorAgent
from utils.logger import setup_logger, print_banner, print_success, print_error, print_warning
from utils.graph import (
    print_agents_graph,
    print_agent_status_table,
    print_memory_state,
    console,
)
from utils.ollama_client import OllamaClient

# ─────────────────────────────────────────────────────────────────────────────


async def check_ollama(model: str, base_url: str = "http://localhost:11434") -> bool:
    """Vérifie qu'Ollama est disponible et que le modèle est chargé."""
    async with OllamaClient(model=model, base_url=base_url, timeout=8) as client:
        healthy = await client.check_health()
        if not healthy:
            return False
        models = await client.list_models()
        return any(m.startswith(model) for m in models)


def collect_user_input(settings: Settings) -> dict:
    """
    Collecte interactivement les paramètres utilisateur via Rich CLI.

    Returns:
        Dict avec user_request, target_dir, model, advanced_options.
    """
    console.print(
        Panel(
            "[cyan]Renseignez les paramètres ci-dessous pour générer votre MAS.[/cyan]",
            title="[bold]⚙️  Configuration[/bold]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    # ── 1. Description ───────────────────────────────────────────────────────
    console.print(
        "\n[bold yellow]1. Décrivez le système multi-agents à créer :[/bold yellow]"
    )
    console.print(
        "[dim]  Exemple : 'Un MAS de service client avec un agent de tri des requêtes, "
        "un agent FAQ, un agent d\\'escalade et un agent de satisfaction'[/dim]"
    )
    user_request = Prompt.ask("\n[cyan]Description[/cyan]")
    if not user_request.strip():
        console.print("[red]Erreur : la description ne peut pas être vide.[/red]")
        sys.exit(1)

    # ── 2. Dossier cible ─────────────────────────────────────────────────────
    console.print(
        "\n[bold yellow]2. Dossier de destination du MAS généré :[/bold yellow]"
    )
    console.print("[dim]  Le dossier sera créé s'il n'existe pas.[/dim]")
    target_dir = Prompt.ask(
        "[cyan]Dossier cible[/cyan]",
        default="./generated_mas",
    )

    # ── 3. Modèle Ollama ─────────────────────────────────────────────────────
    console.print("\n[bold yellow]3. Modèle Ollama à utiliser :[/bold yellow]")
    console.print("[dim]  Recommandés : mistral · llama3 · llama3.2 · codellama[/dim]")
    model = Prompt.ask("[cyan]Modèle[/cyan]", default="mistral")

    # ── 4. Options avancées ──────────────────────────────────────────────────
    advanced: dict = {}
    console.print()
    if Confirm.ask("[bold yellow]4. Configurer les options avancées ?[/bold yellow]", default=False):
        advanced["max_retries"] = int(
            Prompt.ask("  Max tentatives de validation", default="3")
        )
        advanced["timeout"] = int(
            Prompt.ask("  Timeout LLM (secondes)", default="120")
        )
        advanced["verbose"] = Confirm.ask("  Mode verbose (DEBUG) ?", default=False)

    return {
        "user_request": user_request,
        "target_dir": target_dir,
        "model": model,
        "advanced": advanced,
    }


def print_summary(params: dict) -> None:
    """Affiche le récapitulatif de la configuration."""
    table = Table(
        title="[bold cyan]📋 Récapitulatif[/bold cyan]",
        show_header=False,
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Paramètre", style="cyan", min_width=24)
    table.add_column("Valeur", style="white")

    desc = params["user_request"]
    table.add_row("Description", desc[:80] + ("…" if len(desc) > 80 else ""))
    table.add_row("Dossier cible", params["target_dir"])
    table.add_row("Modèle Ollama", params["model"])

    for k, v in params.get("advanced", {}).items():
        table.add_row(k, str(v))

    console.print(table)


def print_result_panel(result: dict) -> None:
    """Affiche le panneau de résultat final."""
    if result.get("success"):
        target = result.get("target_directory", "?")
        agents_count = result.get("agents_count", 0)
        files_count = result.get("files_count", 0)
        score = result.get("validation_score", 0)
        system_name = result.get("system_name", "MAS")

        # Arborescence
        tree_str = result.get("tree", "")

        launch_cmds = (
            f"[bold white]cd[/bold white] [yellow]{target}[/yellow]\n"
            f"[bold white]uv sync[/bold white]\n"
            f"[bold white]uv run python main.py[/bold white]"
        )

        body = (
            f"[bold white]✅ {system_name}[/bold white] généré avec succès !\n\n"
            f"[dim]Agents créés    :[/dim]  [cyan]{agents_count}[/cyan]\n"
            f"[dim]Fichiers créés  :[/dim]  [cyan]{files_count}[/cyan]\n"
            f"[dim]Score validation:[/dim]  [cyan]{score:.0f}/100[/cyan]\n"
            f"[dim]Emplacement      :[/dim]  [yellow]{target}[/yellow]\n\n"
            f"[bold cyan]Pour lancer votre MAS :[/bold cyan]\n"
            f"{launch_cmds}"
        )

        console.print()
        console.print(
            Panel(
                body,
                title="[bold green]🎉 Déploiement réussi ![/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        if tree_str:
            console.print()
            console.print("[bold cyan]📁 Arborescence du MAS généré :[/bold cyan]")
            console.print(tree_str, style="dim")

    else:
        error = result.get("error", "Erreur inconnue")
        console.print()
        console.print(
            Panel(
                f"[bold white]{error}[/bold white]",
                title="[bold red]❌ Échec du pipeline[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

async def main(show_graph_only: bool = False, demo_mode: bool = False) -> None:
    """
    Fonction principale async du Meta-MAS.

    Args:
        show_graph_only: Afficher uniquement le graphe du pipeline.
        demo_mode      : Tester sans appel LLM réel.
    """
    # ── Bannière ─────────────────────────────────────────────────────────────
    print_banner()

    # ── Mode --graph ─────────────────────────────────────────────────────────
    if show_graph_only:
        print_agents_graph()
        console.print(
            "[dim]Lancez [bold]uv run python main.py[/bold] pour démarrer le pipeline.[/dim]\n"
        )
        return

    # ── Mode --demo ──────────────────────────────────────────────────────────
    if demo_mode:
        await run_demo()
        return

    console.print(
        "[bold]Bienvenue dans le Meta-MAS — le générateur de Systèmes Multi-Agents.[/bold]\n"
    )

    # ── Afficher le graphe au démarrage ──────────────────────────────────────
    if Confirm.ask("Afficher le graphe du pipeline avant de commencer ?", default=True):
        print_agents_graph()

    # ── Collecte des paramètres ──────────────────────────────────────────────
    params = collect_user_input(Settings())
    console.print()
    print_summary(params)
    console.print()

    # ── Vérification Ollama ──────────────────────────────────────────────────
    model = params["model"]
    console.print(f"[dim]Vérification d'Ollama avec le modèle '{model}'…[/dim]")

    ollama_ok = await check_ollama(model)
    if ollama_ok:
        print_success(f"Ollama disponible avec le modèle '{model}'")
    else:
        print_warning(f"Modèle '{model}' introuvable dans Ollama")
        console.print(
            f"[yellow]  → Assurez-vous qu'Ollama tourne : [bold]ollama serve[/bold][/yellow]"
        )
        console.print(
            f"[yellow]  → Téléchargez le modèle : [bold]ollama pull {model}[/bold][/yellow]"
        )
        if not Confirm.ask("\nContinuer quand même ?", default=False):
            sys.exit(0)

    # ── Confirmation finale ──────────────────────────────────────────────────
    console.print()
    if not Confirm.ask("[bold]Lancer la génération du MAS ?[/bold]", default=True):
        console.print("[yellow]Annulé.[/yellow]")
        sys.exit(0)

    # ── Initialisation du système ────────────────────────────────────────────
    advanced = params.get("advanced", {})

    if advanced.get("verbose"):
        setup_logger(level="DEBUG")
    else:
        setup_logger(level="INFO")

    settings = Settings(
        ollama_model=model,
        target_directory=params["target_dir"],
        max_validation_retries=advanced.get("max_retries", 3),
        llm_timeout=advanced.get("timeout", 120),
    )

    # Composants partagés
    memory = SharedMemory()
    bus = MessageBus()
    registry = AgentRegistry()

    # Orchestrateur
    orchestrator = OrchestratorAgent(
        model=model,
        message_bus=bus,
        memory=memory,
        registry=registry,
        settings=settings,
    )
    registry.register(orchestrator)
    bus.register_agent(orchestrator.name)

    # ── Lancement du pipeline ────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold green]🚀 Pipeline Meta-MAS démarré…[/bold green]",
            border_style="green",
            padding=(0, 2),
        )
    )
    console.print()

    try:
        result = await orchestrator.run_pipeline(
            user_request=params["user_request"],
            target_directory=params["target_dir"],
        )

        # Afficher l'état final des agents
        console.print()
        console.print(Rule("[bold cyan]État final des agents[/bold cyan]", style="cyan"))
        print_agent_status_table(registry.get_status_all())

        # Afficher l'état mémoire en mode verbose
        if advanced.get("verbose"):
            console.print()
            console.print(Rule("[bold cyan]Mémoire partagée[/bold cyan]", style="cyan"))
            print_memory_state(memory.snapshot())

        # Résultat
        print_result_panel(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrompu par l'utilisateur.[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.exception("Erreur fatale dans le pipeline")
        print_error(f"Erreur fatale : {e}")
        sys.exit(1)


async def run_demo() -> None:
    """
    Mode démo : simule le pipeline sans appels LLM réels.
    Utile pour tester l'interface et le graphe.
    """
    from unittest.mock import AsyncMock

    console.print(
        Panel(
            "[yellow]Mode démo activé — aucun appel LLM réel.[/yellow]\n"
            "[dim]Les agents simulent leurs réponses.[/dim]",
            title="[bold yellow]⚡ Mode Démo[/bold yellow]",
            border_style="yellow",
        )
    )

    # Afficher le graphe
    print_agents_graph()

    # Simuler un résultat
    demo_result = {
        "success": True,
        "target_directory": "./demo_mas",
        "agents_count": 4,
        "files_count": 14,
        "validation_score": 87.5,
        "system_name": "DemoMAS",
        "tree": (
            "./demo_mas\n"
            "├── main.py\n"
            "├── pyproject.toml\n"
            "├── README.md\n"
            "├── config/\n"
            "│   └── settings.py\n"
            "├── agents/\n"
            "│   ├── base_agent.py\n"
            "│   ├── orchestrator.py\n"
            "│   ├── query_handler.py\n"
            "│   └── responder.py\n"
            "├── core/\n"
            "│   ├── orchestrator.py\n"
            "│   └── message_bus.py\n"
            "└── tests/\n"
            "    ├── test_orchestrator.py\n"
            "    └── test_query_handler.py"
        ),
    }

    # Simuler la progression
    import asyncio
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    steps = [
        ("🔍 Analyse des besoins…", 1.0),
        ("🏗️  Conception architecture…", 0.8),
        ("💻 Génération du code…", 1.2),
        ("✅ Validation…", 0.6),
        ("🚀 Déploiement…", 0.5),
    ]

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Pipeline", total=len(steps))
        for desc, delay in steps:
            progress.update(task, description=desc)
            await asyncio.sleep(delay)
            progress.advance(task)

    print_result_panel(demo_result)

    demo_agents = [
        {"name": "orchestrator", "role": "Coordonne les agents"},
        {"name": "query_handler", "role": "Gère les requêtes"},
        {"name": "responder", "role": "Génère les réponses"},
        {"name": "validator", "role": "Valide les réponses"},
    ]
    print_agents_graph(agents=demo_agents)


# ─────────────────────────────────────────────────────────────────────────────
# CLI via argparse
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Meta-MAS — Générateur de Systèmes Multi-Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  uv run python main.py               # Mode interactif\n"
            "  uv run python main.py --graph        # Afficher le graphe\n"
            "  uv run python main.py --demo         # Mode démo (sans LLM)\n"
        ),
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Afficher uniquement le graphe du pipeline Meta-MAS",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Lancer en mode démo (sans appel LLM)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Fix asyncio sur Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    args = parse_args()

    setup_logger(level="INFO")

    asyncio.run(
        main(
            show_graph_only=args.graph,
            demo_mode=args.demo,
        )
    )

