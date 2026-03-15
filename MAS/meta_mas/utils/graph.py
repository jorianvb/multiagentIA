"""
Visualisation du graphe du pipeline Meta-MAS.

Affiche l'arborescence des agents et les flux de messages avec Rich.
"""
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Graphe statique du pipeline Meta-MAS
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE_GRAPH = """
╔══════════════════════════════════════════════════════════════════╗
║                    🗺️  GRAPHE DU PIPELINE META-MAS              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   👤 Utilisateur                                                 ║
║        │  description du MAS + dossier cible                    ║
║        ▼                                                         ║
║   ┌─────────────────────────────────────────────────────────┐   ║
║   │              🤖  ORCHESTRATEUR                          │   ║
║   │         (Coordonne tout le pipeline)                    │   ║
║   └──────┬──────────────────────────────────────────────────┘   ║
║          │                                                       ║
║          │  ① ANALYSE                                            ║
║          ▼                                                       ║
║   ┌──────────────────┐                                          ║
║   │  🔍  ANALYSTE    │ ← identifie les agents, rôles, flux     ║
║   └────────┬─────────┘                                          ║
║            │  JSON: {agents, communication_pattern, workflow}    ║
║            │  ② ARCHITECTURE                                     ║
║            ▼                                                     ║
║   ┌──────────────────┐                                          ║
║   │  🏗️   ARCHITECTE │ ← topologie, system_prompts, protocoles ║
║   └────────┬─────────┘                                          ║
║            │  JSON: {topology, agent_details, message_flow}      ║
║            │  ③ GÉNÉRATION                                       ║
║            ▼                                                     ║
║   ┌──────────────────┐       ┌─────────────────────────────┐   ║
║   │ 💻  GÉNÉRATEUR   │──────►│  ✅  VALIDATEUR             │   ║
║   │  de Code         │◄──────│  (syntaxe + structure)      │   ║
║   └────────┬─────────┘  ×3   └─────────────────────────────┘   ║
║            │  Dict: {filename: code_python}                      ║
║            │  ④ DÉPLOIEMENT                                      ║
║            ▼                                                     ║
║   ┌──────────────────┐                                          ║
║   │  🚀  DÉPLOYEUR   │ ← écrit les fichiers dans {cible/}      ║
║   └────────┬─────────┘                                          ║
║            │                                                     ║
║            ▼                                                     ║
║   📁 {dossier_cible}/                                           ║
║      ├── main.py                                                ║
║      ├── agents/                                                ║
║      │   ├── base_agent.py                                      ║
║      │   └── {agent_1..n}.py                                    ║
║      ├── core/                                                  ║
║      │   ├── orchestrator.py                                    ║
║      │   └── message_bus.py                                     ║
║      ├── config/settings.py                                     ║
║      ├── tests/                                                 ║
║      ├── pyproject.toml                                         ║
║      └── README.md                                              ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_pipeline_graph() -> None:
    """Affiche le graphe du pipeline dans le terminal."""
    console.print(PIPELINE_GRAPH, style="cyan")


def print_agents_graph(agents: Optional[List[Dict]] = None) -> None:
    """
    Affiche le graphe des agents du pipeline Meta-MAS sous forme d'arbre Rich.

    Args:
        agents: Liste optionnelle d'agents générés (pour enrichir l'affichage).
    """
    root = Tree(
        "🤖 [bold cyan]Meta-MAS Pipeline[/bold cyan]",
        guide_style="cyan",
    )

    # ── Entrée utilisateur ──────────────────────────────────────────────
    user_node = root.add("👤 [white]Utilisateur[/white] [dim](description + dossier cible)[/dim]")
    user_node.add("[dim]─ description du MAS souhaité[/dim]")
    user_node.add("[dim]─ dossier de destination[/dim]")
    user_node.add("[dim]─ modèle Ollama (mistral, llama3…)[/dim]")

    # ── Orchestrateur ───────────────────────────────────────────────────
    orch_node = root.add(
        "🎯 [bold yellow]Orchestrateur[/bold yellow] "
        "[dim]— coordonne tout le pipeline[/dim]"
    )

    # ── Analyste ────────────────────────────────────────────────────────
    analyst_node = orch_node.add(
        "① 🔍 [bold green]Analyste[/bold green] "
        "[dim]— identifie les agents nécessaires[/dim]"
    )
    analyst_node.add("[dim]→ Entrée : description texte de l'utilisateur[/dim]")
    analyst_node.add("[dim]← Sortie : JSON {agents, rôles, responsabilités}[/dim]")

    # ── Architecte ──────────────────────────────────────────────────────
    arch_node = orch_node.add(
        "② 🏗️  [bold blue]Architecte[/bold blue] "
        "[dim]— conçoit la topologie[/dim]"
    )
    arch_node.add("[dim]→ Entrée : résultat Analyste[/dim]")
    arch_node.add("[dim]← Sortie : JSON {topology, system_prompts, message_flow}[/dim]")
    arch_node.add("[dim]   Patterns : hub_and_spoke | peer_to_peer | blackboard[/dim]")

    # ── Générateur + Validateur ─────────────────────────────────────────
    gen_node = orch_node.add(
        "③ 💻 [bold magenta]Générateur de Code[/bold magenta] "
        "[dim]— génère le Python de chaque agent[/dim]"
    )
    gen_node.add("[dim]→ Entrée : spécifications Analyste + Architecte[/dim]")
    gen_node.add("[dim]← Sortie : Dict {filename → code_python}[/dim]")

    val_node = gen_node.add(
        "↔️  ✅ [bold]Validateur[/bold] "
        "[dim]— boucle de validation (max 3×)[/dim]"
    )
    val_node.add("[dim]→ ast.parse() + contrôle structurel[/dim]")
    val_node.add("[dim]← score 0-100 + liste erreurs/warnings[/dim]")
    val_node.add("[dim]✗ si invalide → retour au Générateur[/dim]")

    # ── Déployeur ───────────────────────────────────────────────────────
    deploy_node = orch_node.add(
        "④ 🚀 [bold red]Déployeur[/bold red] "
        "[dim]— écrit tous les fichiers sur disque[/dim]"
    )
    deploy_node.add("[dim]→ Entrée : fichiers générés + métadonnées[/dim]")
    deploy_node.add("[dim]← Sortie : arborescence dans {dossier_cible}/[/dim]")

    # ── MAS Généré ──────────────────────────────────────────────────────
    output_node = root.add(
        "📁 [bold white]MAS Généré[/bold white] "
        "[dim]— résultat final déployé[/dim]"
    )

    if agents:
        agents_sub = output_node.add(
            f"🤖 [cyan]{len(agents)} agent(s) créé(s)[/cyan]"
        )
        for ag in agents:
            name = ag.get("name", "?")
            role = ag.get("role", "?")
            agents_sub.add(f"[white]{name}[/white] [dim]— {role}[/dim]")
    else:
        output_node.add("[dim]agents/{agent_1..n}.py[/dim]")

    output_node.add("[dim]core/orchestrator.py  +  core/message_bus.py[/dim]")
    output_node.add("[dim]config/settings.py[/dim]")
    output_node.add("[dim]main.py  +  pyproject.toml  +  README.md[/dim]")
    output_node.add("[dim]tests/test_{agent_n}.py[/dim]")

    console.print()
    console.print(
        Panel(
            root,
            title="[bold cyan]🗺️  Graphe du Pipeline Meta-MAS[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def print_agent_status_table(registry_status: Dict[str, Dict]) -> None:
    """
    Affiche un tableau de l'état de tous les agents.

    Args:
        registry_status: Dict retourné par AgentRegistry.get_status_all().
    """
    table = Table(
        title="[bold cyan]État des Agents[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Agent", style="bold white", min_width=16)
    table.add_column("Rôle", style="cyan")
    table.add_column("Modèle", style="dim")
    table.add_column("Statut", justify="center")
    table.add_column("Mémoire", justify="right", style="dim")

    status_styles = {
        "IDLE": "[green]● IDLE[/green]",
        "THINKING": "[yellow]◉ THINKING[/yellow]",
        "ACTING": "[blue]▶ ACTING[/blue]",
        "WAITING": "[dim]◌ WAITING[/dim]",
        "ERROR": "[red]✗ ERROR[/red]",
    }

    for name, info in registry_status.items():
        status_val = info.get("status", "IDLE")
        status_display = status_styles.get(status_val, status_val)
        table.add_row(
            info.get("name", name),
            info.get("role", "?"),
            info.get("model", "?"),
            status_display,
            str(info.get("memory_size", 0)),
        )

    console.print(table)


def print_memory_state(memory_snapshot: Dict[str, Any]) -> None:
    """
    Affiche l'état de la SharedMemory de façon condensée.

    Args:
        memory_snapshot: Snapshot de la SharedMemory.
    """
    table = Table(
        title="[bold cyan]État de la Mémoire Partagée[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Clé", style="cyan", min_width=22)
    table.add_column("Valeur", style="white")

    for key, value in memory_snapshot.items():
        if value is None:
            display = "[dim]—[/dim]"
        elif isinstance(value, list):
            display = f"[dim][{len(value)} éléments][/dim]"
        elif isinstance(value, dict):
            display = f"[dim]{{dict: {len(value)} clés}}[/dim]"
        elif isinstance(value, str) and len(value) > 80:
            display = value[:77] + "…"
        else:
            display = str(value)
        table.add_row(key, display)

    console.print(table)

