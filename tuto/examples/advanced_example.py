# examples/advanced_example.py
"""
Exemple d'exécution complet du système multi-agent.

Usage:
    python examples/advanced_example.py
    python examples/advanced_example.py --topic "L'IA dans la médecine"
"""

import argparse
import sys
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.prompt import Prompt, IntPrompt

from src.graphs.advanced_graph import run_advanced_workflow
from src.graphs.simple_graph import run_simple_workflow

console = Console()


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Système Multi-Agent avec Ollama & LangGraph"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Sujet à traiter"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "advanced"],
        default="advanced",
        help="Mode d'exécution (défaut: advanced)"
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Nombre maximum de révisions (défaut: 2)"
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="",
        help="Instructions supplémentaires pour les agents"
    )
    return parser.parse_args()


def interactive_mode():
    """Mode interactif avec prompts Rich."""
    console.print("\n[bold blue]═══════════════════════════════════════[/bold blue]")
    console.print("[bold blue]   🤖 Système Multi-Agent Local        [/bold blue]")
    console.print("[bold blue]   Ollama + LangGraph                  [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════[/bold blue]\n")

    # Exemples de sujets
    examples = [
        "L'intelligence artificielle et son impact sur l'emploi",
        "Les énergies renouvelables en 2024",
        "La blockchain et ses applications concrètes",
        "La nutrition et la santé intestinale",
    ]

    console.print("[cyan]Exemples de sujets :[/cyan]")
    for i, ex in enumerate(examples, 1):
        console.print(f"  {i}. {ex}")

    topic = Prompt.ask(
        "\n[bold]Entrez votre sujet",
        default=examples[0]
    )

    instructions = Prompt.ask(
        "[bold]Instructions supplémentaires (optionnel)",
        default="Article informatif pour un public général"
    )

    max_revisions = IntPrompt.ask(
        "[bold]Nombre max de révisions",
        default=2
    )

    mode = Prompt.ask(
        "[bold]Mode",
        choices=["simple", "advanced"],
        default="advanced"
    )

    return topic, instructions, max_revisions, mode


def main():
    """Point d'entrée principal."""
    args = parse_args()

    # Mode interactif si pas de sujet fourni
    if not args.topic:
        topic, instructions, max_revisions, mode = interactive_mode()
    else:
        topic = args.topic
        instructions = args.instructions
        max_revisions = args.max_revisions
        mode = args.mode

    console.print(f"\n[dim]Mode sélectionné: {mode}[/dim]\n")

    try:
        if mode == "simple":
            # Workflow simple : Researcher → Writer
            result = run_simple_workflow(topic, instructions)
            console.print("\n[bold green]✅ Workflow simple terminé ![/bold green]")
            console.print(f"\n[bold]Contenu généré:[/bold]\n{result.draft_content}")

        else:
            # Workflow avancé : avec boucle de révision
            result = run_advanced_workflow(
                topic=topic,
                instructions=instructions,
                max_revisions=max_revisions,
                thread_id=f"session-{topic[:20].replace(' ', '-')}"
            )

            # Résumé final
            console.print("\n[bold green]✅ Workflow avancé terminé ![/bold green]")
            console.print(f"\n[bold]Résumé :[/bold]")
            console.print(f"  • Révisions effectuées : {result.revision_count}")
            console.print(f"  • Agents utilisés : {len(set(o.agent_name for o in result.agent_outputs))}")

            if result.review:
                console.print(f"  • Score final : [bold green]{result.review.overall_score}/10[/bold green]")

            # Sauvegarder le résultat
            output_file = Path(f"output_{topic[:30].replace(' ', '_')}.md")
            output_file.write_text(result.final_content or result.draft_content)
            console.print(f"\n[cyan]📄 Résultat sauvegardé dans : {output_file}[/cyan]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Workflow interrompu par l'utilisateur[/yellow]")
        sys.exit(0)
    except RuntimeError as e:
        console.print(f"\n[bold red]❌ Erreur : {e}[/bold red]")
        console.print("[dim]Vérifiez qu'Ollama est bien lancé : ollama serve[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
