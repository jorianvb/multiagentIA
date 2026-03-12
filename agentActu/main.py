# main.py
# Point d'entrée principal du système de veille informationnelle multi-agents
# Lance le workflow LangGraph et affiche les résultats formatés

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text

from graph.workflow import construire_workflow, visualiser_workflow
from graph.state import AgentState
from config.settings import app_config, ollama_config, logger

# Initialisation de la console Rich pour un affichage coloré
console = Console()


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def creer_etat_initial(query: str) -> AgentState:
    """
    Crée l'état initial vide du workflow avec la requête utilisateur.

    Cet état est le point de départ transmis au premier agent.
    Tous les champs sont initialisés à des valeurs vides ou nulles
    et seront progressivement remplis par les agents.

    Args:
        query: Le sujet de recherche saisi par l'utilisateur

    Returns:
        Un AgentState initialisé prêt à être injecté dans le graph
    """
    return AgentState({
        "query": query.strip(),
        "raw_results": [],
        "search_metadata": {},
        "summary": "",
        "summary_metadata": {},
        "validation_result": None,
        "final_report": "",
        "current_step": "search",
        "errors": [],
        "timestamps": {
            "workflow_start": datetime.now().isoformat()
        }
    })


def verifier_ollama() -> bool:
    """
    Vérifie qu'Ollama est démarré et que le modèle est disponible.

    Effectue une requête HTTP simple vers l'API Ollama pour
    confirmer que le service est actif et répond correctement.

    Returns:
        True si Ollama est disponible, False sinon
    """
    try:
        import httpx

        # Vérifier que le service Ollama répond
        response = httpx.get(
            f"{ollama_config.base_url}/api/tags",
            timeout=5.0
        )

        if response.status_code != 200:
            logger.error(
                f"Ollama répond avec le code HTTP {response.status_code}"
            )
            return False

        # Vérifier que le modèle demandé est bien installé
        data = response.json()
        modeles_disponibles = [
            m.get("name", "").split(":")[0]
            for m in data.get("models", [])
        ]

        modele_cible = ollama_config.model.split(":")[0]

        if modele_cible not in modeles_disponibles:
            logger.error(
                f"Modèle '{ollama_config.model}' non trouvé. "
                f"Modèles disponibles : {modeles_disponibles}"
            )
            console.print(
                f"[bold red]❌ Modèle '{ollama_config.model}' non installé.[/bold red]\n"
                f"Exécutez : [bold]ollama pull {ollama_config.model}[/bold]"
            )
            return False

        logger.info(
            f"✅ Ollama OK — Modèle '{ollama_config.model}' disponible"
        )
        return True

    except httpx.ConnectError:
        logger.error(
            f"Impossible de se connecter à Ollama sur {ollama_config.base_url}"
        )
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la vérification d'Ollama : {str(e)}")
        return False


def sauvegarder_rapport(
        rapport: str,
        query: str,
        state: AgentState
) -> str:
    """
    Sauvegarde le rapport final dans un fichier Markdown horodaté.

    Crée automatiquement le dossier de sortie si nécessaire.
    Le nom du fichier intègre la date, l'heure et le sujet recherché
    pour faciliter l'archivage et la retrouvabilité.

    Args:
        rapport: Le contenu Markdown du rapport final
        query: La requête de recherche (utilisée dans le nom du fichier)
        state: L'état final du workflow (pour les métadonnées)

    Returns:
        Le chemin absolu du fichier sauvegardé
    """
    try:
        # Créer le dossier de sortie
        dossier_sortie = Path(app_config.output_dir)
        dossier_sortie.mkdir(parents=True, exist_ok=True)

        # Générer un nom de fichier propre et horodaté
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        nom_propre = (
            query.lower()
            .replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            [:50]  # Limiter la longueur du nom
        )
        nom_fichier = f"veille_{horodatage}_{nom_propre}.md"
        chemin_fichier = dossier_sortie / nom_fichier

        # Préparer le contenu avec en-tête de métadonnées
        validation = state.get("validation_result", {}) or {}
        score = validation.get("score_fiabilite", "N/A")
        decision = validation.get("decision", "N/A")
        nb_articles = len(state.get("raw_results", []))
        erreurs = state.get("errors", [])

        en_tete = f"""---
# Métadonnées du rapport
query: "{query}"
date_generation: "{datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}"
score_fiabilite: {score}
decision_validation: "{decision}"
nb_articles_analyses: {nb_articles}
nb_erreurs: {len(erreurs)}
modele_ollama: "{ollama_config.model}"
moteur_recherche: "{app_config.search_engine}"
---

"""
        contenu_final = en_tete + rapport

        # Écrire le fichier
        chemin_fichier.write_text(contenu_final, encoding="utf-8")

        logger.info(f"💾 Rapport sauvegardé : {chemin_fichier.resolve()}")
        return str(chemin_fichier.resolve())

    except Exception as e:
        erreur = f"Impossible de sauvegarder le rapport : {str(e)}"
        logger.error(erreur)
        return ""


def afficher_banniere(query: str) -> None:
    """
    Affiche la bannière de démarrage du workflow dans le terminal.

    Présente les informations clés de la session : sujet recherché,
    modèle LLM utilisé et moteur de recherche actif.

    Args:
        query: Le sujet de recherche de la session en cours
    """
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold blue]🤖 Système de Veille Informationnelle Multi-Agents[/bold blue]\n\n"
            f"[cyan]📌 Sujet       :[/cyan] [bold white]{query}[/bold white]\n"
            f"[cyan]🧠 Modèle LLM  :[/cyan] [bold white]{ollama_config.model}[/bold white]\n"
            f"[cyan]🔍 Recherche   :[/cyan] [bold white]{app_config.search_engine.upper()}[/bold white]\n"
            f"[cyan]📅 Démarrage   :[/cyan] [bold white]{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}[/bold white]",
            border_style="blue",
            padding=(1, 4)
        )
    )
    console.print("\n")


def afficher_resultats_finaux(
        etat_final: AgentState,
        chemin_sauvegarde: Optional[str] = None
) -> None:
    """
    Affiche les résultats finaux du workflow dans le terminal.

    Présente le score de validation avec une barre de progression
    colorée, les statistiques du workflow, les éventuels avertissements
    et enfin le rapport complet en Markdown rendu par Rich.

    Args:
        etat_final: L'état final après exécution complète du workflow
        chemin_sauvegarde: Chemin du fichier sauvegardé (optionnel)
    """
    console.print("\n")
    console.print("=" * 60)
    console.print("[bold]📊 RÉSULTATS DU WORKFLOW[/bold]")
    console.print("=" * 60)

    # ----------------------------------------
    # SCORE DE VALIDATION
    # ----------------------------------------
    validation = etat_final.get("validation_result", {}) or {}
    score = validation.get("score_fiabilite", 0)
    decision = validation.get("decision", "INCONNU")
    errors = etat_final.get("errors", [])
    timestamps = etat_final.get("timestamps", {})

    # Choisir la couleur selon le score
    if score >= 80:
        couleur = "green"
        emoji_score = "✅"
    elif score >= 60:
        couleur = "yellow"
        emoji_score = "⚠️"
    else:
        couleur = "red"
        emoji_score = "❌"

    # Construire la barre de progression ASCII
    nb_blocs_pleins = score // 10
    nb_blocs_vides = 10 - nb_blocs_pleins
    barre = "█" * nb_blocs_pleins + "░" * nb_blocs_vides

    console.print(
        Panel(
            f"{emoji_score} [bold {couleur}]Score de Fiabilité : {score}/100[/bold {couleur}]\n\n"
            f"[{couleur}][{barre}] {score}%[/{couleur}]\n\n"
            f"Décision : [bold {couleur}]{decision}[/bold {couleur}]\n\n"
            f"Justification : {validation.get('justification', 'Non disponible')[:300]}",
            title="[bold]🛡️ Résultat de Validation[/bold]",
            border_style=couleur
        )
    )

    # ----------------------------------------
    # TABLEAU DES STATISTIQUES
    # ----------------------------------------
    table = Table(
        title="📊 Statistiques du Workflow",
        show_header=True,
        header_style="bold blue"
    )

    table.add_column("Métrique", style="cyan", width=30)
    table.add_column("Valeur", style="white", width=30)

    table.add_row(
        "Articles collectés",
        str(len(etat_final.get("raw_results", [])))
    )
    table.add_row(
        "Sources uniques",
        str(
            etat_final.get(
                "summary_metadata", {}
            ).get("nombre_sources_uniques", 0)
        )
    )
    table.add_row(
        "Thématiques identifiées",
        str(
            etat_final.get(
                "summary_metadata", {}
            ).get("nombre_thematiques", 0)
        )
    )
    table.add_row(
        "Erreurs rencontrées",
        str(len(errors))
    )

    # Calculer la durée totale du workflow
    try:
        debut = datetime.fromisoformat(
            timestamps.get("workflow_start", "")
        )
        fin = datetime.fromisoformat(
            timestamps.get("validation_end",
                           datetime.now().isoformat())
        )
        duree_secondes = (fin - debut).total_seconds()
        duree_affichage = f"{duree_secondes:.1f}s"
        table.add_row("Durée totale", duree_affichage)
    except Exception:
        table.add_row("Durée totale", "N/A")

    if chemin_sauvegarde:
        table.add_row("Rapport sauvegardé dans", chemin_sauvegarde)

    console.print(table)

    # ----------------------------------------
    # AFFICHAGE DES ERREURS SI PRÉSENTES
    # ----------------------------------------
    if errors:
        console.print(
            "\n[bold yellow]⚠️  Avertissements détectés :[/bold yellow]"
        )
        for i, erreur in enumerate(errors, 1):
            console.print(f"  {i}. {erreur}", style="yellow")

    # ----------------------------------------
    # AFFICHAGE DU RAPPORT FINAL EN MARKDOWN
    # ----------------------------------------
    rapport = etat_final.get("final_report", "")

    if rapport:
        console.print("\n")
        console.print("=" * 60)
        console.print("[bold]📄 RAPPORT FINAL[/bold]")
        console.print("=" * 60 + "\n")

        try:
            # Rich peut rendre le Markdown directement dans le terminal
            console.print(Markdown(rapport))
        except Exception:
            # Fallback : affichage texte brut si Rich échoue
            console.print(rapport)
    else:
        console.print(
            "\n[bold red]❌ Aucun rapport final généré.[/bold red]\n"
            "Vérifiez les logs pour diagnostiquer l'erreur."
        )


# ============================================================
# FONCTION PRINCIPALE DU WORKFLOW
# ============================================================

def lancer_veille(
        query: str,
        afficher_details: bool = False,
        sauvegarder: bool = True
) -> AgentState:
    """
    Orchestre l'exécution complète du workflow de veille.

    Lance les 3 agents dans l'ordre défini par le graph LangGraph :
      1. SearchAgent  → collecte les articles web
      2. SummaryAgent → synthétise les résultats
      3. ValidationAgent → vérifie la fiabilité

    Affiche une progression en temps réel dans le terminal
    et retourne l'état final complet pour traitement ultérieur.

    Args:
        query: Le sujet de recherche saisi par l'utilisateur
        afficher_details: Si True, affiche le graphe Mermaid du workflow
        sauvegarder: Si True, sauvegarde le rapport dans un fichier

    Returns:
        L'AgentState final contenant tous les résultats du workflow

    Raises:
        SystemExit: Si Ollama n'est pas disponible ou si le workflow échoue
    """
    # ----------------------------------------
    # AFFICHAGE DE LA BANNIÈRE DE DÉMARRAGE
    # ----------------------------------------
    afficher_banniere(query)

    # ----------------------------------------
    # VÉRIFICATION D'OLLAMA
    # ----------------------------------------
    console.print("[bold]🔧 Vérification des prérequis...[/bold]")

    if not verifier_ollama():
        console.print(
            Panel(
                f"[bold red]❌ Ollama n'est pas disponible ![/bold red]\n\n"
                f"Pour résoudre ce problème :\n"
                f"1. Démarrez Ollama : [bold]ollama serve[/bold]\n"
                f"2. Vérifiez que le modèle est installé : "
                f"[bold]ollama pull {ollama_config.model}[/bold]\n"
                f"3. Vérifiez l'URL configurée : "
                f"[bold]{ollama_config.base_url}[/bold]",
                border_style="red"
            )
        )
        sys.exit(1)

    console.print("[green]✅ Ollama opérationnel[/green]\n")

    # ----------------------------------------
    # CONSTRUCTION DU WORKFLOW LANGGRAPH
    # ----------------------------------------
    console.print("[bold]🔨 Construction du workflow LangGraph...[/bold]")

    try:
        workflow = construire_workflow()

        if afficher_details:
            console.print("\n[bold]📐 Graphe du workflow (Mermaid) :[/bold]")
            visualiser_workflow(workflow)

        console.print("[green]✅ Workflow construit avec succès[/green]\n")

    except Exception as e:
        console.print(
            f"[bold red]❌ Erreur lors de la construction du workflow :[/bold red] "
            f"{str(e)}"
        )
        logger.error(
            f"Erreur construction workflow : {str(e)}",
            exc_info=True
        )
        sys.exit(1)

    # ----------------------------------------
    # PRÉPARATION DE L'ÉTAT INITIAL
    # ----------------------------------------
    etat_initial = creer_etat_initial(query)
    etat_final: Optional[AgentState] = None

    # ----------------------------------------
    # EXÉCUTION DU WORKFLOW AVEC PROGRESSION
    # ----------------------------------------
    console.print("[bold]🚀 Lancement du workflow...[/bold]\n")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False
    ) as progress:

        tache = progress.add_task(
            "🔍 Agent 1 — SearchAgent : Recherche web en cours...",
            total=None
        )

        try:
            # Parcourir le flux LangGraph en temps réel
            # stream_mode="updates" retourne les mises à jour de chaque nœud
            for event in workflow.stream(
                    etat_initial,
                    stream_mode="updates"
            ):
                for nom_noeud, mise_a_jour in event.items():

                    # --- Fin du SearchAgent ---
                    if nom_noeud == "search_agent":
                        nb_articles = len(
                            mise_a_jour.get("raw_results", [])
                        )
                        progress.update(
                            tache,
                            description=(
                                f"✅ Agent 1 terminé ({nb_articles} articles) | "
                                f"📝 Agent 2 — SummaryAgent : Synthèse en cours..."
                            )
                        )
                        logger.info(
                            f"SearchAgent terminé : {nb_articles} articles"
                        )

                    # --- Fin du SummaryAgent ---
                    elif nom_noeud == "summary_agent":
                        resume = mise_a_jour.get("summary", "")
                        longueur = len(resume)
                        progress.update(
                            tache,
                            description=(
                                f"✅ Agent 2 terminé ({longueur} caractères) | "
                                f"🛡️ Agent 3 — ValidationAgent : Validation en cours..."
                            )
                        )
                        logger.info(
                            f"SummaryAgent terminé : résumé de {longueur} caractères"
                        )

                    # --- Fin du ValidationAgent ---
                    elif nom_noeud == "validation_agent":
                        validation = mise_a_jour.get(
                            "validation_result", {}
                        ) or {}
                        score = validation.get("score_fiabilite", 0)
                        decision = validation.get("decision", "?")
                        progress.update(
                            tache,
                            description=(
                                f"✅ Agent 3 terminé "
                                f"(Score : {score}/100 — {decision})"
                            )
                        )
                        logger.info(
                            f"ValidationAgent terminé : "
                            f"score={score}, décision={decision}"
                        )

                    # --- Nœud d'erreur ---
                    elif nom_noeud == "error_handler":
                        progress.update(
                            tache,
                            description="❌ Erreur détectée — Arrêt du workflow"
                        )
                        logger.warning("Nœud error_handler atteint")

            # Récupérer l'état final complet après le stream
            etat_final = workflow.invoke(
                etat_initial
            ) if etat_final is None else etat_final

        except Exception as e:
            erreur_msg = f"❌ Erreur pendant l'exécution du workflow : {str(e)}"
            progress.update(tache, description=erreur_msg)
            logger.error(erreur_msg, exc_info=True)
            console.print(f"\n[bold red]{erreur_msg}[/bold red]")
            sys.exit(1)

    # ----------------------------------------
    # RÉCUPÉRATION DE L'ÉTAT FINAL COMPLET
    # ----------------------------------------
    # Lancer une seconde fois en mode non-stream pour obtenir
    # l'état final complet avec tous les champs mis à jour
    try:
        console.print(
            "\n[dim]🔄 Consolidation de l'état final...[/dim]"
        )
        etat_final = workflow.invoke(etat_initial)

    except Exception as e:
        logger.error(
            f"Erreur lors de la récupération de l'état final : {str(e)}",
            exc_info=True
        )
        console.print(
            f"[bold red]❌ Impossible de récupérer l'état final : "
            f"{str(e)}[/bold red]"
        )
        sys.exit(1)

    # ----------------------------------------
    # SAUVEGARDE DU RAPPORT
    # ----------------------------------------
    chemin_sauvegarde = None

    if sauvegarder and app_config.save_results:
        rapport = etat_final.get("final_report", "")
        if rapport:
            console.print(
                "\n[bold]💾 Sauvegarde du rapport...[/bold]"
            )
            chemin_sauvegarde = sauvegarder_rapport(
                rapport=rapport,
                query=query,
                state=etat_final
            )
            if chemin_sauvegarde:
                console.print(
                    f"[green]✅ Rapport sauvegardé : "
                    f"{chemin_sauvegarde}[/green]"
                )
            else:
                console.print(
                    "[yellow]⚠️ Échec de la sauvegarde du rapport[/yellow]"
                )

    # ----------------------------------------
    # AFFICHAGE DES RÉSULTATS FINAUX
    # ----------------------------------------
    afficher_resultats_finaux(etat_final, chemin_sauvegarde)

    return etat_final


# ============================================================
# POINT D'ENTRÉE CLI
# ============================================================

def main():
    """
    Point d'entrée en ligne de commande du système de veille.

    Parse les arguments CLI, valide la requête et lance le workflow.
    Gère les interruptions clavier et les erreurs fatales de façon propre.

    Usage depuis le terminal :
        python main.py "intelligence artificielle 2024"
        python main.py "crypto monnaies" --details
        python main.py "résultats sport" --no-save
        python main.py --help
    """
    # ----------------------------------------
    # CONFIGURATION DES ARGUMENTS CLI
    # ----------------------------------------
    parser = argparse.ArgumentParser(
        prog="veille",
        description=(
            "🤖 Système de Veille Informationnelle Multi-Agents\n"
            "Orchestré par LangGraph avec Ollama (LLM local)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python main.py "intelligence artificielle 2024"
  python main.py "crypto monnaies actualités" --details
  python main.py "résultats ligue 1 football" --no-save
  python main.py "cybersécurité menaces récentes" --details --no-save

Codes de sortie :
  0  → Workflow réussi, score de fiabilité >= 60
  1  → Workflow échoué ou score de fiabilité < 60
  130 → Interrompu par l'utilisateur (Ctrl+C)
        """
    )

    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Sujet de recherche (entre guillemets si plusieurs mots)"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        default=False,
        help="Afficher le graphe Mermaid du workflow au démarrage"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        dest="no_save",
        help="Ne pas sauvegarder le rapport dans un fichier Markdown"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Système de Veille v1.0.0"
    )

    args = parser.parse_args()

    # ----------------------------------------
    # VALIDATION DE LA REQUÊTE
    # ----------------------------------------
    query = args.query

    # Mode interactif si aucune requête fournie en argument
    if not query:
        console.print(
            Panel(
                "[bold blue]🤖 Système de Veille Informationnelle[/bold blue]\n\n"
                "Aucun sujet de recherche spécifié en argument.",
                border_style="blue"
            )
        )
        try:
            query = console.input(
                "[bold cyan]📌 Entrez votre sujet de recherche : [/bold cyan]"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print(
                "\n[yellow]⚠️ Saisie annulée.[/yellow]"
            )
            sys.exit(130)

    # Vérifier que la requête n'est pas vide
    if not query or not query.strip():
        console.print(
            "[bold red]❌ Le sujet de recherche ne peut pas être vide.[/bold red]\n"
            "Usage : python main.py \"votre sujet de recherche\""
        )
        sys.exit(1)

    # Vérifier la longueur minimale de la requête
    if len(query.strip()) < 3:
        console.print(
            "[bold red]❌ Le sujet de recherche est trop court "
            "(minimum 3 caractères).[/bold red]"
        )
        sys.exit(1)

    # ----------------------------------------
    # LANCEMENT DU WORKFLOW
    # ----------------------------------------
    try:
        etat_final = lancer_veille(
            query=query,
            afficher_details=args.details,
            sauvegarder=not args.no_save
        )

        # ----------------------------------------
        # CODE DE SORTIE SELON LE SCORE
        # ----------------------------------------
        validation = etat_final.get("validation_result", {}) or {}
        score = validation.get("score_fiabilite", 0)

        if score >= 60:
            logger.info(
                f"✅ Workflow terminé avec succès — Score : {score}/100"
            )
            sys.exit(0)
        else:
            logger.warning(
                f"⚠️ Workflow terminé avec un score faible — Score : {score}/100"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        console.print(
            "\n\n[bold yellow]⚠️  Workflow interrompu par l'utilisateur "
            "(Ctrl+C)[/bold yellow]"
        )
        logger.info("Workflow interrompu par l'utilisateur")
        sys.exit(130)

    except SystemExit as e:
        # Laisser passer les sys.exit() déjà gérés
        raise

    except Exception as e:
        console.print(
            f"\n[bold red]❌ Erreur fatale inattendue : {str(e)}[/bold red]\n"
            f"Consultez les logs pour plus de détails : "
            f"{app_config.log_file}"
        )
        logger.error(
            f"Erreur fatale dans main() : {str(e)}",
            exc_info=True
        )
        sys.exit(1)


# ============================================================
# LANCEMENT DU SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
