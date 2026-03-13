# main.py (version complète)
# Point d'entrée principal du système multi-agent fiction

import uuid
import sys
import json
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  'rich' non installé. Interface basique activée.")

from graph import story_graph
from state import StoryState
from memory.narrative_bible import NarrativeBible
from memory.short_term import ShortTermMemory
from utils.output_formatter import save_response_to_file
from utils.scoring import rank_ideas

console = Console() if RICH_AVAILABLE else None


def _check_ollama_connection(model_name: str) -> bool:
    """
    Vérifie qu'Ollama est disponible et que le modèle est chargé.
    Retourne True si OK, False sinon.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage

        llm = ChatOllama(model=model_name, temperature=0)
        # Test minimal : une question simple
        llm.invoke([HumanMessage(content="Réponds juste 'ok'")])
        print(f"   ✅ Ollama connecté avec le modèle '{model_name}'")
        return True

    except Exception as e:
        print(f"   ❌ Erreur Ollama : {str(e)}")
        return False


def run_story_system(
        story_text: str,
        user_request: str,
        project_name: str = "mon_projet",
        model_name: str = "llama3.1",
        bible: NarrativeBible = None,
        short_memory: ShortTermMemory = None
) -> tuple[str, StoryState]:
    """
    Point d'entrée principal du système multi-agent.

    Args:
        story_text    : Le texte déjà écrit par l'auteur (SOURCE DE VÉRITÉ)
        user_request  : La demande spécifique de l'auteur
        project_name  : Nom du projet pour l'organisation des fichiers
        model_name    : Modèle Ollama à utiliser
        bible         : Instance de la bible narrative (optionnel)
        short_memory  : Instance de la mémoire court terme (optionnel)

    Returns:
        Tuple (réponse_finale: str, état_final: StoryState)
    """

    # ── Vérification Ollama ──────────────────────────────────────────────
    print("\n🔌 Vérification de la connexion Ollama...")
    if not _check_ollama_connection(model_name):
        error_msg = (
            f"❌ Impossible de se connecter à Ollama avec '{model_name}'.\n"
            f"   1. Lancez Ollama      : ollama serve\n"
            f"   2. Chargez le modèle  : ollama pull {model_name}\n"
            f"   3. Vérifiez le port   : http://localhost:11434"
        )
        print(error_msg)
        return error_msg, {}

    # ── Génération de l'ID de session ────────────────────────────────────
    session_id = str(uuid.uuid4())
    timestamp  = datetime.now().isoformat()

    print(f"\n🚀 Démarrage de la session : {session_id[:8]}...")
    print(f"   Projet  : {project_name}")
    print(f"   Modèle  : {model_name}")
    print(f"   Texte   : {len(story_text)} caractères")

    # ── Contexte long terme depuis la bible ──────────────────────────────
    bible_context = ""
    if bible:
        bible_context = bible.get_context_for_agents()
        print(f"   📚 Bible narrative chargée (version {bible.bible['meta']['version']})")

    # ── Contexte court terme ─────────────────────────────────────────────
    recent_context = ""
    if short_memory:
        recent_context = short_memory.get_recent_context(n=3)

    # ── Enrichissement du texte avec la mémoire ──────────────────────────
    enriched_story = story_text
    if bible_context or recent_context:
        enriched_story = (
            f"{bible_context}\n\n"
            f"{recent_context}\n\n"
            f"=== TEXTE ACTUEL (SOURCE DE VÉRITÉ) ===\n"
            f"{story_text}"
        )

    # ── Construction de l'état initial ───────────────────────────────────
    initial_state: StoryState = {
        "existing_story"    : enriched_story,
        "user_request"      : user_request,
        "model_name"        : model_name,
        "characters_summary": {},
        "plots_summary"     : {},
        "story_context"     : "",
        "consistency_report": {},
        "story_ideas"       : [],
        "final_response"    : "",
        "iteration_count"   : 0,
        "session_id"        : session_id,
        "timestamp"         : timestamp,
        "errors"            : []
    }

    # ── Exécution du graphe ───────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("▶  LANCEMENT DU PIPELINE MULTI-AGENT")
    print("═" * 50)

    try:
        final_state = story_graph.invoke(initial_state)
    except Exception as e:
        error_msg = f"❌ Erreur critique lors de l'exécution du graphe : {str(e)}"
        print(error_msg)
        return error_msg, initial_state

    # ── Mise à jour de la bible narrative ────────────────────────────────
    if bible and final_state.get("characters_summary"):
        print("\n📚 Mise à jour de la bible narrative...")
        bible.update_from_analysis(
            characters  = final_state["characters_summary"],
            plots       = final_state["plots_summary"],
            context     = final_state["story_context"],
            session_id  = session_id
        )

    # ── Mise à jour de la mémoire court terme ────────────────────────────
    if short_memory:
        short_memory.add_exchange(
            story_text       = story_text,
            user_request     = user_request,
            analysis_summary = final_state.get("story_context", "")
        )

    # ── Sauvegarde de la réponse finale ──────────────────────────────────
    final_response = final_state.get("final_response", "Aucune réponse générée.")

    saved_path = save_response_to_file(
        response     = final_response,
        project_name = project_name,
        session_id   = session_id
    )

    # ── Affichage des erreurs non bloquantes ─────────────────────────────
    if final_state.get("errors"):
        print("\n⚠️  Avertissements non bloquants :")
        for err in final_state["errors"]:
            print(f"   • {err}")

    print("\n" + "═" * 50)
    print("✅ PIPELINE TERMINÉ")
    print(f"   Fichier sauvegardé : {saved_path}")
    print("═" * 50)

    return final_response, final_state


# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE CLI
# ═══════════════════════════════════════════════════════════════════════════

def cli_interface():
    """
    Interface en ligne de commande pour interagir avec le système.
    Gère la session complète avec mémoire persistante.
    """

    # ── Bannière ─────────────────────────────────────────────────────────
    banner = """
╔══════════════════════════════════════════════════════╗
║       SYSTÈME MULTI-AGENT ÉCRITURE DE FICTION        ║
║              LangGraph + Ollama                      ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Configuration du projet ──────────────────────────────────────────
    project_name = input("📁 Nom du projet (défaut: mon_projet) : ").strip()
    if not project_name:
        project_name = "mon_projet"

    model_name = input("🤖 Modèle Ollama (défaut: llama3.2) : ").strip()
    if not model_name:
        model_name = "llama3.2"

    # ── Initialisation de la mémoire ─────────────────────────────────────
    bible        = NarrativeBible(project_name=project_name)
    short_memory = ShortTermMemory(max_exchanges=10)

    print(f"\n✅ Projet '{project_name}' initialisé")
    print(f"   Répertoire de sortie : ./output/{project_name}/")

    # ── Boucle principale de session ─────────────────────────────────────
    while True:
        print("\n" + "─" * 50)
        print("OPTIONS :")
        print("  [1] Analyser / Continuer l'histoire")
        print("  [2] Charger un fichier texte")
        print("  [3] Voir la bible narrative actuelle")
        print("  [4] Changer de modèle Ollama")
        print("  [q] Quitter")
        print("─" * 50)

        choice = input("\nVotre choix : ").strip().lower()

        # ── Quitter ──────────────────────────────────────────────────────
        if choice == "q":
            print("\n👋 À bientôt ! Bonne écriture !")
            sys.exit(0)

        # ── Voir la bible ────────────────────────────────────────────────
        elif choice == "3":
            context = bible.get_context_for_agents()
            print("\n" + context)
            continue

        # ── Changer de modèle ────────────────────────────────────────────
        elif choice == "4":
            model_name = input("Nouveau modèle Ollama : ").strip()
            print(f"✅ Modèle changé pour : {model_name}")
            continue

        # ── Charger un fichier ───────────────────────────────────────────
        elif choice == "2":
            filepath = input("Chemin du fichier texte : ").strip()
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    story_text = f.read()
                print(f"✅ Fichier chargé : {len(story_text)} caractères")
            except FileNotFoundError:
                print(f"❌ Fichier introuvable : {filepath}")
                continue
            except Exception as e:
                print(f"❌ Erreur lecture fichier : {e}")
                continue

        # ── Analyse de l'histoire ────────────────────────────────────────
        elif choice == "1":
            print("\n📝 Collez votre texte (terminez par une ligne contenant uniquement 'FIN') :")
            lines = []
            while True:
                line = input()
                if line.strip() == "FIN":
                    break
                lines.append(line)
            story_text = "\n".join(lines)

            if not story_text.strip():
                print("❌ Aucun texte fourni.")
                continue

        else:
            print("❌ Option invalide.")
            continue

        # ── Demande spécifique de l'auteur ───────────────────────────────
        print("\n💬 Quelle est votre demande spécifique ?")
        print("   (ex: 'développer le personnage de X', 'trouver une suite dramatique')")
        user_request = input("   Votre demande : ").strip()
        if not user_request:
            user_request = "Analyse générale et propositions de suite"

        # ── Lancement du système ─────────────────────────────────────────
        response, state = run_story_system(
            story_text   = story_text,
            user_request = user_request,
            project_name = project_name,
            model_name   = model_name,
            bible        = bible,
            short_memory = short_memory
        )

        # ── Affichage de la réponse ──────────────────────────────────────
        print("\n" + "═" * 60)
        print(response)
        print("═" * 60)

        # ── Option de sauvegarde supplémentaire ──────────────────────────
        save_extra = input("\n💾 Sauvegarder aussi en JSON ? (o/N) : ").strip().lower()
        if save_extra == "o":
            _save_state_as_json(state, project_name)


def _save_state_as_json(state: StoryState, project_name: str) -> None:
    """Sauvegarde l'état complet en JSON pour analyse ou débogage."""
    output_path = Path("./output") / project_name / "sessions"
    output_path.mkdir(parents=True, exist_ok=True)

    date_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_path / f"{date_str}_state.json"

    # On retire les champs trop longs pour le JSON de debug
    state_to_save = {
        k: v for k, v in state.items()
        if k not in ["existing_story"]  # Le texte source est déjà sauvegardé ailleurs
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_to_save, f, ensure_ascii=False, indent=2)

    print(f"   ✅ État JSON sauvegardé : {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli_interface()
