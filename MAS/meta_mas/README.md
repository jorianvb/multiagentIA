# 🤖 Meta-MAS — Générateur de Systèmes Multi-Agents

> Un système multi-agents **méta** : il génère et déploie d'autres systèmes multi-agents.

---

## Architecture

### Choix : Custom asyncio (vs LangGraph)

| Critère | Custom asyncio ✅ | LangGraph |
|---------|-----------------|-----------|
| Transparence | Totale | Abstraite |
| Overhead | Minimal | Framework lourd |
| Contrôle | Complet | Contraint |
| Debugging | Natif Python | Via LangSmith |
| Ollama natif | ✅ Direct | Via LangChain |

### Graphe du pipeline

```
👤 Utilisateur (description + dossier cible)
        │
        ▼
🎯 Orchestrateur (coordonne tout)
        │
        ├─① 🔍 Analyste      → JSON: {agents, rôles, workflow}
        │
        ├─② 🏗️  Architecte    → JSON: {topologie, system_prompts}
        │
        ├─③ 💻 Générateur    ←→ ✅ Validateur (boucle max 3×)
        │          └→ Dict: {filename → code_python}
        │
        └─④ 🚀 Déployeur     → {dossier_cible}/ complet
```

### Structure du projet

```
meta_mas/
├── main.py                       # Point d'entrée CLI (--graph, --demo)
├── pyproject.toml
├── README.md
├── .env.example
├── config/
│   ├── settings.py               # Config Pydantic (Ollama, timeouts…)
│   └── agents_config.yaml        # Définition des agents méta
├── core/
│   ├── base_agent.py             # Classe abstraite BaseAgent
│   ├── message_bus.py            # Bus async (unicast/broadcast/multicast)
│   ├── agent_registry.py         # Registre singleton des agents
│   └── memory.py                 # SharedMemory (asyncio.Lock)
├── agents/
│   ├── orchestrator.py           # Chef du pipeline
│   ├── analyst.py                # Identifie les agents nécessaires
│   ├── architect.py              # Conçoit la topologie
│   ├── code_generator.py         # Génère le code Python
│   ├── validator.py              # Valide syntaxe + structure
│   └── deployer.py               # Écrit les fichiers sur disque
├── templates/                    # Templates Jinja2
│   ├── agent_template.py.jinja2
│   ├── pyproject_template.toml.jinja2
│   ├── readme_template.md.jinja2
│   └── config_template.yaml.jinja2
├── utils/
│   ├── ollama_client.py          # Wrapper httpx async (retry + backoff)
│   ├── code_parser.py            # Extraction et validation du code LLM
│   ├── file_manager.py           # Gestion fichiers / rollback
│   ├── graph.py                  # Visualisation Rich du pipeline
│   └── logger.py                 # Logger loguru + Rich
└── tests/
    ├── test_base_agent.py
    ├── test_message_bus.py
    └── test_code_generator.py
```

---

## Prérequis

- [ ] Python 3.11+
- [ ] [`uv`](https://docs.astral.sh/uv/) installé
- [ ] [Ollama](https://ollama.ai) installé et en cours d'exécution
- [ ] Au moins un modèle téléchargé (`mistral` recommandé)

---

## Installation

### 1. Créer / accéder au projet

```bash
cd C:/Projet/multiagentIA/mas/meta_mas
```

### 2. Installer uv (si absent)

**Windows (PowerShell) :**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS :**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Installer les dépendances

```bash
uv sync
```

### 4. Configurer l'environnement

```bash
cp .env.example .env
# Éditer .env si besoin
```

### 5. Lancer Ollama et télécharger un modèle

```bash
# Démarrer Ollama (tourne en arrière-plan)
ollama serve

# Dans un autre terminal, télécharger le modèle
ollama pull mistral

# Vérifier les modèles disponibles
ollama list
```

---

## Utilisation

### Mode interactif (normal)

```bash
uv run python main.py
```

L'interface vous demande :
1. **Description** du MAS à créer (texte libre)
2. **Dossier cible** (créé automatiquement)
3. **Modèle Ollama** (défaut : `mistral`)
4. **Options avancées** (retries, timeout, verbose)

### Afficher uniquement le graphe

```bash
uv run python main.py --graph
```

### Mode démo (sans LLM, pour tester l'UI)

```bash
uv run python main.py --demo
```

---

## Exemple d'utilisation complet

```
$ uv run python main.py

  ╔══════════════════════════════════════════════════╗
  ║              🤖  Meta-MAS  v0.1.0               ║
  ║    Multi-Agent System Generator & Orchestrator  ║
  ╚══════════════════════════════════════════════════╝

Afficher le graphe du pipeline avant de commencer ? [y/n]: y

[Graphe affiché]

1. Décrivez le MAS à créer :
   Description: Un MAS de service client avec un agent FAQ, un agent
                de tri et un agent d'escalade

2. Dossier cible: ./customer_service_mas

3. Modèle Ollama [mistral]: mistral

✅ Ollama disponible avec le modèle 'mistral'

Lancer la génération ? [Y/n]: Y

─────────────────────────────────────────────────
🔄 ÉTAPE 1/5 — Analyse des besoins
─────────────────────────────────────────────────
  ▶ Analyst analyse la demande utilisateur...
✅ Analyse terminée — 4 agents identifiés

🔄 ÉTAPE 2/5 — Conception architecturale
  ▶ Architect conçoit l'architecture...
✅ Architecture conçue — topologie: hub_and_spoke

🔄 ÉTAPE 3/5 — Génération du code
  ▶ CodeGenerator génère le code pour 4 agents...
✅ Code généré — 16 fichiers

🔄 ÉTAPE 4/5 — Validation
  ▶ Validator valide 12 fichiers Python...
✅ Validation réussie — score moyen: 88/100

🔄 ÉTAPE 5/5 — Déploiement
  ▶ Deployer déploie dans ./customer_service_mas...
✅ Déploiement terminé — 18 fichiers créés

╔══════════════════════════════════════╗
║  🎉 CustomerServiceMAS généré !     ║
║  Agents : 4    Fichiers : 18        ║
║  cd ./customer_service_mas          ║
║  uv sync && uv run python main.py   ║
╚══════════════════════════════════════╝
```

---

## Utiliser le MAS généré

```bash
cd ./customer_service_mas
uv sync
uv run python main.py
```

---

## Tests

```bash
uv run pytest tests/ -v

# Avec couverture
uv run pytest tests/ -v --tb=short
```

---

## Personnalisation

### Changer le modèle par défaut

Dans `.env` :
```
OLLAMA_DEFAULT_MODEL=llama3
```

### Ajouter un middleware au MessageBus

```python
from core.message_bus import MessageBus

bus = MessageBus()

async def log_middleware(message):
    print(f"Message: {message.sender} → {message.receiver}")
    return message

bus.add_middleware(log_middleware)
```

### Étendre BaseAgent

```python
from core.base_agent import BaseAgent, Message, MessageType

class MonAgent(BaseAgent):
    SYSTEM_PROMPT = "Tu es mon agent spécialisé..."

    async def act(self, message: Message) -> Message:
        result = await self.think({"tâche": message.content})
        return Message(sender=self.name, receiver=message.sender,
                       type=MessageType.RESPONSE, content=result)
```

---

## Troubleshooting

| Problème | Solution |
|----------|----------|
| `Ollama not found` | Lancer `ollama serve` |
| `Model not found` | `ollama pull mistral` |
| `Timeout` | Augmenter `OLLAMA_TIMEOUT` dans `.env` |
| `SyntaxError` dans le code généré | Le Validateur réessaie automatiquement (max 3×) |
| Code généré vide | Vérifier la connexion Ollama et le modèle |
| `uv not found` | Installer uv (voir Installation) |

---

## Stack technique

| Composant | Librairie |
|-----------|-----------|
| Runtime | Python 3.11+ / asyncio |
| LLM | Ollama (local) via httpx |
| CLI | Rich |
| Templates | Jinja2 |
| Validation | Pydantic v2 |
| Logging | Loguru |
| Tests | pytest + pytest-asyncio |
| Package manager | uv |

---

*Meta-MAS v0.1.0 — 2026*

