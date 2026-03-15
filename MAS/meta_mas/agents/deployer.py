"""
Agent Déployeur — crée la structure complète du MAS généré sur le disque.

Utilise Jinja2 pour les templates (README, pyproject.toml, etc.)
et FileManager pour l'écriture des fichiers.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from core.base_agent import BaseAgent, Message, MessageType, AgentStatus
from core.memory import SharedMemory
from utils.file_manager import FileManager


class DeployerAgent(BaseAgent):
    """
    Agent spécialisé dans le déploiement du MAS généré.

    Actions :
    1. Crée la structure de répertoires dans le dossier cible
    2. Écrit tous les fichiers Python générés
    3. Génère pyproject.toml depuis le template Jinja2
    4. Génère README.md documenté
    5. Génère .env.example
    6. Retourne un rapport de déploiement avec l'arborescence
    """

    SYSTEM_PROMPT = """Tu es un Expert en DevOps et déploiement de systèmes Python.
Tu crées et organises les fichiers d'un projet Python de manière professionnelle.
Tu génères une documentation claire et un README complet."""

    def __init__(
        self,
        model: str = "mistral",
        message_bus=None,
        memory: SharedMemory = None,
    ):
        super().__init__(
            name="Deployer",
            role="System Deployer",
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            message_bus=message_bus,
            timeout=60,
        )
        self.memory_store = memory

        templates_dir = Path(__file__).parent.parent / "templates"
        if templates_dir.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self._jinja_env = None

    async def act(self, message: Message) -> Message:
        """
        Déploie le MAS dans le dossier cible.

        Le message.content est un JSON contenant :
        - "target_directory": chemin cible
        - "generated_files": Dict {filename: content}
        - "analysis": résultat de l'Analyste
        - "architecture": résultat de l'Architecte

        Returns:
            Message de rapport de déploiement.
        """
        logger.info(f"[{self.name}] 🚀 Déploiement en cours...")
        self.status = AgentStatus.ACTING

        try:
            data = json.loads(message.content)
            target_dir = data.get("target_directory", "./generated_mas")
            generated_files = data.get("generated_files", {})
            analysis = data.get("analysis", {})
            architecture = data.get("architecture", {})

            report = await self._deploy(
                target_dir, generated_files, analysis, architecture
            )

            if self.memory_store:
                await self.memory_store.set("deployment_status", report)
                await self.memory_store.set_pipeline_stage("DEPLOYMENT_DONE")

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=json.dumps(report, ensure_ascii=False, indent=2),
                metadata={
                    "success": report.get("success", False),
                    "files_count": report.get("files_count", 0),
                    "target_directory": target_dir,
                },
            )

        except Exception as e:
            logger.error(f"[{self.name}] Erreur déploiement: {e}")
            if self.memory_store:
                await self.memory_store.add_error(str(e), context="Deployer.act")
            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.ERROR,
                content=f"Erreur déploiement: {str(e)}",
            )
        finally:
            self.status = AgentStatus.IDLE

    async def _deploy(
        self,
        target_dir: str,
        generated_files: Dict[str, str],
        analysis: Dict[str, Any],
        architecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Effectue le déploiement complet."""
        fm = FileManager(target_dir)
        fm.ensure_base()

        # Créer la structure standard
        fm.create_standard_structure()

        # Écrire les fichiers générés par le CodeGenerator
        written: List[str] = []
        for filename, content in generated_files.items():
            fm.write_file(filename, content)
            written.append(filename)
            logger.debug(f"[{self.name}] Écrit: {filename}")

        # Générer les fichiers de projet
        system_name = analysis.get("system_name", "GeneratedMAS")
        agents = analysis.get("agents", [])

        # pyproject.toml
        pyproject_content = self._render_pyproject(system_name, agents)
        fm.write_file("pyproject.toml", pyproject_content)
        written.append("pyproject.toml")

        # README.md
        readme_content = self._render_readme(system_name, analysis, architecture, agents)
        fm.write_file("README.md", readme_content)
        written.append("README.md")

        # .env.example
        env_content = self._render_env_example()
        fm.write_file(".env.example", env_content)
        written.append(".env.example")

        # __init__.py pour les modules
        for module in ("core", "config"):
            init_path = f"{module}/__init__.py"
            if not fm.file_exists(init_path):
                fm.write_file(init_path, f'"""{module.title()} module."""\n')
                written.append(init_path)

        # Arborescence finale
        tree = fm.get_tree()
        logger.info(f"[{self.name}] ✅ Déploiement terminé — {len(written)} fichiers")

        return {
            "success": True,
            "target_directory": str(fm.base_path),
            "files_count": len(written),
            "files_written": written,
            "tree": tree,
            "system_name": system_name,
            "agents_count": len(agents),
        }

    # ------------------------------------------------------------------
    # Rendus Jinja2 / templates inline
    # ------------------------------------------------------------------

    def _render_pyproject(self, system_name: str, agents: List[Dict]) -> str:
        """Génère pyproject.toml pour le MAS cible."""
        snake_name = system_name.lower().replace(" ", "-")
        agent_names = [a.get("name", "?") for a in agents]
        agents_list = ", ".join(agent_names)

        if self._jinja_env:
            try:
                template = self._jinja_env.get_template("pyproject_template.toml.jinja2")
                return template.render(
                    system_name=system_name,
                    snake_name=snake_name,
                    agents_list=agents_list,
                    date=datetime.now().strftime("%Y-%m-%d"),
                )
            except Exception:
                pass  # Fallback sur inline

        return f"""\
[project]
name = "{snake_name}"
version = "0.1.0"
description = "{system_name} — généré par Meta-MAS"
requires-python = ">=3.11"

# Agents: {agents_list}

dependencies = [
    "ollama>=0.3",
    "httpx>=0.27",
    "pydantic>=2.0",
    "loguru>=0.7",
    "rich>=13.0",
    "python-dotenv>=1.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
"""

    def _render_readme(
        self,
        system_name: str,
        analysis: Dict[str, Any],
        architecture: Dict[str, Any],
        agents: List[Dict],
    ) -> str:
        """Génère README.md pour le MAS cible."""
        if self._jinja_env:
            try:
                template = self._jinja_env.get_template("readme_template.md.jinja2")
                return template.render(
                    system_name=system_name,
                    analysis=analysis,
                    architecture=architecture,
                    agents=agents,
                    date=datetime.now().strftime("%Y-%m-%d"),
                )
            except Exception:
                pass  # Fallback sur inline

        description = analysis.get("description", "Système multi-agents généré par Meta-MAS")
        pattern = analysis.get("communication_pattern", "hub_and_spoke")
        topology = architecture.get("topology", pattern)
        workflow = analysis.get("workflow_description", "Pipeline séquentiel")

        agents_section = ""
        for a in agents:
            name = a.get("name", "?")
            role = a.get("role", "?")
            responsibilities = a.get("responsibilities", [])
            resp_md = "\n".join(f"  - {r}" for r in responsibilities)
            agents_section += f"\n### `{name}` — {role}\n{resp_md}\n"

        return f"""\
# {system_name}

> {description}

Généré automatiquement par **Meta-MAS** le {datetime.now().strftime("%Y-%m-%d %H:%M")}.

## Architecture

- **Topologie** : `{topology}`
- **Pattern** : `{pattern}`
- **Workflow** : {workflow}

## Agents

{agents_section}

## Installation

```bash
# Installer les dépendances avec uv
uv sync

# Ou avec pip
pip install -r requirements.txt
```

## Configuration

Copier `.env.example` en `.env` et configurer les variables :

```bash
cp .env.example .env
```

Assurez-vous qu'Ollama est en marche :

```bash
ollama serve
ollama pull mistral
```

## Utilisation

```bash
uv run python main.py
```

## Tests

```bash
uv run pytest tests/ -v
```

## Structure

```
{system_name.lower()}/
├── main.py           # Point d'entrée
├── config/
│   └── settings.py  # Configuration
├── agents/
│   ├── base_agent.py
{"".join(f"│   ├── {a.get('name', '?')}.py" + chr(10) for a in agents)}└── core/
│   ├── orchestrator.py
│   └── message_bus.py
└── tests/
{"".join(f"    ├── test_{a.get('name', '?')}.py" + chr(10) for a in agents)}```

---
*Généré par [Meta-MAS](https://github.com/meta-mas) — {datetime.now().strftime("%Y")}*
"""

    def _render_env_example(self) -> str:
        """Génère .env.example."""
        return """\
# Configuration Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Logging
LOG_LEVEL=INFO
"""

