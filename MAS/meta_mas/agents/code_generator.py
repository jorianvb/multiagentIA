"""
Agent Générateur de Code — génère le code Python de chaque agent du MAS.

Utilise Jinja2 + LLM pour produire du code complet, documenté et fonctionnel.
Itère avec le Validateur en cas d'erreur (max 3 tentatives).
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger

from config.settings import Settings
from core.base_agent import BaseAgent, Message, MessageType, AgentStatus
from core.memory import SharedMemory
from utils.code_parser import CodeParser


class CodeGeneratorAgent(BaseAgent):
    """
    Agent spécialisé dans la génération de code Python pour les MAS.

    Fonctionnement :
    1. Reçoit la spécification d'un agent (nom, rôle, system_prompt, etc.)
    2. Charge le template Jinja2 correspondant
    3. Demande au LLM de générer l'implémentation complète
    4. Valide la syntaxe avec ast.parse()
    5. Retourne le code Python prêt à être déployé
    """

    SYSTEM_PROMPT = """Tu es un Expert Développeur Python spécialisé dans les systèmes multi-agents asynchrones.
Tu génères du code Python 3.11+ complet, fonctionnel et bien documenté.

RÈGLES ABSOLUES :
1. Tout le code doit être async-first (utiliser asyncio).
2. Utilise loguru pour le logging (from loguru import logger).
3. Toutes les classes héritent de BaseAgent.
4. Chaque méthode doit avoir une docstring.
5. Utilise des type hints partout.
6. Le code doit être PEP8 compliant.
7. NE PAS laisser de TODO ou de pass sans implémentation.
8. Gestion d'erreurs avec try/except sur toutes les opérations critiques.
9. Réponds avec UN SEUL bloc de code Python dans ```python ... ```.

IMPORTS STANDARD À UTILISER :
```python
import asyncio
import json
from typing import Any, Dict, List, Optional
from loguru import logger
from core.base_agent import BaseAgent, Message, MessageType, AgentStatus
```

STRUCTURE D'UN AGENT :
```python
class MonAgent(BaseAgent):
    SYSTEM_PROMPT = "..."
    
    def __init__(self, model="mistral", message_bus=None, memory=None):
        super().__init__(name="...", role="...", model=model,
                         system_prompt=self.SYSTEM_PROMPT, message_bus=message_bus)
        self.memory_store = memory
    
    async def act(self, message: Message) -> Message:
        # Traitement principal
        result = await self.think({"task": message.content})
        return Message(sender=self.name, receiver=message.sender,
                       type=MessageType.RESPONSE, content=result)
```"""

    def __init__(
        self,
        model: str = "mistral",
        message_bus=None,
        memory: SharedMemory = None,
        settings: Settings = None,
    ):
        super().__init__(
            name="CodeGenerator",
            role="Code Generator",
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            message_bus=message_bus,
            timeout=180,
        )
        self.memory_store = memory
        self.settings = settings or Settings()

        # Charger les templates Jinja2
        templates_dir = Path(__file__).parent.parent / "templates"
        if templates_dir.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self._jinja_env = None
            logger.warning(f"[{self.name}] Dossier templates introuvable: {templates_dir}")

    async def act(self, message: Message) -> Message:
        """
        Génère le code pour un ou plusieurs agents.

        Le contenu du message doit être un JSON avec :
        - "agents": liste des specs d'agents (analyse + architecture)
        - "analysis": résultat de l'Analyste
        - "architecture": résultat de l'Architecte

        Returns:
            Message contenant un JSON {filename: code_content}
        """
        logger.info(f"[{self.name}] 💻 Génération du code en cours...")
        self.status = AgentStatus.ACTING

        try:
            request_data = json.loads(message.content)
            analysis = request_data.get("analysis", {})
            architecture = request_data.get("architecture", {})

            generated_files = await self._generate_all(analysis, architecture)

            if self.memory_store:
                for filename, code in generated_files.items():
                    await self.memory_store.add_generated_file(filename, code)
                await self.memory_store.set_pipeline_stage("CODE_GENERATION_DONE")

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=json.dumps(generated_files, ensure_ascii=False),
                metadata={"files_count": len(generated_files)},
            )

        except Exception as e:
            logger.error(f"[{self.name}] Erreur génération: {e}")
            if self.memory_store:
                await self.memory_store.add_error(str(e), context="CodeGenerator.act")
            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.ERROR,
                content=f"Erreur génération: {str(e)}",
            )
        finally:
            self.status = AgentStatus.IDLE

    async def _generate_all(
        self,
        analysis: Dict[str, Any],
        architecture: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Génère tous les fichiers du MAS cible.

        Returns:
            Dict {chemin_relatif: contenu_code}
        """
        generated: Dict[str, str] = {}
        agents = analysis.get("agents", [])
        agent_details_map = {
            d.get("name"): d
            for d in architecture.get("agent_details", [])
            if isinstance(d, dict)
        }

        logger.info(f"[{self.name}] Génération de {len(agents)} agents...")

        # 1. Générer base_agent.py pour le MAS cible
        base_agent_code = self._generate_base_agent()
        generated["agents/base_agent.py"] = base_agent_code

        # 2. Générer chaque agent
        for agent_spec in agents:
            agent_name = agent_spec.get("name", "agent")
            detail = agent_details_map.get(agent_name, {})
            merged = {**agent_spec, **detail}

            logger.info(f"[{self.name}] Génération de '{agent_name}'...")
            code = await self._generate_agent_with_retry(merged, analysis)
            filename = f"agents/{agent_name}.py"
            generated[filename] = code

        # 3. Générer agents/__init__.py
        generated["agents/__init__.py"] = self._generate_agents_init(agents)

        # 4. Générer core/message_bus.py (version simplifiée pour le MAS généré)
        generated["core/message_bus.py"] = self._generate_target_message_bus()

        # 5. Générer core/orchestrator.py (runner principal)
        generated["core/orchestrator.py"] = self._generate_target_orchestrator(agents, analysis)

        # 6. Générer config/settings.py
        generated["config/settings.py"] = self._generate_config(analysis)

        # 7. Générer main.py
        generated["main.py"] = self._generate_main(analysis, agents)

        # 8. Générer les tests
        for agent_spec in agents:
            agent_name = agent_spec.get("name", "agent")
            generated[f"tests/test_{agent_name}.py"] = self._generate_test(agent_spec)

        generated["tests/__init__.py"] = '"""Tests du MAS généré."""\n'
        generated["config/__init__.py"] = '"""Configuration du MAS généré."""\n'
        generated["agents/__init__.py"] = self._generate_agents_init(agents)

        logger.info(f"[{self.name}] ✅ {len(generated)} fichiers générés")
        return generated

    async def _generate_agent_with_retry(
        self,
        agent_spec: Dict[str, Any],
        analysis: Dict[str, Any],
        max_retries: int = 3,
    ) -> str:
        """
        Génère le code d'un agent avec jusqu'à max_retries tentatives.

        Args:
            agent_spec: Spécification complète de l'agent.
            analysis: Contexte global de l'analyse.
            max_retries: Nombre maximum de tentatives.

        Returns:
            Code Python valide.
        """
        errors_history: List[str] = []

        for attempt in range(1, max_retries + 1):
            code = await self._generate_single_agent(agent_spec, errors_history)

            is_valid, errors = CodeParser.validate_python_syntax(code)
            if is_valid:
                logger.info(
                    f"[{self.name}] ✅ '{agent_spec['name']}' valide "
                    f"(tentative {attempt})"
                )
                return code

            logger.warning(
                f"[{self.name}] ⚠️  '{agent_spec['name']}' invalide "
                f"(tentative {attempt}): {errors}"
            )
            errors_history.extend(errors)

        # Après max_retries, retourner un template minimal fonctionnel
        logger.error(
            f"[{self.name}] ❌ Génération échouée pour '{agent_spec['name']}' "
            f"après {max_retries} tentatives, utilisation du template fallback"
        )
        return self._fallback_agent_code(agent_spec)

    async def _generate_single_agent(
        self,
        agent_spec: Dict[str, Any],
        errors_history: List[str],
    ) -> str:
        """Génère le code d'un seul agent via LLM."""
        name = agent_spec.get("name", "agent")
        class_name = agent_spec.get("class_name", f"{name.title()}Agent")
        role = agent_spec.get("role", "agent spécialisé")
        responsibilities = agent_spec.get("responsibilities", [])
        system_prompt = agent_spec.get("system_prompt", f"Tu es l'agent {name}. Ton rôle: {role}.")
        needs_llm = agent_spec.get("needs_llm", True)

        resp_str = "\n".join(f"    - {r}" for r in responsibilities)
        error_hint = ""
        if errors_history:
            error_hint = (
                f"\n\nERREURS À CORRIGER (tentatives précédentes) :\n"
                + "\n".join(f"  - {e}" for e in errors_history[-3:])
            )

        context = {
            "AGENT_NAME": name,
            "CLASS_NAME": class_name,
            "ROLE": role,
            "RESPONSABILITÉS": resp_str,
            "SYSTEM_PROMPT": system_prompt,
            "NEEDS_LLM": str(needs_llm),
            "INSTRUCTION": (
                f"Génère le code Python complet pour la classe {class_name} "
                f"qui hérite de BaseAgent. "
                f"Implémente la méthode async act(self, message: Message) -> Message "
                f"de manière complète et fonctionnelle selon le rôle décrit. "
                f"Inclus tous les imports nécessaires. "
                f"Réponds avec UN SEUL bloc ```python ... ```."
                + error_hint
            ),
        }

        raw = await self.think(context)
        code = CodeParser.extract_python_code(raw)
        if code is None:
            code = raw  # Fallback: essayer le texte brut

        return code or self._fallback_agent_code(agent_spec)

    # ------------------------------------------------------------------
    # Générateurs de fichiers statiques
    # ------------------------------------------------------------------

    def _generate_base_agent(self) -> str:
        """Génère un base_agent.py autonome pour le MAS cible."""
        return '''\
"""
BaseAgent pour le MAS généré.
Classe abstraite dont héritent tous les agents.
"""
import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    ACTING = "ACTING"
    WAITING = "WAITING"
    ERROR = "ERROR"


class MessageType(str, Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    BROADCAST = "BROADCAST"
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    type: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Classe abstraite de base pour tous les agents du MAS."""

    def __init__(self, name: str, role: str, model: str,
                 system_prompt: str, message_bus=None, timeout: int = 120):
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt
        self.message_bus = message_bus
        self.timeout = timeout
        self.status = AgentStatus.IDLE
        self.memory: List[Dict[str, str]] = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        logger.info(f"[{self.name}] Agent initialisé | rôle={self.role}")

    async def think(self, context: dict) -> str:
        """Appel LLM via Ollama."""
        import httpx
        messages = [{"role": "system", "content": self.system_prompt}]
        for exchange in self.memory[-8:]:
            messages.append(exchange)
        context_str = "\\n".join(f"{k}: {v}" for k, v in context.items())
        messages.append({"role": "user", "content": context_str})

        self.status = AgentStatus.THINKING
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "http://localhost:11434/api/chat",
                    json={"model": self.model, "messages": messages, "stream": False},
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                self.memory.append({"role": "user", "content": context_str})
                self.memory.append({"role": "assistant", "content": content})
                return content
        finally:
            self.status = AgentStatus.IDLE

    @abstractmethod
    async def act(self, message: "Message") -> "Message":
        """Méthode d\'action principale."""
        ...

    async def communicate(self, target: str, content: str,
                          msg_type: MessageType = MessageType.REQUEST) -> None:
        """Envoie un message via le bus."""
        if self.message_bus is None:
            raise RuntimeError(f"[{self.name}] Pas de bus de messages connecté")
        msg = Message(sender=self.name, receiver=target,
                      type=msg_type, content=content)
        await self.message_bus.publish(msg)

    def add_to_inbox(self, message: "Message") -> None:
        self._inbox.put_nowait(message)

    def get_status(self) -> Dict[str, Any]:
        return {"name": self.name, "role": self.role,
                "status": self.status.value, "memory_size": len(self.memory)}
'''

    def _generate_agents_init(self, agents: List[Dict]) -> str:
        """Génère le __init__.py du dossier agents."""
        imports = []
        exports = []
        for agent in agents:
            name = agent.get("name", "agent")
            class_name = agent.get("class_name", f"{name.title()}Agent")
            module = name
            imports.append(f"from agents.{module} import {class_name}")
            exports.append(f'    "{class_name}"')

        lines = ['"""Agents du MAS généré."""']
        lines.extend(imports)
        lines.append("")
        lines.append("__all__ = [")
        lines.extend([e + "," for e in exports])
        lines.append("]")
        return "\n".join(lines) + "\n"

    def _generate_target_message_bus(self) -> str:
        """Génère un message_bus.py simplifié pour le MAS cible."""
        return '''\
"""Bus de messages async pour le MAS généré."""
import asyncio
from typing import Dict, List, Optional
from loguru import logger
from agents.base_agent import Message


class MessageBus:
    """Bus de messages centralisé (unicast + broadcast)."""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._history: List[Message] = []

    def register(self, agent_name: str) -> None:
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()

    async def publish(self, message: Message) -> None:
        self._history.append(message)
        if message.receiver == "broadcast":
            for name, q in self._queues.items():
                if name != message.sender:
                    await q.put(message)
        elif message.receiver in self._queues:
            await self._queues[message.receiver].put(message)
        else:
            logger.error(f"Bus: destinataire inconnu \'{message.receiver}\'")

    async def receive(self, agent_name: str,
                      timeout: Optional[float] = None) -> Optional[Message]:
        if agent_name not in self._queues:
            raise KeyError(f"Agent \'{agent_name}\' non enregistré")
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._queues[agent_name].get(), timeout=timeout)
            return await self._queues[agent_name].get()
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> Dict:
        return {"agents": list(self._queues.keys()),
                "history": len(self._history)}
'''

    def _generate_target_orchestrator(
        self,
        agents: List[Dict],
        analysis: Dict[str, Any],
    ) -> str:
        """Génère le runner principal pour le MAS cible."""
        system_name = analysis.get("system_name", "GeneratedMAS")
        agent_imports = "\n".join(
            f"    from agents.{a['name']} import {a.get('class_name', a['name'].title() + 'Agent')}"
            for a in agents
        )
        agent_inits = "\n".join(
            f'    agents["{a["name"]}"] = {a.get("class_name", a["name"].title() + "Agent")}('
            f'model=model, message_bus=bus)'
            for a in agents
        )
        agent_registers = "\n".join(
            f'    bus.register("{a["name"]}")'
            for a in agents
        )

        return f'''\
"""
Orchestrateur principal du {system_name}.
Lance et coordonne tous les agents.
"""
import asyncio
from typing import Dict, Any
from loguru import logger
from core.message_bus import MessageBus
from agents.base_agent import Message, MessageType


async def run_mas(user_input: str, model: str = "mistral") -> Dict[str, Any]:
    """
    Lance le MAS avec l\'entrée utilisateur.
    
    Args:
        user_input: Requête de l\'utilisateur.
        model: Modèle Ollama à utiliser.
    
    Returns:
        Résultat final du MAS.
    """
    bus = MessageBus()
    agents: Dict[str, Any] = {{}}

    # Initialisation des agents
    logger.info("Initialisation du {system_name}...")
{agent_imports}
{agent_inits}

    # Enregistrement dans le bus
{agent_registers}

    # Enregistrement des buses dans les agents
    for agent in agents.values():
        agent.message_bus = bus

    # Démarrage : envoyer la requête à l\'orchestrateur
    orchestrator_name = next(
        (name for name in agents if "orchestrat" in name.lower()), 
        list(agents.keys())[0]
    )
    
    logger.info(f"Envoi de la requête à {{orchestrator_name}}...")
    
    start_msg = Message(
        sender="user",
        receiver=orchestrator_name,
        type=MessageType.REQUEST,
        content=user_input,
    )
    
    await bus.publish(start_msg)
    
    # Traitement par l\'agent principal
    orchestrator = agents[orchestrator_name]
    response = await orchestrator.act(start_msg)
    
    logger.info(f"Réponse reçue de {{orchestrator_name}}")
    
    return {{
        "success": True,
        "response": response.content,
        "agent": orchestrator_name,
        "stats": bus.get_stats(),
    }}
'''

    def _generate_config(self, analysis: Dict[str, Any]) -> str:
        """Génère la configuration pour le MAS cible."""
        system_name = analysis.get("system_name", "GeneratedMAS")
        return f'''\
"""Configuration du {system_name}."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
'''

    def _generate_main(
        self,
        analysis: Dict[str, Any],
        agents: List[Dict],
    ) -> str:
        """Génère le main.py pour le MAS cible."""
        system_name = analysis.get("system_name", "GeneratedMAS")
        description = analysis.get("description", "Système multi-agents généré par Meta-MAS")
        agent_names = [a.get("name", "?") for a in agents]

        return f'''\
"""
{system_name} — point d\'entrée principal.

{description}

Agents disponibles : {", ".join(agent_names)}

Généré automatiquement par Meta-MAS.
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from loguru import logger

from core.orchestrator import run_mas
from config.settings import LOG_LEVEL


console = Console()


def setup_logging():
    logger.remove()
    import sys
    logger.add(sys.stdout, level=LOG_LEVEL, colorize=True,
               format="<green>{{time:HH:mm:ss}}</green> | <level>{{level: <8}}</level> | <level>{{message}}</level>")


async def main():
    setup_logging()
    
    console.print(Panel(
        "[bold cyan]{system_name}[/bold cyan]\\n[white]{description}[/white]",
        title="🤖 MAS Généré par Meta-MAS",
        border_style="cyan",
    ))
    
    console.print("\\n[dim]Agents disponibles: {", ".join(agent_names)}[/dim]\\n")
    
    while True:
        user_input = Prompt.ask("[bold cyan]Votre requête[/bold cyan] (ou \'exit\' pour quitter)")
        
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Au revoir ![/yellow]")
            break
        
        if not user_input.strip():
            continue
        
        console.print("\\n[dim]Traitement en cours...[/dim]")
        
        try:
            result = await run_mas(user_input)
            
            if result.get("success"):
                console.print(Panel(
                    result["response"],
                    title=f"[bold green]Réponse de {{result.get(\'agent\', \'MAS\')}}[/bold green]",
                    border_style="green",
                ))
            else:
                console.print(f"[red]Erreur: {{result.get(\'error\', \'Inconnue\')}}[/red]")
                
        except Exception as e:
            logger.exception("Erreur pendant le traitement")
            console.print(f"[red]Erreur: {{e}}[/red]")
        
        console.print()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
'''

    def _generate_test(self, agent_spec: Dict[str, Any]) -> str:
        """Génère un fichier de test pour un agent."""
        name = agent_spec.get("name", "agent")
        class_name = agent_spec.get("class_name", f"{name.title()}Agent")

        return f'''\
"""Tests pour {class_name}."""
import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.{name} import {class_name}
from agents.base_agent import Message, MessageType


@pytest.fixture
def agent():
    """Fixture: instance de {class_name}."""
    return {class_name}(model="mistral")


@pytest.mark.asyncio
async def test_{name}_initialization(agent):
    """Vérifie l\'initialisation de l\'agent."""
    assert agent.name is not None
    assert agent.role is not None
    assert agent.model == "mistral"
    assert agent.status.value == "IDLE"


@pytest.mark.asyncio
async def test_{name}_has_system_prompt(agent):
    """Vérifie que le system_prompt est défini."""
    assert hasattr(agent, "SYSTEM_PROMPT") or agent.system_prompt
    assert len(agent.system_prompt) > 10


@pytest.mark.asyncio
async def test_{name}_get_status(agent):
    """Vérifie la méthode get_status."""
    status = agent.get_status()
    assert "name" in status
    assert "role" in status
    assert "status" in status


@pytest.mark.asyncio
async def test_{name}_memory_management(agent):
    """Vérifie la gestion de la mémoire."""
    assert isinstance(agent.memory, list)
    agent.memory.append({{"role": "user", "content": "test"}})
    assert len(agent.memory) == 1
    agent.clear_memory()
    assert len(agent.memory) == 0
'''

    def _fallback_agent_code(self, agent_spec: Dict[str, Any]) -> str:
        """Code de fallback minimal valide si la génération LLM échoue."""
        name = agent_spec.get("name", "agent")
        class_name = agent_spec.get("class_name", f"{name.title()}Agent")
        role = agent_spec.get("role", "Agent spécialisé")
        responsibilities = agent_spec.get("responsibilities", [])
        resp_str = "; ".join(responsibilities[:3])
        system_prompt = agent_spec.get(
            "system_prompt",
            f"Tu es l'agent {name}. Rôle: {role}. {resp_str}. Réponds de manière utile et précise."
        )
        # Escape quotes in system_prompt for embedding in code
        system_prompt = system_prompt.replace('"""', "'''")

        return f'''\
"""
{class_name} — {role}

Généré automatiquement par Meta-MAS (mode fallback).
"""
import asyncio
from typing import Any, Dict, Optional
from loguru import logger
from agents.base_agent import BaseAgent, Message, MessageType, AgentStatus


class {class_name}(BaseAgent):
    """{role}

    Responsabilités : {resp_str}
    """

    SYSTEM_PROMPT = """{system_prompt}"""

    def __init__(
        self,
        name: str = "{name}",
        model: str = "mistral",
        message_bus=None,
        memory=None,
    ):
        super().__init__(
            name=name,
            role="{role}",
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            message_bus=message_bus,
            timeout=120,
        )
        self.memory_store = memory

    async def act(self, message: Message) -> Message:
        """Traite un message entrant et retourne une réponse.

        Args:
            message: Message à traiter.

        Returns:
            Message de réponse.
        """
        logger.info(f"[{{self.name}}] Traitement du message de {{message.sender}}")
        self.status = AgentStatus.ACTING

        try:
            context = {{
                "tâche": message.content,
                "expéditeur": message.sender,
                "contexte": str(message.metadata),
            }}

            response_content = await self.think(context)

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=response_content,
                metadata={{"original_id": message.id}},
            )

        except Exception as e:
            logger.error(f"[{{self.name}}] Erreur: {{e}}")
            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.ERROR,
                content=f"Erreur dans {{self.name}}: {{str(e)}}",
            )
        finally:
            self.status = AgentStatus.IDLE
'''

