"""Tests pour CodeGeneratorAgent et CodeParser."""
import ast
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.code_parser import CodeParser, CodeBlock
from agents.code_generator import CodeGeneratorAgent
from core.base_agent import Message, MessageType
from core.memory import SharedMemory


# ──────────────────────────────────────────────────────────────────────────────
# Tests CodeParser
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeParser:

    def test_extract_python_block(self):
        """Extrait un bloc ```python ... ```."""
        text = "Voici le code :\n```python\nprint('hello')\n```"
        code = CodeParser.extract_python_code(text)
        assert code is not None
        assert "print" in code

    def test_extract_multiple_blocks(self):
        """Extrait plusieurs blocs."""
        text = "```python\nx = 1\n```\n\n```python\ny = 2\n```"
        blocks = CodeParser.extract_code_blocks(text)
        assert len(blocks) == 2
        assert all(b.language == "python" for b in blocks)

    def test_extract_no_block_returns_none(self):
        """Retourne None si aucun bloc Python."""
        text = "Voici une réponse en prose sans code."
        code = CodeParser.extract_python_code(text)
        assert code is None

    def test_validate_valid_syntax(self):
        """Valide une syntaxe correcte."""
        code = "def hello():\n    return 'world'\n"
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid is True
        assert errors == []

    def test_validate_invalid_syntax(self):
        """Détecte une syntaxe invalide."""
        code = "def hello(\n    return 'world'\n"
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid is False
        assert len(errors) > 0

    def test_check_imports(self):
        """Liste les imports."""
        code = "import asyncio\nfrom loguru import logger\n"
        imports = CodeParser.check_imports(code)
        assert "asyncio" in imports
        assert "loguru" in imports

    def test_extract_class_names(self):
        """Extrait les noms de classes."""
        code = "class Foo:\n    pass\nclass Bar(Foo):\n    pass\n"
        classes = CodeParser.extract_class_names(code)
        assert "Foo" in classes
        assert "Bar" in classes

    def test_extract_function_names(self):
        """Extrait les noms de fonctions."""
        code = "async def act(self):\n    pass\ndef think(self):\n    pass\n"
        funcs = CodeParser.extract_function_names(code)
        assert "act" in funcs
        assert "think" in funcs

    def test_extract_files_from_structured_response(self):
        """Extrait plusieurs fichiers d'une réponse structurée."""
        text = (
            "# agents/agent_a.py\n"
            "```python\nclass AgentA:\n    pass\n```\n"
            "# agents/agent_b.py\n"
            "```python\nclass AgentB:\n    pass\n```"
        )
        files = CodeParser.extract_files_from_response(text)
        assert "agents/agent_a.py" in files or "agents/agent_b.py" in files

    def test_extract_json_from_response(self):
        """Extrait un bloc JSON."""
        text = 'Voici le résultat :\n```json\n{"key": "value"}\n```'
        json_str = CodeParser.extract_json_from_response(text)
        assert json_str is not None
        assert '"key"' in json_str

    def test_clean_code_removes_markdown_markers(self):
        """Nettoie les marqueurs Markdown résiduels."""
        code = "```python\nclass Foo:\n    pass\n```"
        cleaned = CodeParser.clean_code(code)
        assert "```" not in cleaned
        assert "class Foo" in cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Tests CodeGeneratorAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestCodeGeneratorAgent:

    @pytest.fixture
    def agent(self):
        return CodeGeneratorAgent(model="mistral", memory=SharedMemory())

    def test_initialization(self, agent):
        """Vérifie l'initialisation."""
        assert agent.name == "CodeGenerator"
        assert agent.model == "mistral"
        assert agent.timeout == 180

    def test_generate_base_agent(self, agent):
        """Vérifie que le base_agent généré est du Python valide."""
        code = agent._generate_base_agent()
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid, f"base_agent invalide: {errors}"
        assert "BaseAgent" in code
        assert "async def think" in code

    def test_generate_agents_init(self, agent):
        """Vérifie le __init__.py généré."""
        agents = [
            {"name": "foo", "class_name": "FooAgent"},
            {"name": "bar", "class_name": "BarAgent"},
        ]
        code = agent._generate_agents_init(agents)
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid, f"__init__.py invalide: {errors}"
        assert "FooAgent" in code
        assert "BarAgent" in code

    def test_generate_target_message_bus(self, agent):
        """Vérifie que le message_bus généré est valide."""
        code = agent._generate_target_message_bus()
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid, f"message_bus invalide: {errors}"
        assert "MessageBus" in code
        assert "async def publish" in code

    def test_fallback_agent_code_is_valid(self, agent):
        """Vérifie que le code fallback est toujours du Python valide."""
        spec = {
            "name": "test_agent",
            "class_name": "TestAgent",
            "role": "Agent de test",
            "responsibilities": ["Tester les choses"],
            "system_prompt": "Tu es un agent de test.",
        }
        code = agent._fallback_agent_code(spec)
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid, f"Fallback invalide: {errors}"
        assert "TestAgent" in code
        assert "async def act" in code

    def test_generate_main_is_valid(self, agent):
        """Vérifie que le main.py généré est du Python valide."""
        analysis = {
            "system_name": "TestMAS",
            "description": "Un MAS de test",
        }
        agents_list = [{"name": "agent_a", "class_name": "AgentA"}]
        code = agent._generate_main(analysis, agents_list)
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid, f"main.py invalide: {errors}"

    def test_generate_test_file_is_valid(self, agent):
        """Vérifie que les fichiers de test générés sont valides."""
        spec = {
            "name": "my_agent",
            "class_name": "MyAgent",
        }
        code = agent._generate_test(spec)
        valid, errors = CodeParser.validate_python_syntax(code)
        assert valid, f"test invalide: {errors}"
        assert "MyAgent" in code
        assert "pytest" in code

    @pytest.mark.asyncio
    async def test_act_returns_response(self, agent):
        """Vérifie que act() retourne un Message de type RESPONSE."""
        import json
        analysis = {
            "system_name": "TestMAS",
            "description": "Test",
            "agents": [
                {
                    "name": "worker",
                    "class_name": "WorkerAgent",
                    "role": "Travailleur",
                    "responsibilities": ["Travailler"],
                    "needs_llm": False,
                }
            ],
            "communication_pattern": "hub_and_spoke",
            "workflow_description": "Simple",
        }
        architecture = {"topology": "hub_and_spoke", "agent_details": []}

        payload = json.dumps({"analysis": analysis, "architecture": architecture})

        msg = Message(
            sender="Orchestrator",
            receiver="CodeGenerator",
            type=MessageType.REQUEST,
            content=payload,
        )

        # Mocker l'appel LLM pour éviter un appel réseau
        mock_code = '''\
from agents.base_agent import BaseAgent, Message, MessageType, AgentStatus
from loguru import logger


class WorkerAgent(BaseAgent):
    """Agent travailleur."""

    SYSTEM_PROMPT = "Tu es un travailleur."

    def __init__(self, name="worker", model="mistral", message_bus=None, memory=None):
        super().__init__(name=name, role="Travailleur", model=model,
                         system_prompt=self.SYSTEM_PROMPT, message_bus=message_bus)

    async def act(self, message: Message) -> Message:
        """Traite le message."""
        try:
            result = await self.think({"task": message.content})
            return Message(sender=self.name, receiver=message.sender,
                           type=MessageType.RESPONSE, content=result)
        except Exception as e:
            logger.error(f"[{self.name}] Erreur: {e}")
            return Message(sender=self.name, receiver=message.sender,
                           type=MessageType.ERROR, content=str(e))
'''
        with patch.object(agent, "_generate_single_agent", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_code
            response = await agent.act(msg)

        assert response.type == MessageType.RESPONSE
        files = json.loads(response.content)
        assert isinstance(files, dict)
        assert "agents/worker.py" in files

