"""Tests pour BaseAgent et les primitives core."""
import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import (
    AgentStatus,
    BaseAgent,
    Message,
    MessageType,
)


# ── Implémentation minimale pour les tests ────────────────────────────────────

class ConcreteAgent(BaseAgent):
    """Implémentation minimale de BaseAgent pour les tests."""

    SYSTEM_PROMPT = "Tu es un agent de test."

    def __init__(self, **kwargs):
        super().__init__(
            name="TestAgent",
            role="Tester",
            model="mistral",
            system_prompt=self.SYSTEM_PROMPT,
            **kwargs,
        )

    async def act(self, message: Message) -> Message:
        return Message(
            sender=self.name,
            receiver=message.sender,
            type=MessageType.RESPONSE,
            content=f"Echo: {message.content}",
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return ConcreteAgent()


@pytest.fixture
def sample_message():
    return Message(
        sender="user",
        receiver="TestAgent",
        type=MessageType.REQUEST,
        content="Bonjour, agent !",
    )


# ── Tests d'initialisation ────────────────────────────────────────────────────

def test_agent_initialization(agent):
    """Vérifie que l'agent s'initialise correctement."""
    assert agent.name == "TestAgent"
    assert agent.role == "Tester"
    assert agent.model == "mistral"
    assert agent.status == AgentStatus.IDLE
    assert isinstance(agent.memory, list)
    assert len(agent.memory) == 0


def test_agent_repr(agent):
    """Vérifie la représentation textuelle."""
    repr_str = repr(agent)
    assert "TestAgent" in repr_str
    assert "Tester" in repr_str


# ── Tests Message ─────────────────────────────────────────────────────────────

def test_message_creation():
    """Vérifie la création d'un message."""
    msg = Message(
        sender="agent_a",
        receiver="agent_b",
        type=MessageType.REQUEST,
        content="Hello",
    )
    assert msg.sender == "agent_a"
    assert msg.receiver == "agent_b"
    assert msg.type == MessageType.REQUEST
    assert msg.id is not None
    assert len(msg.id) > 0


def test_message_has_timestamp():
    """Vérifie que le timestamp est auto-généré."""
    msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="x")
    assert msg.timestamp is not None


def test_message_metadata_default():
    """Vérifie que les métadonnées sont un dict vide par défaut."""
    msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="x")
    assert msg.metadata == {}


# ── Tests act() ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_act_returns_message(agent, sample_message):
    """Vérifie que act() retourne un Message valide."""
    response = await agent.act(sample_message)
    assert isinstance(response, Message)
    assert response.sender == agent.name
    assert response.receiver == sample_message.sender
    assert response.type == MessageType.RESPONSE


@pytest.mark.asyncio
async def test_agent_act_echo(agent, sample_message):
    """Vérifie le contenu de la réponse echo."""
    response = await agent.act(sample_message)
    assert "Echo" in response.content
    assert sample_message.content in response.content


# ── Tests mémoire ─────────────────────────────────────────────────────────────

def test_clear_memory(agent):
    """Vérifie que clear_memory() vide l'historique."""
    agent.memory = [{"role": "user", "content": "test"}]
    assert len(agent.memory) == 1
    agent.clear_memory()
    assert len(agent.memory) == 0


def test_get_status(agent):
    """Vérifie le dict de statut."""
    status = agent.get_status()
    assert status["name"] == "TestAgent"
    assert status["role"] == "Tester"
    assert status["status"] == "IDLE"
    assert "memory_size" in status
    assert "inbox_size" in status


# ── Tests inbox ───────────────────────────────────────────────────────────────

def test_add_to_inbox(agent, sample_message):
    """Vérifie l'ajout d'un message dans l'inbox."""
    agent.add_to_inbox(sample_message)
    assert agent._inbox.qsize() == 1


@pytest.mark.asyncio
async def test_receive_from_inbox(agent, sample_message):
    """Vérifie la réception d'un message depuis l'inbox."""
    agent.add_to_inbox(sample_message)
    msg = await agent.receive(timeout=1.0)
    assert msg is not None
    assert msg.content == sample_message.content


@pytest.mark.asyncio
async def test_receive_timeout(agent):
    """Vérifie que receive() retourne None après timeout."""
    msg = await agent.receive(timeout=0.1)
    assert msg is None


# ── Tests think() (avec mock LLM) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_think_calls_ollama(agent):
    """Vérifie que think() appelle OllamaClient.chat()."""
    mock_response = "Réponse simulée du LLM"

    with patch("utils.ollama_client.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        with patch("utils.ollama_client.OllamaClient.close", new_callable=AsyncMock):
            mock_chat.return_value = mock_response
            result = await agent.think({"tâche": "Test"})

    assert result == mock_response
    assert len(agent.memory) == 2  # user + assistant


@pytest.mark.asyncio
async def test_think_updates_memory(agent):
    """Vérifie que think() met à jour la mémoire."""
    with patch("utils.ollama_client.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        with patch("utils.ollama_client.OllamaClient.close", new_callable=AsyncMock):
            mock_chat.return_value = "Réponse"
            await agent.think({"question": "42 ?"})

    assert len(agent.memory) == 2
    assert agent.memory[0]["role"] == "user"
    assert agent.memory[1]["role"] == "assistant"
    assert agent.memory[1]["content"] == "Réponse"

