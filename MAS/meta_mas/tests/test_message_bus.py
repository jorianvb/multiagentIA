"""Tests pour MessageBus."""
import asyncio
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import Message, MessageType
from core.message_bus import MessageBus


@pytest.fixture
def bus():
    b = MessageBus(max_history=50)
    b.register_agent("alpha")
    b.register_agent("beta")
    b.register_agent("gamma")
    return b


def make_msg(sender: str, receiver: str, content: str = "hello") -> Message:
    return Message(
        sender=sender,
        receiver=receiver,
        type=MessageType.REQUEST,
        content=content,
    )


# ── Enregistrement ────────────────────────────────────────────────────────────

def test_register_agent(bus):
    """Vérifie l'enregistrement d'un agent."""
    bus.register_agent("delta")
    stats = bus.get_stats()
    assert "delta" in stats["registered_agents"]


def test_unregister_agent(bus):
    """Vérifie le désenregistrement d'un agent."""
    bus.unregister_agent("gamma")
    stats = bus.get_stats()
    assert "gamma" not in stats["registered_agents"]


# ── Unicast ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unicast_delivery(bus):
    """Vérifie qu'un message unicast arrive chez le bon agent."""
    msg = make_msg("alpha", "beta", "unicast test")
    await bus.publish(msg)

    received = await bus.receive("beta", timeout=1.0)
    assert received is not None
    assert received.content == "unicast test"
    assert received.sender == "alpha"


@pytest.mark.asyncio
async def test_unicast_wrong_receiver_raises(bus):
    """Vérifie qu'un destinataire inconnu lève une exception."""
    msg = make_msg("alpha", "unknown_agent")
    with pytest.raises(KeyError):
        await bus.publish(msg)


# ── Broadcast ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_broadcast_reaches_all_except_sender(bus):
    """Vérifie que le broadcast atteint tous les agents sauf l'expéditeur."""
    msg = Message(
        sender="alpha",
        receiver="broadcast",
        type=MessageType.BROADCAST,
        content="broadcast test",
    )
    await bus.publish(msg)

    # beta et gamma doivent recevoir
    beta_msg = await bus.receive("beta", timeout=1.0)
    gamma_msg = await bus.receive("gamma", timeout=1.0)
    assert beta_msg is not None
    assert gamma_msg is not None

    # alpha ne doit PAS recevoir son propre broadcast
    alpha_msg = await bus.receive("alpha", timeout=0.1)
    assert alpha_msg is None


# ── Multicast ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multicast_delivery(bus):
    """Vérifie qu'un message multicast arrive chez les bons agents."""
    msg = make_msg("alpha", "beta,gamma", "multicast test")
    await bus.publish(msg)

    beta_msg = await bus.receive("beta", timeout=1.0)
    gamma_msg = await bus.receive("gamma", timeout=1.0)
    assert beta_msg is not None
    assert gamma_msg is not None


# ── Historique ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_records_messages(bus):
    """Vérifie que l'historique enregistre les messages."""
    await bus.publish(make_msg("alpha", "beta", "msg1"))
    await bus.publish(make_msg("beta", "alpha", "msg2"))

    history = bus.get_history()
    assert len(history) == 2


@pytest.mark.asyncio
async def test_history_filter_by_agent(bus):
    """Vérifie le filtrage de l'historique par agent."""
    await bus.publish(make_msg("alpha", "beta", "msg_ab"))
    await bus.publish(make_msg("gamma", "beta", "msg_gb"))

    alpha_history = bus.get_history(agent_name="alpha")
    assert len(alpha_history) == 1
    assert alpha_history[0].content == "msg_ab"


@pytest.mark.asyncio
async def test_history_max_size(bus):
    """Vérifie que l'historique respecte la taille maximale."""
    for i in range(60):  # max_history=50
        await bus.publish(make_msg("alpha", "beta", f"msg_{i}"))
        # Vider la queue pour éviter le blocage
        try:
            await asyncio.wait_for(bus._queues["beta"].get(), timeout=0.01)
        except asyncio.TimeoutError:
            pass

    assert len(bus.get_history()) <= 50


# ── Timeout ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_receive_timeout_returns_none(bus):
    """Vérifie que receive() retourne None si aucun message."""
    msg = await bus.receive("alpha", timeout=0.1)
    assert msg is None


# ── Statistiques ──────────────────────────────────────────────────────────────

def test_get_stats(bus):
    """Vérifie les statistiques du bus."""
    stats = bus.get_stats()
    assert "registered_agents" in stats
    assert "history_size" in stats
    assert "queue_sizes" in stats
    assert "alpha" in stats["registered_agents"]


# ── Middleware ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_middleware_transforms_message(bus):
    """Vérifie qu'un middleware peut transformer les messages."""
    async def uppercase_middleware(msg: Message) -> Message:
        return msg.model_copy(update={"content": msg.content.upper()})

    bus.add_middleware(uppercase_middleware)
    await bus.publish(make_msg("alpha", "beta", "hello world"))

    received = await bus.receive("beta", timeout=1.0)
    assert received is not None
    assert received.content == "HELLO WORLD"


@pytest.mark.asyncio
async def test_middleware_can_block_message(bus):
    """Vérifie qu'un middleware retournant None bloque le message."""
    async def block_all(msg: Message):
        return None  # Bloquer tous les messages

    bus.add_middleware(block_all)
    await bus.publish(make_msg("alpha", "beta", "blocked"))

    received = await bus.receive("beta", timeout=0.1)
    assert received is None

