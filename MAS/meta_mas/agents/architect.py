"""
Agent Architecte — conçoit la structure et les protocoles du MAS.

Prend les résultats de l'Analyste et produit un plan architectural détaillé.
"""
import json
from typing import Any, Dict

from loguru import logger

from core.base_agent import BaseAgent, Message, MessageType, AgentStatus
from core.memory import SharedMemory
from utils.code_parser import CodeParser


class ArchitectAgent(BaseAgent):
    """
    Agent spécialisé dans la conception architecturale de MAS.

    À partir de la spécification des agents (fournie par l'Analyste),
    conçoit :
    - La topologie de communication (hub-and-spoke, peer-to-peer, etc.)
    - Le format des messages entre agents
    - Les patterns de coordination
    - La stratégie de gestion d'erreurs
    - La structure des fichiers du MAS

    Output JSON :
    {
      "topology": "hub_and_spoke",
      "message_format": {...},
      "coordination_pattern": "...",
      "error_strategy": "...",
      "agent_details": [
        {
          "name": "...",
          "system_prompt": "...",
          "dependencies": ["..."],
          "message_types_in": [...],
          "message_types_out": [...]
        }
      ],
      "shared_state_keys": [...],
      "file_structure": {...}
    }
    """

    SYSTEM_PROMPT = """Tu es un Expert Architecte de Systèmes Multi-Agents avec 15 ans d'expérience.
Ta mission : concevoir l'architecture technique détaillée d'un MAS Python async.

CONTRAINTES TECHNIQUES :
- Python 3.11+ avec asyncio
- Communication via asyncio.Queue (MessageBus)
- LLM via Ollama (local)
- Loguru pour le logging
- Pydantic pour la validation

RÈGLES :
1. Réponds UNIQUEMENT avec un JSON valide, sans texte autour.
2. Chaque agent doit avoir un system_prompt spécialisé et détaillé (min 100 mots).
3. Les system_prompts doivent être en français.
4. Identifie clairement les dépendances entre agents.
5. Définis les clés de l'état partagé (SharedMemory).

FORMAT JSON ATTENDU (respecte EXACTEMENT cette structure) :
{
  "topology": "hub_and_spoke",
  "coordination_pattern": "sequential_pipeline",
  "error_strategy": "retry_with_fallback",
  "agent_details": [
    {
      "name": "orchestrator",
      "system_prompt": "Tu es l'orchestrateur principal...",
      "dependencies": [],
      "message_types_in": ["REQUEST"],
      "message_types_out": ["REQUEST", "RESPONSE"]
    }
  ],
  "shared_state_keys": ["user_input", "results", "errors"],
  "message_flow": [
    {"from": "orchestrator", "to": "agent_name", "trigger": "description"}
  ]
}"""

    def __init__(
        self,
        model: str = "mistral",
        message_bus=None,
        memory: SharedMemory = None,
    ):
        super().__init__(
            name="Architect",
            role="System Architect",
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            message_bus=message_bus,
            timeout=120,
        )
        self.memory_store = memory

    async def act(self, message: Message) -> Message:
        """
        Conçoit l'architecture à partir des spécifications de l'Analyste.

        Args:
            message: Message contenant le JSON d'analyse.

        Returns:
            Message contenant le JSON architectural.
        """
        logger.info(f"[{self.name}] 🏗️  Conception de l'architecture en cours...")
        self.status = AgentStatus.ACTING

        try:
            # Parser l'analyse reçue
            analysis = json.loads(message.content)
            architecture = await self._design(analysis)

            if self.memory_store:
                await self.memory_store.set("architecture_result", architecture)
                await self.memory_store.set_pipeline_stage("ARCHITECTURE_DONE")

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=json.dumps(architecture, ensure_ascii=False, indent=2),
                metadata={"topology": architecture.get("topology", "unknown")},
            )

        except Exception as e:
            logger.error(f"[{self.name}] Erreur architecture: {e}")
            if self.memory_store:
                await self.memory_store.add_error(str(e), context="Architect.act")
            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.ERROR,
                content=f"Erreur architecture: {str(e)}",
            )
        finally:
            self.status = AgentStatus.IDLE

    async def _design(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Effectue la conception architecturale via LLM.

        Args:
            analysis: Résultat de l'Analyste.

        Returns:
            Dict contenant les spécifications architecturales.
        """
        agents_summary = json.dumps(
            [
                {
                    "name": a["name"],
                    "role": a["role"],
                    "responsibilities": a.get("responsibilities", []),
                }
                for a in analysis.get("agents", [])
            ],
            ensure_ascii=False,
            indent=2,
        )

        context = {
            "SYSTÈME": analysis.get("system_name", "MAS"),
            "DESCRIPTION": analysis.get("description", ""),
            "AGENTS_IDENTIFIÉS": agents_summary,
            "PATTERN_COMMUNICATION": analysis.get("communication_pattern", "hub_and_spoke"),
            "INSTRUCTION": (
                "Conçois l'architecture technique détaillée de ce MAS. "
                "Pour chaque agent, fournis un system_prompt spécialisé "
                "d'au moins 5 phrases décrivant son rôle, ses capacités et ses contraintes. "
                "Réponds UNIQUEMENT avec le JSON, sans texte autour."
            ),
        }

        raw_response = await self.think(context)
        architecture = self._parse_json_response(raw_response)
        architecture = self._validate_and_enrich(architecture, analysis)

        logger.info(
            f"[{self.name}] ✅ Architecture conçue: "
            f"topologie={architecture.get('topology', '?')}"
        )
        return architecture

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Parse la réponse JSON avec plusieurs stratégies."""
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        json_str = CodeParser.extract_json_from_response(raw)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        import re
        matches = re.findall(r"\{.*\}", raw, re.DOTALL)
        for m in matches:
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue

        logger.warning(f"[{self.name}] JSON non parseable, structure par défaut")
        return {}

    def _validate_and_enrich(
        self,
        architecture: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Valide et enrichit l'architecture avec les données de l'analyse."""
        architecture.setdefault("topology", analysis.get("communication_pattern", "hub_and_spoke"))
        architecture.setdefault("coordination_pattern", "sequential_pipeline")
        architecture.setdefault("error_strategy", "retry_with_fallback")
        architecture.setdefault("shared_state_keys", ["user_input", "results", "errors"])
        architecture.setdefault("message_flow", [])

        # Enrichir les détails des agents depuis l'analyse
        existing_details = {
            d.get("name"): d
            for d in architecture.get("agent_details", [])
            if isinstance(d, dict) and "name" in d
        }

        enriched_details = []
        for agent in analysis.get("agents", []):
            name = agent.get("name", "")
            detail = existing_details.get(name, {})
            detail["name"] = name
            detail.setdefault(
                "system_prompt",
                self._generate_default_system_prompt(agent),
            )
            detail.setdefault("dependencies", [])
            detail.setdefault("message_types_in", ["REQUEST"])
            detail.setdefault("message_types_out", ["RESPONSE"])
            enriched_details.append(detail)

        architecture["agent_details"] = enriched_details
        return architecture

    @staticmethod
    def _generate_default_system_prompt(agent: Dict[str, Any]) -> str:
        """Génère un system_prompt par défaut pour un agent."""
        name = agent.get("name", "agent")
        role = agent.get("role", "agent spécialisé")
        responsibilities = agent.get("responsibilities", [])
        resp_str = "; ".join(responsibilities[:3]) if responsibilities else "traiter les requêtes"

        return (
            f"Tu es l'agent {name.replace('_', ' ').title()} d'un système multi-agents. "
            f"Ton rôle est : {role}. "
            f"Tes responsabilités principales sont : {resp_str}. "
            f"Tu reçois des messages structurés et tu dois produire des réponses précises et utiles. "
            f"Communique de manière claire et concise. "
            f"En cas d'erreur, explique le problème de manière détaillée pour permettre la correction. "
            f"Travaille toujours en collaboration avec les autres agents du système. "
            f"Priorise la qualité et la cohérence de tes réponses."
        )

