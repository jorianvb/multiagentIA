"""
Agent Analyste — analyse la demande utilisateur et identifie les agents nécessaires.

Retourne un JSON structuré décrivant tous les agents du MAS à créer.
"""
import json
from typing import Any, Dict

from loguru import logger

from core.base_agent import BaseAgent, Message, MessageType, AgentStatus
from core.memory import SharedMemory
from utils.code_parser import CodeParser


class AnalystAgent(BaseAgent):
    """
    Agent spécialisé dans l'analyse des besoins en MAS.

    Reçoit une description textuelle d'un système multi-agent désiré
    et produit une spécification JSON structurée des agents nécessaires,
    leurs rôles, responsabilités et interactions.

    Output JSON :
    {
      "system_name": "NomDuMAS",
      "description": "...",
      "agents": [
        {
          "name": "agent_name",
          "class_name": "AgentClassName",
          "role": "...",
          "responsibilities": ["...", "..."],
          "inputs": ["msg_type"],
          "outputs": ["msg_type"],
          "needs_llm": true
        }
      ],
      "communication_pattern": "hub_and_spoke|peer_to_peer|blackboard",
      "workflow_description": "..."
    }
    """

    SYSTEM_PROMPT = """Tu es un Expert Architecte en Systèmes Multi-Agents (MAS).
Ta mission : analyser une description de MAS et identifier PRÉCISÉMENT les agents nécessaires.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après.
2. Identifie entre 3 et 8 agents selon la complexité.
3. Chaque agent doit avoir un rôle clair et non-redondant.
4. Inclus TOUJOURS un agent orchestrateur principal.
5. Les noms d'agents doivent être en snake_case (ex: query_handler).
6. Les class_name doivent être en PascalCase (ex: QueryHandlerAgent).

FORMAT DE RÉPONSE (JSON pur, pas de Markdown) :
{
  "system_name": "NomDuSystème",
  "description": "Description courte du MAS",
  "agents": [
    {
      "name": "orchestrator",
      "class_name": "OrchestratorAgent",
      "role": "Coordonne tous les agents",
      "responsibilities": ["Recevoir les requêtes", "Déléguer les tâches", "Agréger les résultats"],
      "inputs": ["user_request"],
      "outputs": ["final_response"],
      "needs_llm": true
    }
  ],
  "communication_pattern": "hub_and_spoke",
  "workflow_description": "Description du flux d'exécution"
}

PATTERNS DE COMMUNICATION :
- hub_and_spoke : un agent central coordonne tout (simple, recommandé par défaut)
- peer_to_peer  : les agents communiquent directement entre eux
- blackboard    : les agents lisent/écrivent dans une mémoire partagée
- pipeline      : les agents traitent séquentiellement dans une chaîne"""

    def __init__(
        self,
        model: str = "mistral",
        message_bus=None,
        memory: SharedMemory = None,
    ):
        super().__init__(
            name="Analyst",
            role="Requirements Analyst",
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            message_bus=message_bus,
            timeout=120,
        )
        self.memory_store = memory

    async def act(self, message: Message) -> Message:
        """
        Analyse la demande et retourne la spécification JSON des agents.

        Args:
            message: Message contenant la description du MAS désiré.

        Returns:
            Message de réponse contenant le JSON d'analyse.
        """
        logger.info(f"[{self.name}] 🔍 Analyse de la demande en cours...")
        self.status = AgentStatus.ACTING

        try:
            analysis = await self._analyze(message.content)

            # Stocker le résultat en mémoire partagée
            if self.memory_store:
                await self.memory_store.set("analysis_result", analysis)
                await self.memory_store.set_pipeline_stage("ANALYSIS_DONE")

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=json.dumps(analysis, ensure_ascii=False, indent=2),
                metadata={"agents_count": len(analysis.get("agents", []))},
            )

        except Exception as e:
            logger.error(f"[{self.name}] Erreur d'analyse: {e}")
            if self.memory_store:
                await self.memory_store.add_error(str(e), context="Analyst.act")
            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.ERROR,
                content=f"Erreur d'analyse: {str(e)}",
            )
        finally:
            self.status = AgentStatus.IDLE

    async def _analyze(self, user_request: str) -> Dict[str, Any]:
        """
        Effectue l'analyse LLM et parse le résultat JSON.

        Args:
            user_request: Description textuelle du MAS désiré.

        Returns:
            Dict contenant la spécification des agents.
        """
        context = {
            "DEMANDE_UTILISATEUR": user_request,
            "INSTRUCTION": (
                "Analyse cette demande et retourne le JSON structuré "
                "des agents nécessaires pour ce MAS. "
                "Réponds UNIQUEMENT avec le JSON, sans texte autour."
            ),
        }

        raw_response = await self.think(context)
        logger.debug(f"[{self.name}] Réponse brute: {raw_response[:200]}...")

        # Tenter d'extraire le JSON
        analysis = self._parse_json_response(raw_response)

        # Valider et compléter la structure
        analysis = self._validate_and_complete(analysis, user_request)

        logger.info(
            f"[{self.name}] ✅ Analyse terminée: "
            f"{len(analysis['agents'])} agents identifiés"
        )
        return analysis

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Parse la réponse JSON du LLM avec plusieurs stratégies."""
        # Stratégie 1 : JSON direct
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Stratégie 2 : Extraire le bloc JSON via CodeParser
        json_str = CodeParser.extract_json_from_response(raw)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Stratégie 3 : Chercher le premier { ... } valide
        import re
        matches = re.findall(r"\{.*\}", raw, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        logger.warning(f"[{self.name}] Impossible de parser le JSON, utilisation d'une structure par défaut")
        return {}

    def _validate_and_complete(
        self, analysis: Dict[str, Any], user_request: str
    ) -> Dict[str, Any]:
        """Valide et complète la structure JSON si nécessaire."""
        # S'assurer que les champs obligatoires existent
        if "system_name" not in analysis:
            # Extraire un nom depuis la demande
            words = user_request.split()[:3]
            analysis["system_name"] = "".join(w.capitalize() for w in words) + "MAS"

        if "description" not in analysis:
            analysis["description"] = user_request[:200]

        if "agents" not in analysis or not isinstance(analysis["agents"], list):
            analysis["agents"] = []

        if "communication_pattern" not in analysis:
            analysis["communication_pattern"] = "hub_and_spoke"

        if "workflow_description" not in analysis:
            analysis["workflow_description"] = "Pipeline séquentiel coordonné par l'orchestrateur"

        # Valider chaque agent
        validated_agents = []
        for agent in analysis["agents"]:
            if not isinstance(agent, dict):
                continue
            # Champs obligatoires
            if "name" not in agent:
                continue
            agent.setdefault("class_name", self._to_class_name(agent["name"]))
            agent.setdefault("role", f"Agent {agent['name']}")
            agent.setdefault("responsibilities", [f"Gérer les tâches de {agent['name']}"])
            agent.setdefault("inputs", ["request"])
            agent.setdefault("outputs", ["response"])
            agent.setdefault("needs_llm", True)
            validated_agents.append(agent)

        # S'assurer qu'il y a un orchestrateur
        has_orchestrator = any(
            "orchestrat" in a.get("name", "").lower() or
            "orchestrat" in a.get("role", "").lower()
            for a in validated_agents
        )
        if not has_orchestrator:
            validated_agents.insert(0, {
                "name": "orchestrator",
                "class_name": "OrchestratorAgent",
                "role": "Coordonne tous les agents du MAS",
                "responsibilities": [
                    "Recevoir les requêtes utilisateur",
                    "Déléguer aux agents spécialisés",
                    "Agréger et retourner les résultats",
                ],
                "inputs": ["user_request"],
                "outputs": ["final_response"],
                "needs_llm": True,
            })

        analysis["agents"] = validated_agents
        return analysis

    @staticmethod
    def _to_class_name(name: str) -> str:
        """Convertit snake_case en PascalCase + 'Agent'."""
        parts = name.replace("-", "_").split("_")
        base = "".join(p.capitalize() for p in parts)
        if not base.endswith("Agent"):
            base += "Agent"
        return base

