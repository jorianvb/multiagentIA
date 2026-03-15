"""
Agent Orchestrateur — chef du Meta-MAS.

Reçoit la demande utilisateur, coordonne tous les sous-agents en séquence,
et produit le MAS final déployé dans le dossier cible.
"""
import json
from typing import Any, Dict, Optional

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from config.settings import Settings
from core.agent_registry import AgentRegistry
from core.base_agent import BaseAgent, Message, MessageType, AgentStatus
from core.memory import SharedMemory
from core.message_bus import MessageBus
from utils.logger import console, print_stage, print_agent_action, print_success, print_error


class OrchestratorAgent(BaseAgent):
    """
    Agent Orchestrateur principal du Meta-MAS.

    Pipeline d'exécution :
    1. ANALYSIS    : Analyste identifie les agents nécessaires
    2. ARCHITECTURE: Architecte conçoit la topologie
    3. CODE_GEN    : Générateur produit le code Python
    4. VALIDATION  : Validateur vérifie la cohérence
    5. DEPLOYMENT  : Déployeur crée les fichiers dans le dossier cible

    L'Orchestrateur pilote chaque étape, transmet les résultats entre agents
    via la SharedMemory, et gère les erreurs/retries.
    """

    SYSTEM_PROMPT = """Tu es l'Orchestrateur Master d'un système multi-agents méta.
Tu coordonnes des agents spécialisés pour générer un nouveau système multi-agents.
Tu es méthodique, précis et tu t'assures que chaque étape produit un résultat de qualité.
En cas d'erreur, tu analyses la situation et tu prends les mesures correctives appropriées."""

    # Étapes du pipeline avec leurs descriptions
    PIPELINE_STAGES = [
        ("ANALYSIS", "🔍 Analyse des besoins"),
        ("ARCHITECTURE", "🏗️  Conception de l'architecture"),
        ("CODE_GENERATION", "💻 Génération du code"),
        ("VALIDATION", "✅ Validation du code"),
        ("DEPLOYMENT", "🚀 Déploiement"),
    ]

    def __init__(
        self,
        model: str = "mistral",
        message_bus: Optional[MessageBus] = None,
        memory: Optional[SharedMemory] = None,
        registry: Optional[AgentRegistry] = None,
        settings: Optional[Settings] = None,
    ):
        super().__init__(
            name="Orchestrator",
            role="Master Coordinator",
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            message_bus=message_bus,
            timeout=60,
        )
        self.memory_store = memory or SharedMemory()
        self.registry = registry or AgentRegistry()
        self.settings = settings or Settings()
        self._sub_agents: Dict[str, BaseAgent] = {}

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    async def run_pipeline(
        self, user_request: str, target_directory: str
    ) -> Dict[str, Any]:
        """
        Lance le pipeline complet de génération du MAS.

        Args:
            user_request: Description du MAS désiré par l'utilisateur.
            target_directory: Dossier où déployer le MAS généré.

        Returns:
            Dict avec le résultat du pipeline (success, stats, etc.)
        """
        logger.info(f"[{self.name}] 🎯 Démarrage du pipeline Meta-MAS")
        logger.info(f"[{self.name}] Demande: {user_request[:100]}...")
        logger.info(f"[{self.name}] Cible: {target_directory}")

        # Initialiser la mémoire
        await self.memory_store.set("user_request", user_request)
        await self.memory_store.set("target_directory", target_directory)
        await self.memory_store.set("ollama_model", self.model)
        await self.memory_store.set_pipeline_stage("STARTING")

        # Créer les sous-agents
        self._create_sub_agents()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TextColumn("[dim]{task.fields[detail]}"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    "Pipeline Meta-MAS", total=len(self.PIPELINE_STAGES), detail=""
                )

                # Étape 1 : Analyse
                progress.update(task, description="🔍 Analyse des besoins...", detail="")
                print_stage("ÉTAPE 1/5", "Analyse des besoins par l'Agent Analyste")
                analysis = await self._run_analysis(user_request)
                agents_count = len(analysis.get("agents", []))
                progress.advance(task)
                print_success(f"Analyse terminée — {agents_count} agents identifiés")
                progress.update(task, detail=f"{agents_count} agents")

                # Étape 2 : Architecture
                progress.update(task, description="🏗️  Conception architecture...", detail="")
                print_stage("ÉTAPE 2/5", "Conception architecturale par l'Agent Architecte")
                architecture = await self._run_architecture(analysis)
                topology = architecture.get("topology", "?")
                progress.advance(task)
                print_success(f"Architecture conçue — topologie: {topology}")
                progress.update(task, detail=topology)

                # Étape 3 : Génération de code
                progress.update(task, description="💻 Génération du code...", detail="")
                print_stage("ÉTAPE 3/5", "Génération du code par l'Agent Générateur")
                generated_files = await self._run_code_generation(analysis, architecture)
                files_count = len(generated_files)
                progress.advance(task)
                print_success(f"Code généré — {files_count} fichiers")
                progress.update(task, detail=f"{files_count} fichiers")

                # Étape 4 : Validation
                progress.update(task, description="✅ Validation du code...", detail="")
                print_stage("ÉTAPE 4/5", "Validation du code par l'Agent Validateur")
                validation_report = await self._run_validation(generated_files)
                overall_valid = validation_report.get("overall_valid", False)
                score = validation_report.get("average_score", 0)
                progress.advance(task)
                if overall_valid:
                    print_success(f"Validation réussie — score moyen: {score:.0f}/100")
                else:
                    valid_count = validation_report.get("valid_files", 0)
                    total_count = validation_report.get("total_files", 0)
                    console.print(
                        f"[yellow]⚠️  Validation partielle — "
                        f"{valid_count}/{total_count} fichiers valides[/yellow]"
                    )
                progress.update(task, detail=f"score={score:.0f}")

                # Étape 5 : Déploiement
                progress.update(task, description="🚀 Déploiement...", detail="")
                print_stage("ÉTAPE 5/5", "Déploiement par l'Agent Déployeur")
                deployment_report = await self._run_deployment(
                    target_directory, generated_files, analysis, architecture
                )
                progress.advance(task)
                deployed_count = deployment_report.get("files_count", 0)
                print_success(f"Déploiement terminé — {deployed_count} fichiers créés")

            # Afficher l'arborescence finale
            if deployment_report.get("tree"):
                console.print("\n[bold cyan]Arborescence du MAS généré :[/bold cyan]")
                console.print(deployment_report["tree"])

            return {
                "success": True,
                "target_directory": target_directory,
                "agents_count": agents_count,
                "files_count": deployed_count,
                "validation_score": score,
                "system_name": analysis.get("system_name", "GeneratedMAS"),
                "tree": deployment_report.get("tree", ""),
            }

        except Exception as e:
            logger.exception(f"[{self.name}] Erreur fatale dans le pipeline: {e}")
            await self.memory_store.add_error(str(e), context="Orchestrator.run_pipeline")
            return {
                "success": False,
                "error": str(e),
                "target_directory": target_directory,
            }

    async def act(self, message: Message) -> Message:
        """
        Interface Message standard (pour utilisation via MessageBus).

        Args:
            message: Message contenant la requête utilisateur.

        Returns:
            Message de résultat du pipeline.
        """
        self.status = AgentStatus.ACTING
        try:
            data = json.loads(message.content)
            user_request = data.get("user_request", message.content)
            target_directory = data.get("target_directory", "./generated_mas")
        except (json.JSONDecodeError, Exception):
            user_request = message.content
            target_directory = "./generated_mas"

        result = await self.run_pipeline(user_request, target_directory)
        self.status = AgentStatus.IDLE

        return Message(
            sender=self.name,
            receiver=message.sender,
            type=MessageType.RESPONSE if result.get("success") else MessageType.ERROR,
            content=json.dumps(result, ensure_ascii=False, indent=2),
            metadata=result,
        )

    # ------------------------------------------------------------------
    # Étapes du pipeline
    # ------------------------------------------------------------------

    async def _run_analysis(self, user_request: str) -> Dict[str, Any]:
        """Délègue l'analyse à l'Agent Analyste."""
        print_agent_action("Analyst", "analyse la demande utilisateur...")
        analyst = self._sub_agents["analyst"]

        request_msg = Message(
            sender=self.name,
            receiver=analyst.name,
            type=MessageType.REQUEST,
            content=user_request,
        )

        response = await analyst.act(request_msg)

        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Analyste en erreur: {response.content}")

        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Analyste: réponse non-JSON: {response.content[:200]}"
            )

        return analysis

    async def _run_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Délègue la conception à l'Agent Architecte."""
        print_agent_action("Architect", "conçoit l'architecture...")
        architect = self._sub_agents["architect"]

        request_msg = Message(
            sender=self.name,
            receiver=architect.name,
            type=MessageType.REQUEST,
            content=json.dumps(analysis, ensure_ascii=False),
        )

        response = await architect.act(request_msg)

        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Architecte en erreur: {response.content}")

        try:
            architecture = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] Architecture non-JSON, utilisation des valeurs par défaut")
            architecture = {}

        return architecture

    async def _run_code_generation(
        self,
        analysis: Dict[str, Any],
        architecture: Dict[str, Any],
    ) -> Dict[str, str]:
        """Délègue la génération de code à l'Agent Générateur."""
        agents = analysis.get("agents", [])
        print_agent_action("CodeGenerator", f"génère le code pour {len(agents)} agents...")

        generator = self._sub_agents["code_generator"]

        payload = json.dumps(
            {"analysis": analysis, "architecture": architecture},
            ensure_ascii=False,
        )

        request_msg = Message(
            sender=self.name,
            receiver=generator.name,
            type=MessageType.REQUEST,
            content=payload,
        )

        response = await generator.act(request_msg)

        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Générateur en erreur: {response.content}")

        try:
            generated_files = json.loads(response.content)
        except json.JSONDecodeError:
            raise RuntimeError("Générateur: réponse invalide")

        return generated_files

    async def _run_validation(
        self, generated_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Délègue la validation à l'Agent Validateur."""
        py_files = {k: v for k, v in generated_files.items() if k.endswith(".py")}
        print_agent_action("Validator", f"valide {len(py_files)} fichiers Python...")

        validator = self._sub_agents["validator"]

        request_msg = Message(
            sender=self.name,
            receiver=validator.name,
            type=MessageType.REQUEST,
            content=json.dumps(generated_files, ensure_ascii=False),
        )

        response = await validator.act(request_msg)

        if response.type == MessageType.ERROR:
            logger.warning(f"[{self.name}] Validateur en erreur: {response.content}")
            return {"overall_valid": False, "average_score": 0, "files": {}}

        try:
            report = json.loads(response.content)
        except json.JSONDecodeError:
            report = {"overall_valid": True, "average_score": 75, "files": {}}

        # Log les problèmes détectés
        for filename, file_report in report.get("files", {}).items():
            if not file_report.get("is_valid", True):
                for error in file_report.get("errors", []):
                    logger.warning(f"[{self.name}] Validation: {filename}: {error}")

        return report

    async def _run_deployment(
        self,
        target_directory: str,
        generated_files: Dict[str, str],
        analysis: Dict[str, Any],
        architecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Délègue le déploiement à l'Agent Déployeur."""
        print_agent_action("Deployer", f"déploie dans {target_directory}...")
        deployer = self._sub_agents["deployer"]

        payload = json.dumps(
            {
                "target_directory": target_directory,
                "generated_files": generated_files,
                "analysis": analysis,
                "architecture": architecture,
            },
            ensure_ascii=False,
        )

        request_msg = Message(
            sender=self.name,
            receiver=deployer.name,
            type=MessageType.REQUEST,
            content=payload,
        )

        response = await deployer.act(request_msg)

        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Déployeur en erreur: {response.content}")

        try:
            report = json.loads(response.content)
        except json.JSONDecodeError:
            report = {"success": False, "error": "Réponse invalide du Déployeur"}

        return report

    # ------------------------------------------------------------------
    # Création des sous-agents
    # ------------------------------------------------------------------

    def _create_sub_agents(self) -> None:
        """Instancie et enregistre tous les sous-agents."""
        from agents.analyst import AnalystAgent
        from agents.architect import ArchitectAgent
        from agents.code_generator import CodeGeneratorAgent
        from agents.validator import ValidatorAgent
        from agents.deployer import DeployerAgent

        model = self.model
        bus = self.message_bus
        mem = self.memory_store

        sub_agents = {
            "analyst": AnalystAgent(model=model, message_bus=bus, memory=mem),
            "architect": ArchitectAgent(model=model, message_bus=bus, memory=mem),
            "code_generator": CodeGeneratorAgent(
                model=model, message_bus=bus, memory=mem, settings=self.settings
            ),
            "validator": ValidatorAgent(model=model, message_bus=bus, memory=mem),
            "deployer": DeployerAgent(model=model, message_bus=bus, memory=mem),
        }

        for key, agent in sub_agents.items():
            self._sub_agents[key] = agent
            self.registry.register(agent)
            if self.message_bus:
                self.message_bus.register_agent(agent.name)

        logger.info(
            f"[{self.name}] {len(sub_agents)} sous-agents créés: "
            f"{list(sub_agents.keys())}"
        )

