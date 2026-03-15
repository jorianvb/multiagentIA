"""
Agent Validateur — vérifie la cohérence et la qualité du code généré.

Effectue des validations syntaxiques (ast.parse), structurelles et logiques.
Propose des corrections en cas d'erreur.
"""
import ast
import json
from typing import Any, Dict, List, Tuple

from loguru import logger

from core.base_agent import BaseAgent, Message, MessageType, AgentStatus
from core.memory import SharedMemory
from utils.code_parser import CodeParser


class ValidatorAgent(BaseAgent):
    """
    Agent spécialisé dans la validation du code généré.

    Niveaux de validation :
    1. Syntaxe Python (ast.parse) — bloquant
    2. Imports requis (BaseAgent, Message, etc.) — avertissement
    3. Structure de classe (hérite de BaseAgent, a la méthode act) — bloquant
    4. Méthodes async requises — avertissement
    5. Docstrings et commentaires — informatif

    Output JSON :
    {
      "filename": "agents/my_agent.py",
      "is_valid": true/false,
      "score": 0-100,
      "errors": [...],
      "warnings": [...],
      "suggestions": [...],
      "passed_checks": [...]
    }
    """

    SYSTEM_PROMPT = """Tu es un Expert en Revue de Code Python spécialisé dans les systèmes multi-agents.
Tu analyses du code Python et identifies les problèmes de qualité, de cohérence et de bonnes pratiques.

CRITÈRES D'ÉVALUATION :
1. Syntaxe Python correcte (ast.parse valide)
2. Héritage correct depuis BaseAgent
3. Implémentation de la méthode async act(self, message: Message) -> Message
4. Gestion d'erreurs avec try/except
5. Logging avec loguru (logger.info, logger.error, etc.)
6. Type hints sur les méthodes
7. Docstrings présents
8. Code async-first (pas de code bloquant)

RÉPONDS EN JSON :
{
  "is_valid": true/false,
  "score": 0-100,
  "errors": ["erreur critique 1", ...],
  "warnings": ["avertissement 1", ...],
  "suggestions": ["suggestion 1", ...],
  "corrections": "code Python corrigé si nécessaire"
}"""

    def __init__(
        self,
        model: str = "mistral",
        message_bus=None,
        memory: SharedMemory = None,
    ):
        super().__init__(
            name="Validator",
            role="Code Validator",
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            message_bus=message_bus,
            timeout=90,
        )
        self.memory_store = memory

    async def act(self, message: Message) -> Message:
        """
        Valide le code reçu et retourne un rapport de validation.

        Le contenu du message peut être :
        - Un JSON {filename: code_content} pour valider plusieurs fichiers
        - Un code Python brut pour valider un seul fichier

        Returns:
            Message contenant le rapport JSON de validation.
        """
        logger.info(f"[{self.name}] 🔍 Validation du code en cours...")
        self.status = AgentStatus.ACTING

        try:
            # Essayer de parser comme JSON (multi-fichiers)
            try:
                files_dict = json.loads(message.content)
                if isinstance(files_dict, dict):
                    report = await self._validate_all_files(files_dict)
                else:
                    raise ValueError("Pas un dict")
            except (json.JSONDecodeError, ValueError):
                # Valider comme code Python brut
                report = {
                    "overall": await self._validate_single(
                        "code", message.content
                    ),
                    "files": {},
                }

            if self.memory_store:
                await self.memory_store.append("validation_results", report)
                overall_valid = report.get("overall_valid", False)
                if overall_valid:
                    await self.memory_store.set_pipeline_stage("VALIDATION_PASSED")
                else:
                    await self.memory_store.set_pipeline_stage("VALIDATION_FAILED")

            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=json.dumps(report, ensure_ascii=False, indent=2),
                metadata={
                    "overall_valid": report.get("overall_valid", False),
                    "score": report.get("average_score", 0),
                },
            )

        except Exception as e:
            logger.error(f"[{self.name}] Erreur validation: {e}")
            if self.memory_store:
                await self.memory_store.add_error(str(e), context="Validator.act")
            return Message(
                sender=self.name,
                receiver=message.sender,
                type=MessageType.ERROR,
                content=f"Erreur validation: {str(e)}",
            )
        finally:
            self.status = AgentStatus.IDLE

    async def _validate_all_files(
        self, files_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        """Valide tous les fichiers Python dans le dict."""
        results: Dict[str, Any] = {}
        total_score = 0
        all_valid = True

        for filename, code in files_dict.items():
            if not filename.endswith(".py"):
                continue

            result = await self._validate_single(filename, code)
            results[filename] = result

            if not result["is_valid"]:
                all_valid = False
            total_score += result.get("score", 0)

        py_count = max(1, sum(1 for f in files_dict if f.endswith(".py")))
        avg_score = total_score / py_count

        logger.info(
            f"[{self.name}] ✅ Validation: "
            f"{sum(1 for r in results.values() if r['is_valid'])}/{py_count} valides, "
            f"score moyen={avg_score:.0f}"
        )

        return {
            "overall_valid": all_valid,
            "average_score": round(avg_score, 1),
            "files": results,
            "total_files": py_count,
            "valid_files": sum(1 for r in results.values() if r["is_valid"]),
        }

    async def _validate_single(
        self, filename: str, code: str
    ) -> Dict[str, Any]:
        """Effectue toutes les validations sur un fichier."""
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []
        passed: List[str] = []

        # --- Validation 1: Syntaxe Python ---
        is_syntax_valid, syntax_errors = CodeParser.validate_python_syntax(code)
        if is_syntax_valid:
            passed.append("syntaxe Python valide")
        else:
            errors.extend(syntax_errors)
            # Si la syntaxe est invalide, arrêter ici
            return {
                "filename": filename,
                "is_valid": False,
                "score": 0,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "passed_checks": passed,
            }

        # --- Validation 2: Imports ---
        imports = CodeParser.check_imports(code)
        if "loguru" in str(imports) or "logger" in code:
            passed.append("loguru utilisé pour le logging")
        else:
            warnings.append("Loguru non importé, utiliser 'from loguru import logger'")

        # --- Validation 3: Structure de classe ---
        classes = CodeParser.extract_class_names(code)
        if classes:
            passed.append(f"classe(s) trouvée(s): {', '.join(classes)}")
        else:
            warnings.append("Aucune classe définie")

        # Vérifier l'héritage BaseAgent
        if "BaseAgent" in code:
            passed.append("héritage de BaseAgent présent")
        elif filename not in ("agents/base_agent.py", "base_agent.py"):
            warnings.append("Héritage de BaseAgent non détecté")

        # --- Validation 4: Méthode act ---
        functions = CodeParser.extract_function_names(code)
        if "act" in functions:
            passed.append("méthode 'act' implémentée")
        elif filename not in ("agents/base_agent.py", "base_agent.py",
                               "core/message_bus.py", "core/orchestrator.py",
                               "config/settings.py", "main.py"):
            errors.append("Méthode 'act(self, message: Message) -> Message' manquante")

        # --- Validation 5: Async ---
        if "async def" in code:
            passed.append("code async-first")
        elif filename not in ("config/settings.py",):
            suggestions.append("Considérer l'utilisation de méthodes async")

        # --- Validation 6: Gestion d'erreurs ---
        if "try:" in code and "except" in code:
            passed.append("gestion d'erreurs présente")
        else:
            suggestions.append("Ajouter try/except pour la gestion d'erreurs")

        # --- Validation 7: Docstrings ---
        if '"""' in code or "'''" in code:
            passed.append("docstrings présentes")
        else:
            suggestions.append("Ajouter des docstrings aux classes et méthodes")

        # --- Calcul du score ---
        score = self._calculate_score(errors, warnings, suggestions, passed)

        is_valid = len(errors) == 0

        return {
            "filename": filename,
            "is_valid": is_valid,
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "passed_checks": passed,
        }

    @staticmethod
    def _calculate_score(
        errors: List[str],
        warnings: List[str],
        suggestions: List[str],
        passed: List[str],
    ) -> int:
        """Calcule un score de qualité de 0 à 100."""
        score = 100
        score -= len(errors) * 25      # -25 par erreur critique
        score -= len(warnings) * 10    # -10 par avertissement
        score -= len(suggestions) * 5  # -5 par suggestion
        score += len(passed) * 5       # +5 par vérification passée
        return max(0, min(100, score))

