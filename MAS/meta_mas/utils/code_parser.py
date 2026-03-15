"""
Parser et validateur de code généré par les LLMs.

Extrait les blocs de code Markdown, valide la syntaxe Python,
et fournit des informations structurelles sur le code.
"""
import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class CodeBlock:
    """Représente un bloc de code extrait d'une réponse LLM."""

    language: str
    content: str
    filename: Optional[str] = None
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)


class CodeParser:
    """
    Parse et valide le code généré par les LLMs.

    Fonctionnalités :
    - Extraire les blocs de code Markdown (```python ... ```)
    - Valider la syntaxe Python via ast.parse()
    - Identifier les classes, fonctions et imports
    - Extraire plusieurs fichiers depuis une réponse structurée
    - Nettoyer les artefacts LLM courants
    """

    # Pattern pour les blocs de code Markdown
    CODE_BLOCK_RE = re.compile(r"```(?P<lang>\w+)?\n(?P<code>.*?)```", re.DOTALL)

    # Pattern pour détecter un nom de fichier avant un bloc de code
    FILE_HEADER_RE = re.compile(
        r"(?:📄\s*|#{1,3}\s*)?([a-zA-Z0-9_\-./]+\.(?:py|toml|yaml|yml|md|txt))\s*\n+```(?:\w+)?\n(.*?)```",
        re.DOTALL | re.MULTILINE,
    )

    @classmethod
    def extract_code_blocks(cls, text: str) -> List[CodeBlock]:
        """
        Extrait tous les blocs de code d'un texte Markdown.

        Args:
            text: Texte Markdown contenant des blocs de code.

        Returns:
            Liste de CodeBlock.
        """
        blocks: List[CodeBlock] = []
        for match in cls.CODE_BLOCK_RE.finditer(text):
            lang = match.group("lang") or "text"
            code = match.group("code").strip()
            blocks.append(CodeBlock(language=lang, content=code))
        logger.debug(f"CodeParser: {len(blocks)} bloc(s) de code extraits")
        return blocks

    @classmethod
    def extract_python_code(cls, text: str) -> Optional[str]:
        """
        Extrait le premier bloc de code Python.
        Si aucun bloc Markdown trouvé, tente d'extraire du code brut.

        Args:
            text: Réponse du LLM.

        Returns:
            Code Python ou None.
        """
        blocks = cls.extract_code_blocks(text)
        for block in blocks:
            if block.language.lower() in ("python", "py", ""):
                return block.content

        # Pas de bloc Markdown : essayer d'extraire le code brut
        stripped = text.strip()
        first_line = stripped.split("\n")[0] if stripped else ""
        code_starters = ("import ", "from ", "class ", "def ", "async def ", "#", '"""')
        if any(first_line.startswith(s) for s in code_starters):
            return stripped

        return None

    @classmethod
    def validate_python_syntax(cls, code: str) -> Tuple[bool, List[str]]:
        """
        Valide la syntaxe Python via ast.parse().

        Args:
            code: Code Python à valider.

        Returns:
            (is_valid: bool, errors: List[str])
        """
        try:
            ast.parse(code)
            logger.debug("CodeParser: syntaxe Python valide ✅")
            return True, []
        except SyntaxError as e:
            msg = f"SyntaxError ligne {e.lineno}: {e.msg}"
            logger.warning(f"CodeParser: {msg}")
            return False, [msg]
        except Exception as e:
            return False, [str(e)]

    @classmethod
    def check_imports(cls, code: str) -> List[str]:
        """
        Liste tous les modules importés dans le code.

        Returns:
            Liste des noms de modules importés.
        """
        imports: List[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
        except SyntaxError:
            pass
        return imports

    @classmethod
    def extract_class_names(cls, code: str) -> List[str]:
        """Extrait les noms de toutes les classes définies."""
        classes: List[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except SyntaxError:
            pass
        return classes

    @classmethod
    def extract_function_names(cls, code: str) -> List[str]:
        """Extrait les noms de toutes les fonctions/méthodes définies."""
        functions: List[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    functions.append(node.name)
        except SyntaxError:
            pass
        return functions

    @classmethod
    def extract_files_from_response(cls, text: str) -> Dict[str, str]:
        """
        Extrait plusieurs fichiers depuis une réponse LLM structurée.

        Cherche des patterns du type :
            # agents/mon_agent.py
            ```python
            ... code ...
            ```

        Returns:
            Dict {filename: code_content}
        """
        files: Dict[str, str] = {}
        for match in cls.FILE_HEADER_RE.finditer(text):
            filename = match.group(1).strip()
            code = match.group(2).strip()
            files[filename] = code
            logger.debug(f"CodeParser: fichier extrait — '{filename}'")
        return files

    @classmethod
    def clean_code(cls, code: str) -> str:
        """
        Nettoie le code en supprimant les artefacts LLM courants.

        - Supprime les balises Markdown résiduelles
        - Supprime les lignes de commentaire LLM avant le code
        """
        code = code.strip()
        # Supprimer les balises Markdown
        code = re.sub(r"^```\w*\n", "", code)
        code = re.sub(r"\n```$", "", code)

        # Supprimer les lignes de prose avant le code réel
        lines = code.split("\n")
        cleaned: List[str] = []
        in_code = False
        starters = ("import ", "from ", "class ", "def ", "async def ", "#", '"""', "'''", "@")
        for line in lines:
            if not in_code and not any(line.startswith(s) for s in starters) and line.strip():
                # Ligne de prose avant le code
                continue
            in_code = True
            cleaned.append(line)

        return "\n".join(cleaned)

    @classmethod
    def extract_json_from_response(cls, text: str) -> Optional[str]:
        """
        Extrait un bloc JSON depuis une réponse LLM.

        Returns:
            Contenu JSON brut ou None.
        """
        # Bloc json Markdown
        json_block = re.search(r"```json\n(.*?)```", text, re.DOTALL)
        if json_block:
            return json_block.group(1).strip()

        # Objet JSON brut
        json_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_obj:
            return json_obj.group(1).strip()

        return None

