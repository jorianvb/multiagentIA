# src/utils/prompt_optimizer.py
"""
Guide des bonnes pratiques pour les prompts.
Ces patterns sont testés et optimisés pour Ollama/Llama3.
"""

# ✅ BONNES PRATIQUES
PROMPT_BEST_PRACTICES = {

    "structure_claire": """
    # Utilise une structure claire avec des sections
    Tu es un {role}.
    
    ## Contexte
    {context}
    
    ## Tâche
    {task}
    
    ## Format de réponse attendu
    {format}
    """,

    "json_forcing": """
    # Pour forcer du JSON, répète l'instruction ET donne un exemple
    Retourne UNIQUEMENT un JSON valide avec cette structure exacte :
    ```json
    {{"key": "value", "score": 0}}
    ```
    Ne retourne RIEN d'autre que le JSON.
    """,

    "chain_of_thought": """
    # Chain-of-thought pour les tâches complexes
    Réfléchis étape par étape :
    1. D'abord, analyse ...
    2. Ensuite, identifie ...
    3. Finalement, synthétise ...
    """,

    "contexte_limite": """
    # Limite le contexte passé aux agents
    # Utiliser max 60% de la fenêtre de contexte
    # Laisser 40% pour la réponse
    """,
}

# ❌ ANTI-PATTERNS À ÉVITER
PROMPT_ANTI_PATTERNS = {

    "trop_long": """
    ❌ Éviter les prompts > 2000 tokens
    ✅ Résumer le contexte si nécessaire
    """,

    "ambigu": """
    ❌ "Écris quelque chose sur l'IA"
    ✅ "Écris un article de 400 mots sur l'impact de l'IA sur l'emploi,
        pour un public de managers non-techniques"
    """,

    "trop_de_roles": """
    ❌ "Tu es à la fois un chercheur, un écrivain ET un critique..."
    ✅ Un agent = un rôle = un focus
    """,

    "sans_format": """
    ❌ Ne pas spécifier le format de sortie attendu
    ✅ Toujours préciser : JSON, Markdown, liste, etc.
    """,
}


def build_optimized_prompt(
        role: str,
        task: str,
        context: str,
        output_format: str,
        examples: list[str] | None = None,
        constraints: list[str] | None = None
) -> str:
    """
    Constructeur de prompt optimisé selon les meilleures pratiques.

    Args:
        role: Le rôle de l'agent
        task: La tâche spécifique
        context: Le contexte pertinent (résumé si trop long)
        output_format: Le format de sortie attendu
        examples: Exemples optionnels (few-shot)
        constraints: Contraintes à respecter

    Returns:
        Un prompt optimisé
    """
    parts = [f"Tu es {role}.\n"]

    if context:
        # Tronquer si trop long
        ctx = context[:1500] + "..." if len(context) > 1500 else context
        parts.append(f"## Contexte\n{ctx}\n")

    parts.append(f"## Tâche\n{task}\n")

    if constraints:
        parts.append("## Contraintes\n")
        for c in constraints:
            parts.append(f"- {c}")
        parts.append("")

    if examples:
        parts.append("## Exemples\n")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Exemple {i}: {ex}")
        parts.append("")

    parts.append(f"## Format de réponse\n{output_format}")

    return "\n".join(parts)
