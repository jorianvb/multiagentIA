# memory/short_term.py
# Gestion de la mémoire court terme (session en cours)

from typing import List, Dict, Any
from datetime import datetime


class ShortTermMemory:
    """
    Mémoire court terme : garde le contexte de la session en cours.
    Réinitialisée à chaque nouvelle session.

    Contient :
    - L'historique des échanges de la session
    - Les analyses successives
    - Le contexte compressé
    """

    def __init__(self, max_exchanges: int = 10):
        self.max_exchanges = max_exchanges  # Fenêtre glissante
        self.exchanges: List[Dict] = []
        self.session_start = datetime.now().isoformat()

    def add_exchange(self, story_text: str, user_request: str,
                     analysis_summary: str) -> None:
        """Ajoute un échange à la mémoire court terme."""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "story_excerpt": story_text[-500:],  # Garde les 500 derniers chars
            "user_request": user_request,
            "analysis_summary": analysis_summary
        }
        self.exchanges.append(exchange)

        # Fenêtre glissante : on garde les N derniers échanges
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges:]

    def get_recent_context(self, n: int = 3) -> str:
        """Retourne les N derniers échanges sous forme de texte."""
        recent = self.exchanges[-n:] if len(self.exchanges) >= n else self.exchanges

        if not recent:
            return "Première analyse de la session."

        lines = ["=== CONTEXTE RÉCENT DE LA SESSION ==="]
        for i, exc in enumerate(recent):
            lines.append(f"\n[Échange {i+1}] {exc['timestamp'][:16]}")
            lines.append(f"Demande : {exc['user_request']}")
            lines.append(f"Résumé : {exc['analysis_summary'][:200]}...")

        return "\n".join(lines)

    def compress_context(self) -> str:
        """
        Compresse le contexte pour éviter de dépasser la fenêtre du LLM.
        Stratégie : garder les faits clés, supprimer les détails répétitifs.
        """
        if not self.exchanges:
            return ""

        # Extraction des éléments uniques mentionnés
        all_requests = [e["user_request"] for e in self.exchanges]
        unique_requests = list(dict.fromkeys(all_requests))  # Déduplication ordonnée

        return f"Session avec {len(self.exchanges)} échange(s). " \
               f"Demandes principales : {'; '.join(unique_requests[:3])}"
