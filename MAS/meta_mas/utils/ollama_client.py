"""
Wrapper async pour l'API Ollama.

Utilise la librairie officielle `ollama` avec retry et backoff exponentiel.
Fallback sur httpx si besoin.
"""
import asyncio
import json
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from loguru import logger


class OllamaClient:
    """
    Client async pour Ollama.

    Fonctionnalités :
    - Chat completion (async)
    - Streaming pour les longues générations
    - Retry avec backoff exponentiel
    - Timeout configurable
    - Health check & liste des modèles
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Appel chat completion vers Ollama.

        Args:
            messages : Liste de dicts {role, content}.
            stream   : Active le streaming.
            temperature : Température du modèle (0.0–1.0).

        Returns:
            Contenu de la réponse (str).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, **kwargs},
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"OllamaClient [{self.model}]: tentative {attempt}/{self.max_retries}"
                )

                if stream:
                    return await self._stream_chat(payload)

                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                content: str = data["message"]["content"]
                logger.debug(
                    f"OllamaClient [{self.model}]: réponse reçue ({len(content)} chars)"
                )
                return content

            except httpx.TimeoutException:
                logger.warning(f"OllamaClient: timeout tentative {attempt}")
                if attempt < self.max_retries:
                    wait = 2**attempt
                    logger.info(f"OllamaClient: nouvelle tentative dans {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Ollama timeout après {self.max_retries} tentatives"
                    )

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"OllamaClient: erreur HTTP {e.response.status_code} — {e.response.text[:200]}"
                )
                raise

            except Exception as e:
                logger.error(f"OllamaClient: erreur inattendue — {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("OllamaClient: toutes les tentatives ont échoué")

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Génération simple (endpoint /api/generate).

        Args:
            prompt: Prompt à compléter.

        Returns:
            Texte généré.
        """
        payload = {"model": self.model, "prompt": prompt, "stream": False, **kwargs}
        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "")

    async def _stream_chat(self, payload: Dict) -> str:
        """Effectue un chat en streaming et retourne le texte complet."""
        chunks: List[str] = []
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        chunks.append(chunk["message"]["content"])
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    async def list_models(self) -> List[str]:
        """Liste les modèles disponibles dans Ollama."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"OllamaClient: impossible de lister les modèles — {e}")
            return []

    async def check_health(self) -> bool:
        """Vérifie qu'Ollama est disponible."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Ferme le client HTTP."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

