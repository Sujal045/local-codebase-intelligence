"""Ollama-backed chat / completion client.

Ollama runs a local HTTP server (default ``http://127.0.0.1:11434``).
We send messages to ``POST /api/chat`` with ``stream: false`` and read
the assistant message content.

Under the hood (conceptually):
1. The model tokenizes the prompt messages.
2. An autoregressive neural network predicts the next tokens one by one.
3. Decoding stops at an end-of-turn token or a configured limit.
4. We receive the decoded text string.

We are not implementing the neural net; Ollama loads and runs the model.
We *are* responsible for the API contract and clear error messages.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_MODEL = "qwen2.5-coder:3b"


class OllamaChatLLM:
    """Generate answers via a local Ollama chat model.

    Example::

        llm = OllamaChatLLM()
        answer = llm.complete(
            system="You answer questions about code.",
            user="What does add do?",
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_s: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str) -> str:
        if not system.strip():
            raise ValueError("system prompt must be non-empty")
        if not user.strip():
            raise ValueError("user prompt must be non-empty")

        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = self._post_chat(payload)
        return self._parse_message(data)

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OllamaChatLLM:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_s,
            )
        return self._client

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = client.post("/api/chat", json=payload)
        except httpx.RequestError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                "Is Ollama installed and running? "
                f"Then: ollama pull {self._model}"
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"Chat model {self._model!r} not found on Ollama. "
                f"Run: ollama pull {self._model}"
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama /api/chat failed ({response.status_code}): {response.text}"
            ) from exc

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Ollama response type: {type(data)!r}")
        return data

    @staticmethod
    def _parse_message(data: dict[str, Any]) -> str:
        message = data.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama response missing 'message' object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty assistant message")
        return content
