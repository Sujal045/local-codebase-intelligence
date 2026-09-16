"""Ollama-backed chat / completion client.

Ollama runs a local HTTP server (default ``http://127.0.0.1:11434``).
We send messages to ``POST /api/chat`` with ``stream: false``.

Slice 1d used ``complete(system=, user=)`` for one-shot RAG text.
Slice 6B adds ``respond(messages, tools=)`` for native tool calling:

    request  →  messages + tools schemas
    response →  content and/or message.tool_calls
    we map   →  AgentTurn(content, tool_calls)

When a small model prints a tool request as JSON in ``content`` instead
of filling ``tool_calls``, ``parse_agent_turn`` recovers it so the agent
loop can still execute the tool.

Under the hood (conceptually):
1. The model tokenizes the prompt messages (and tool schemas).
2. An autoregressive neural network predicts the next tokens.
3. Decoding stops at an end-of-turn token or a configured limit.
4. For tools, the runtime parses a structured tool-call payload
   instead of (or in addition to) plain assistant text.

We are not implementing the neural net; Ollama loads and runs the model.
We *are* responsible for the API contract, message shaping, and clear errors.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.agent.types import AgentTurn, ToolCall
from app.config import DEFAULT_CHAT_MODEL

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaChatLLM:
    """Generate answers (and optional tool calls) via a local Ollama model.

    Example::

        llm = OllamaChatLLM()
        answer = llm.complete(
            system="You answer questions about code.",
            user="What does add do?",
        )
        turn = llm.respond(
            [{"role": "user", "content": "Find spam scoring"}],
            tools=[search_code_tool.spec()],
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

    def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """One agent round: send history + tool schemas, parse the next turn."""
        if not messages:
            raise ValueError("messages must be non-empty")

        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "messages": messages_for_ollama(messages),
        }
        if tools:
            payload["tools"] = tools

        data = self._post_chat(payload)
        return parse_agent_turn(data)

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


def messages_for_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert loop messages into the wire shape Ollama expects.

    The agent loop stores a compact form::

        {"role": "assistant", "tool_calls": [{"name": ..., "arguments": {...}}]}
        {"role": "tool", "name": ..., "content": "..."}

    Ollama prefers OpenAI-ish nesting and ``tool_name`` on tool results.
    """
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] must be a dict")
        role = message.get("role")
        if role == "assistant":
            out.append(_assistant_for_ollama(message))
        elif role == "tool":
            out.append(_tool_for_ollama(message))
        else:
            out.append(dict(message))
    return out


def parse_agent_turn(data: dict[str, Any]) -> AgentTurn:
    """Parse an Ollama /api/chat body into an ``AgentTurn``.

    Small local models sometimes *print* a tool call as JSON in
    ``message.content`` instead of filling ``message.tool_calls``. When
    native ``tool_calls`` are missing, we recover from that text so the
    agent loop can still execute the tool.
    """
    message = data.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama response missing 'message' object")

    raw_content = message.get("content")
    content = raw_content if isinstance(raw_content, str) else ""
    tool_calls = parse_tool_calls(message.get("tool_calls"))

    if not tool_calls:
        recovered = tool_calls_from_content(content)
        if recovered:
            return AgentTurn(content="", tool_calls=recovered)

    if not content.strip() and not tool_calls:
        raise RuntimeError(
            "Ollama returned neither assistant content nor tool_calls"
        )
    return AgentTurn(content=content, tool_calls=tool_calls)


def tool_calls_from_content(content: str) -> tuple[ToolCall, ...]:
    """Recover tool calls from assistant text that looks like a tool request.

    Handles shapes small models commonly emit as plain text::

        {"name": "search_code", "arguments": {"query": "spam"}}
        {"name": "echo", "parameters": {"text": "hi"}}
        {"tool_calls": [{"function": {"name": "echo", "arguments": {...}}}]}

    Also strips optional markdown `` ```json `` fences. Returns ``()`` when
    the text is not a recognizable tool-call payload (normal final answers).
    """
    stripped = _strip_markdown_fence(content).strip()
    if not stripped:
        return ()

    payload = _loads_json_object(stripped)
    if payload is None:
        return ()

    if "tool_calls" in payload:
        try:
            return parse_tool_calls(payload["tool_calls"])
        except RuntimeError:
            return ()

    call = _tool_call_from_mapping(payload)
    if call is None:
        return ()
    return (call,)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    # Drop opening ``` or ```json and closing ```
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def _loads_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` (or an embedded `{...}`) as a JSON object."""
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
    return None


def _tool_call_from_mapping(payload: dict[str, Any]) -> ToolCall | None:
    """Build one ``ToolCall`` from a flat ``{name, arguments|parameters}`` object."""
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        function = payload.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                try:
                    arguments = normalize_tool_arguments(
                        function.get("arguments", function.get("parameters", {}))
                    )
                except RuntimeError:
                    return None
                return ToolCall(name=name.strip(), arguments=arguments)
        return None

    raw_args = payload.get("arguments", payload.get("parameters"))
    if raw_args is None and set(payload.keys()) - {"name"}:
        # Reject free-form JSON that is not shaped like a tool call.
        return None
    if raw_args is None:
        raw_args = {}
    try:
        arguments = normalize_tool_arguments(raw_args)
    except RuntimeError:
        return None
    return ToolCall(name=name.strip(), arguments=arguments)


def parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    """Normalize Ollama/OpenAI-ish ``tool_calls`` into ``ToolCall`` tuples."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RuntimeError("Ollama 'tool_calls' field is not a list")

    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"tool_calls[{index}] is not an object")

        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            # Compact form already used by our agent loop.
            name = item.get("name")
            arguments = item.get("arguments", {})

        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"tool_calls[{index}] missing function name")

        calls.append(
            ToolCall(
                name=name.strip(),
                arguments=normalize_tool_arguments(arguments),
            )
        )
    return tuple(calls)


def normalize_tool_arguments(raw: Any) -> dict[str, Any]:
    """Coerce tool arguments to a plain dict.

    Ollama usually returns a JSON object. Some models/APIs return a JSON
    string. A known bug sometimes wraps params as
    ``{"arguments": {...}, "name": "..."}`` — we unwrap that.
    """
    if raw is None:
        return {}
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"tool call arguments are not valid JSON: {raw!r}"
            ) from exc
        return normalize_tool_arguments(parsed)
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"tool call arguments must be a dict or JSON object string, "
            f"got {type(raw)!r}"
        )

    # Unwrap duplicated envelopes from some Ollama model outputs.
    if (
        set(raw.keys()) <= {"arguments", "name"}
        and "arguments" in raw
        and isinstance(raw["arguments"], dict)
    ):
        return dict(raw["arguments"])
    return dict(raw)


def _assistant_for_ollama(message: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content", "") or "",
    }
    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return converted

    ollama_calls: list[dict[str, Any]] = []
    for call in parse_tool_calls(raw_calls):
        ollama_calls.append(
            {
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
        )
    converted["tool_calls"] = ollama_calls
    return converted


def _tool_for_ollama(message: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "role": "tool",
        "content": message.get("content", "") or "",
    }
    name = message.get("tool_name") or message.get("name")
    if isinstance(name, str) and name.strip():
        converted["tool_name"] = name.strip()
    return converted
