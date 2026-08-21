"""Tests for Ollama native tool calling (Slice 6B)."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from app.agent import AgentTurn, ToolCall, run_agent
from app.llm.ollama_chat import (
    OllamaChatLLM,
    messages_for_ollama,
    normalize_tool_arguments,
    parse_agent_turn,
    parse_tool_calls,
)
from app.tools.base import ToolResult


class EchoTool:
    name = "echo"

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Echo text back",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    def run(self, **arguments):
        return ToolResult(name=self.name, content=f"echo:{arguments.get('text', '')}")


def test_messages_for_ollama_rewrites_assistant_and_tool() -> None:
    wire = messages_for_ollama(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "echo", "arguments": {"text": "spam"}}],
            },
            {"role": "tool", "name": "echo", "content": "echo:spam"},
        ]
    )

    assert wire[2]["tool_calls"] == [
        {
            "type": "function",
            "function": {"name": "echo", "arguments": {"text": "spam"}},
        }
    ]
    assert wire[3] == {
        "role": "tool",
        "tool_name": "echo",
        "content": "echo:spam",
    }


def test_normalize_arguments_accepts_dict_string_and_envelope() -> None:
    assert normalize_tool_arguments({"city": "NY"}) == {"city": "NY"}
    assert normalize_tool_arguments('{"city": "NY"}') == {"city": "NY"}
    assert normalize_tool_arguments(
        {"name": "get_city", "arguments": {"city": "NY"}}
    ) == {"city": "NY"}
    assert normalize_tool_arguments(None) == {}


def test_parse_tool_calls_from_ollama_shape() -> None:
    calls = parse_tool_calls(
        [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": {"text": "hi"},
                },
            }
        ]
    )
    assert calls == (ToolCall(name="echo", arguments={"text": "hi"}),)


def test_parse_agent_turn_allows_empty_content_with_tools() -> None:
    turn = parse_agent_turn(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "echo",
                            "arguments": {"text": "x"},
                        }
                    }
                ],
            }
        }
    )
    assert turn.content == ""
    assert turn.tool_calls[0].name == "echo"


def test_parse_agent_turn_rejects_empty_response() -> None:
    with pytest.raises(RuntimeError, match="neither"):
        parse_agent_turn({"message": {"role": "assistant", "content": "  "}})


def test_ollama_respond_sends_tools_and_parses_tool_calls() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content.decode())
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": {"text": "spam"},
                            },
                        }
                    ],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    llm = OllamaChatLLM(client=client)
    turn = llm.respond(
        [{"role": "user", "content": "echo spam"}],
        tools=[EchoTool().spec()],
    )
    llm.close()

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["tools"][0]["function"]["name"] == "echo"
    assert body["stream"] is False
    assert turn.tool_calls == (ToolCall(name="echo", arguments={"text": "spam"}),)


def test_ollama_respond_parses_final_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Done."}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    llm = OllamaChatLLM(client=client)
    turn = llm.respond([{"role": "user", "content": "hi"}], tools=[])
    llm.close()
    assert turn == AgentTurn(content="Done.")


def test_run_agent_with_ollama_tool_protocol() -> None:
    """End-to-end: loop + OllamaChatLLM.respond over mocked HTTP."""
    round_num = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        round_num["n"] += 1
        if round_num["n"] == 1:
            assert body["tools"][0]["function"]["name"] == "echo"
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "echo",
                                    "arguments": {"text": "spam"},
                                }
                            }
                        ],
                    }
                },
            )

        roles = [message["role"] for message in body["messages"]]
        assert roles[-2:] == ["assistant", "tool"]
        assert body["messages"][-1]["tool_name"] == "echo"
        assert body["messages"][-1]["content"] == "echo:spam"
        assert body["messages"][-2]["tool_calls"][0]["function"]["name"] == "echo"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Heard spam."}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    llm = OllamaChatLLM(client=client)
    result = run_agent("Where is spam?", llm=llm, tools=[EchoTool()])
    llm.close()

    assert result.answer == "Heard spam."
    assert result.llm_calls == 2
    assert [event.kind for event in result.events] == [
        "tool_call",
        "observation",
        "final",
    ]


@pytest.mark.integration
def test_live_ollama_tool_call_smoke() -> None:
    """Optional: requires Ollama + a tool-capable chat model.

    Example::

        ollama pull qwen2.5-coder:3b
        LIVE_OLLAMA=1 PYTHONPATH=. pytest tests/test_ollama_tools.py -m integration -v
    """
    if os.environ.get("LIVE_OLLAMA") != "1":
        pytest.skip("Set LIVE_OLLAMA=1 to hit a real Ollama chat model")

    with OllamaChatLLM() as llm:
        turn = llm.respond(
            [
                {
                    "role": "user",
                    "content": (
                        "Call the echo tool with text='ping'. "
                        "Do not answer in plain text first."
                    ),
                }
            ],
            tools=[EchoTool().spec()],
        )

    assert turn.tool_calls or turn.content.strip()
    if turn.tool_calls:
        assert turn.tool_calls[0].name == "echo"
