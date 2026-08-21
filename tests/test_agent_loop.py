"""Tests for the agent loop (Slice 6A)."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent import (
    AgentTurn,
    ToolCall,
    run_agent,
)
from app.tools.base import ToolResult


class EchoTool:
    name = "echo"

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    def run(self, **arguments: Any) -> ToolResult:
        return ToolResult(name=self.name, content=f"echo:{arguments.get('text', '')}")


class ScriptedLLM:
    """Plays back a list of turns; records every ``respond`` call."""

    def __init__(self, turns: list[AgentTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[list[dict[str, Any]]] = []
        self.tool_specs: list[dict[str, Any]] | None = None

    def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        self.calls.append(list(messages))
        self.tool_specs = tools
        if not self._turns:
            return AgentTurn(content="(script exhausted)")
        return self._turns.pop(0)


def test_run_agent_answers_without_tools() -> None:
    llm = ScriptedLLM([AgentTurn(content="Hello from the model.")])
    result = run_agent("hi", llm=llm, tools=[])
    assert result.answer == "Hello from the model."
    assert result.llm_calls == 1
    assert result.stopped_reason == "final_answer"
    assert result.events[-1].kind == "final"


def test_run_agent_executes_tool_then_answers() -> None:
    llm = ScriptedLLM(
        [
            AgentTurn(
                tool_calls=(ToolCall(name="echo", arguments={"text": "spam"}),)
            ),
            AgentTurn(content="Spam was echoed."),
        ]
    )
    result = run_agent("Where is spam?", llm=llm, tools=[EchoTool()])

    assert result.answer == "Spam was echoed."
    assert result.llm_calls == 2
    kinds = [event.kind for event in result.events]
    assert kinds == ["tool_call", "observation", "final"]
    assert result.events[0].name == "echo"
    assert "spam" in result.events[0].content
    assert result.events[1].content == "echo:spam"

    second_messages = llm.calls[1]
    roles = [message["role"] for message in second_messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second_messages[-1]["content"] == "echo:spam"
    assert llm.tool_specs is not None
    assert llm.tool_specs[0]["function"]["name"] == "echo"


def test_run_agent_unknown_tool_becomes_observation() -> None:
    llm = ScriptedLLM(
        [
            AgentTurn(tool_calls=(ToolCall(name="explode", arguments={}),)),
            AgentTurn(content="I could not call explode."),
        ]
    )
    result = run_agent("boom", llm=llm, tools=[EchoTool()])
    assert "unknown tool" in result.events[1].content
    assert result.answer == "I could not call explode."


def test_run_agent_stops_at_max_steps() -> None:
    llm = ScriptedLLM(
        [
            AgentTurn(tool_calls=(ToolCall(name="echo", arguments={"text": "a"}),)),
            AgentTurn(tool_calls=(ToolCall(name="echo", arguments={"text": "b"}),)),
        ]
    )
    result = run_agent("loop", llm=llm, tools=[EchoTool()], max_steps=2)
    assert result.stopped_reason == "max_steps"
    assert result.llm_calls == 2
    assert "Stopped after 2 LLM rounds" in result.answer
    assert result.events[-1].kind == "observation"


def test_run_agent_rejects_empty_question_and_dup_tools() -> None:
    llm = ScriptedLLM([AgentTurn(content="x")])
    with pytest.raises(ValueError, match="question"):
        run_agent("  ", llm=llm, tools=[])
    with pytest.raises(ValueError, match="duplicate tool"):
        run_agent("q", llm=llm, tools=[EchoTool(), EchoTool()])
    with pytest.raises(ValueError, match="max_steps"):
        run_agent("q", llm=llm, tools=[], max_steps=0)
