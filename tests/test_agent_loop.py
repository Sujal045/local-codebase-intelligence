"""Tests for the agent loop (Slice 6A; observation budget in 7C)."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent import (
    AgentTurn,
    ToolCall,
    run_agent,
)
from app.context import ContextBudget
from app.retrieval.vector_store import ScoredChunk
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


class ChunkHitsTool:
    """Returns fixed ScoredChunk hits for session dedupe tests."""

    name = "chunk_hits"

    def __init__(self, hits: list[ScoredChunk]) -> None:
        self._hits = hits

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def run(self, **arguments: Any) -> ToolResult:
        from app.llm.prompt import format_context

        return ToolResult(
            name=self.name,
            content=format_context(self._hits),
            hits=tuple(self._hits),
        )


def test_run_agent_dedupes_duplicate_chunk_hits_across_tools() -> None:
    hit = ScoredChunk(
        path="src/scoring.py",
        start_line=1,
        end_line=8,
        text="def compute_genuineness(job):\n    return score",
        score=0.95,
        symbol="compute_genuineness",
        kind="function",
    )
    tool = ChunkHitsTool([hit])
    llm = ScriptedLLM(
        [
            AgentTurn(tool_calls=(ToolCall(name="chunk_hits", arguments={}),)),
            AgentTurn(tool_calls=(ToolCall(name="chunk_hits", arguments={}),)),
            AgentTurn(content="Done."),
        ]
    )
    result = run_agent("find spam", llm=llm, tools=[tool])

    observations = [event.content for event in result.events if event.kind == "observation"]
    assert "compute_genuineness" in observations[0]
    assert "Already shown" in observations[1]
    assert result.observation_stats is not None
    assert result.observation_stats.chunks_deduped >= 1


def test_run_agent_compresses_observations_with_budget() -> None:
    huge = "x\n" * 400
    llm = ScriptedLLM(
        [
            AgentTurn(tool_calls=(ToolCall(name="echo", arguments={"text": huge}),)),
            AgentTurn(content="Done."),
        ]
    )
    budget = ContextBudget(max_tokens=30, reserved_tokens=0)
    result = run_agent(
        "big",
        llm=llm,
        tools=[EchoTool()],
        observation_budget=budget,
    )

    observation = next(
        event.content for event in result.events if event.kind == "observation"
    )
    assert result.observation_stats is not None
    assert result.observation_stats.tokens_used <= 30
    assert result.observation_stats.compressed >= 1
    assert "[truncated]" in observation

    second_messages = llm.calls[1]
    assert "[truncated]" in second_messages[-1]["content"]


def test_run_agent_no_dedupe_keeps_duplicate_bodies() -> None:
    hit = ScoredChunk(
        path="src/scoring.py",
        start_line=1,
        end_line=8,
        text="def compute_genuineness(job):\n    return score",
        score=0.95,
        symbol="compute_genuineness",
        kind="function",
    )
    tool = ChunkHitsTool([hit])
    llm = ScriptedLLM(
        [
            AgentTurn(tool_calls=(ToolCall(name="chunk_hits", arguments={}),)),
            AgentTurn(tool_calls=(ToolCall(name="chunk_hits", arguments={}),)),
            AgentTurn(content="Done."),
        ]
    )
    result = run_agent(
        "find spam",
        llm=llm,
        tools=[tool],
        dedupe_observations=False,
    )
    observations = [event.content for event in result.events if event.kind == "observation"]
    assert "Already shown" not in observations[1]
    assert "compute_genuineness" in observations[1]
    assert result.observation_stats is None

