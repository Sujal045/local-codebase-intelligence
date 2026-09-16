"""Agent loop: decide → maybe run a tool → observe → decide again (Slice 6A).

One-shot RAG (``ask()``) always retrieves, then generates. An agent
*chooses* whether to call a tool:

    ┌──────────────┐
    │ User question│
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │     LLM      │
    └──────┬───────┘
           │
     tool call? ──no──► final answer
           │ yes
           ▼
    execute tool.run(**args)
           │
           ▼
    dedupe + compress observation (Slice 7C)
           │
           ▼
    observation appended to messages
           │
           └──────────► LLM  (until no tool, or max_steps)

``max_steps`` is the maximum number of LLM rounds. That is the main
guard against infinite tool loops.

Tests can inject a scripted ``ToolCallingLLM``. Production uses
``OllamaChatLLM.respond`` (Slice 6B). The CLI ``agent`` command (Slice 6C)
builds tools and calls this loop; ``ask`` remains one-shot RAG.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.prompt import DEFAULT_AGENT_SYSTEM_PROMPT
from app.agent.types import (
    AgentAnswer,
    AgentEvent,
    AgentObservationStats,
    AgentTurn,
    ToolCall,
    ToolCallingLLM,
)
from app.context.budget import ContextBudget
from app.context.observations import AgentContextState, prepare_tool_observation
from app.tools.base import Tool, ToolResult

DEFAULT_MAX_STEPS = 8
STOPPED_FINAL = "final_answer"
STOPPED_LIMIT = "max_steps"


def run_agent(
    question: str,
    *,
    llm: ToolCallingLLM,
    tools: Sequence[Tool],
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT,
    observation_budget: ContextBudget | None = None,
) -> AgentAnswer:
    """Run the tool loop until a final answer or ``max_steps`` LLM rounds.

    Args:
        observation_budget: When set, dedupe chunk hits the model already
            saw and compress each tool observation to fit the remaining
            budget across rounds. Omit for full tool output (default).
    """
    if not question.strip():
        raise ValueError("question must be non-empty")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if not system_prompt.strip():
        raise ValueError("system_prompt must be non-empty")

    catalog = _tool_catalog(tools)
    specs = [tool.spec() for tool in catalog.values()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": question.strip()},
    ]
    events: list[AgentEvent] = []
    llm_calls = 0
    context_state = AgentContextState()

    for _ in range(max_steps):
        turn = llm.respond(messages, tools=specs)
        llm_calls += 1
        messages.append(_assistant_message(turn))

        if not turn.tool_calls:
            answer = turn.content.strip() or "(empty assistant message)"
            events.append(AgentEvent(kind="final", content=answer))
            return _finish(
                question=question.strip(),
                answer=answer,
                events=events,
                llm_calls=llm_calls,
                stopped_reason=STOPPED_FINAL,
                context_state=context_state,
                observation_budget=observation_budget,
            )

        for call in turn.tool_calls:
            events.append(
                AgentEvent(
                    kind="tool_call",
                    name=call.name,
                    content=_format_arguments(call.arguments),
                )
            )
            result = _execute_tool(catalog, call)
            prepared = prepare_tool_observation(
                result,
                state=context_state,
                budget=observation_budget,
            )
            events.append(
                AgentEvent(
                    kind="observation",
                    name=result.name,
                    content=prepared.content,
                )
            )
            messages.append(
                _tool_message(
                    ToolResult(name=result.name, content=prepared.content)
                )
            )

    return _finish(
        question=question.strip(),
        answer=f"Stopped after {max_steps} LLM rounds without a final answer.",
        events=events,
        llm_calls=llm_calls,
        stopped_reason=STOPPED_LIMIT,
        context_state=context_state,
        observation_budget=observation_budget,
    )


def _finish(
    *,
    question: str,
    answer: str,
    events: tuple[AgentEvent, ...] | list[AgentEvent],
    llm_calls: int,
    stopped_reason: str,
    context_state: AgentContextState,
    observation_budget: ContextBudget | None,
) -> AgentAnswer:
    stats: AgentObservationStats | None = None
    if observation_budget is not None:
        stats = AgentObservationStats(
            tokens_used=context_state.observation_tokens,
            compressed=context_state.compressed,
            chunks_deduped=context_state.chunks_deduped,
        )
    return AgentAnswer(
        question=question,
        answer=answer,
        events=tuple(events),
        llm_calls=llm_calls,
        stopped_reason=stopped_reason,
        observation_stats=stats,
    )


def _tool_catalog(tools: Sequence[Tool]) -> dict[str, Tool]:
    catalog: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in catalog:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        catalog[tool.name] = tool
    return catalog


def _execute_tool(catalog: dict[str, Tool], call: ToolCall) -> ToolResult:
    tool = catalog.get(call.name)
    if tool is None:
        available = ", ".join(sorted(catalog)) or "(none)"
        return ToolResult(
            name=call.name,
            content=(
                f"error: unknown tool {call.name!r}. "
                f"Available: {available}"
            ),
        )
    try:
        return tool.run(**call.arguments)
    except TypeError as exc:
        return ToolResult(name=call.name, content=f"error: {exc}")
    except Exception as exc:
        return ToolResult(name=call.name, content=f"error: {exc}")


def _assistant_message(turn: AgentTurn) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content,
    }
    if turn.tool_calls:
        message["tool_calls"] = [
            {"name": call.name, "arguments": call.arguments}
            for call in turn.tool_calls
        ]
    return message


def _tool_message(result: ToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": result.name,
        "content": result.content,
    }


def _format_arguments(arguments: dict[str, Any]) -> str:
    parts = [f"{key}={arguments[key]!r}" for key in sorted(arguments)]
    return ", ".join(parts) if parts else "(no arguments)"
