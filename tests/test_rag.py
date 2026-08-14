"""Unit tests for prompt construction and Ollama chat client (mocked)."""

from __future__ import annotations

import os

import httpx
import pytest

from app.llm.ollama_chat import OllamaChatLLM
from app.llm.prompt import build_rag_messages, format_context
from app.retrieval.rag import RagAnswer, ask
from app.retrieval.vector_store import ScoredChunk


def _chunk(
    path: str = "lib/scoring.py",
    start: int = 44,
    end: int = 60,
    text: str = "def compute_genuineness(job):\n    return score",
    score: float = 0.91,
) -> ScoredChunk:
    return ScoredChunk(
        path=path,
        start_line=start,
        end_line=end,
        text=text,
        score=score,
    )


def test_format_context_includes_path_and_lines() -> None:
    rendered = format_context([_chunk()])
    assert "lib/scoring.py:44-60" in rendered
    assert "compute_genuineness" in rendered
    assert "score=0.9100" in rendered


def test_format_context_includes_symbol_label() -> None:
    chunk = ScoredChunk(
        path="src/scoring.py",
        start_line=1,
        end_line=8,
        text="def compute_genuineness(job):\n    return score",
        score=0.91,
        language="python",
        symbol="compute_genuineness",
        kind="function",
        name="compute_genuineness",
    )
    rendered = format_context([chunk])
    assert "src/scoring.py:1-8 compute_genuineness (function)" in rendered


def test_format_context_empty() -> None:
    assert "no code context" in format_context([])


def test_build_rag_messages_contains_question_and_context() -> None:
    system, user = build_rag_messages(
        "How do we detect spam jobs?",
        [_chunk()],
    )
    assert "ONLY the provided code" in system or "code context" in system.lower()
    assert "How do we detect spam jobs?" in user
    assert "lib/scoring.py:44-60" in user
    assert "Answer:" in user


def test_build_rag_messages_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question"):
        build_rag_messages("   ", [_chunk()])


def test_ollama_chat_complete_parses_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = request.read()
        assert b"qwen2.5-coder:3b" in body
        assert b"stream" in body
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "It is computed in compute_genuineness.",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    llm = OllamaChatLLM(client=client)

    answer = llm.complete(system="Be brief.", user="Where is spam detected?")

    assert "compute_genuineness" in answer
    assert llm.model_name == "qwen2.5-coder:3b"
    llm.close()


def test_ollama_chat_missing_model_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    llm = OllamaChatLLM(model="missing-chat", client=client)

    with pytest.raises(RuntimeError, match="ollama pull missing-chat"):
        llm.complete(system="sys", user="user")
    llm.close()


def test_ollama_chat_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    llm = OllamaChatLLM(client=client)

    with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
        llm.complete(system="sys", user="user")
    llm.close()


class _FakeEmbedder:
    model_name = "fake"
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class _FakeStore:
    def search(self, query_vector: list[float], *, limit: int = 5) -> list[ScoredChunk]:
        assert len(query_vector) == 4
        return [_chunk()][:limit]


class _FakeLLM:
    model_name = "fake-chat"

    def complete(self, *, system: str, user: str) -> str:
        assert "compute_genuineness" in user
        return "Spam detection lives in compute_genuineness (lib/scoring.py:44-60)."


def test_ask_returns_answer_and_sources() -> None:
    result = ask(
        "How do we detect spam?",
        embedder=_FakeEmbedder(),
        store=_FakeStore(),  # type: ignore[arg-type]
        llm=_FakeLLM(),
        limit=3,
    )

    assert isinstance(result, RagAnswer)
    assert "compute_genuineness" in result.answer
    assert result.sources[0].path == "lib/scoring.py"
    assert result.question == "How do we detect spam?"


def test_ask_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question"):
        ask(
            " ",
            embedder=_FakeEmbedder(),
            store=_FakeStore(),  # type: ignore[arg-type]
            llm=_FakeLLM(),
        )


@pytest.mark.integration
def test_live_ollama_chat_smoke() -> None:
    """Optional: requires Ollama + a pulled chat model.

    Example::

        ollama pull qwen2.5-coder:3b
        LIVE_OLLAMA=1 PYTHONPATH=. pytest tests/test_rag.py -m integration -v
    """
    if os.environ.get("LIVE_OLLAMA") != "1":
        pytest.skip("Set LIVE_OLLAMA=1 to hit a real Ollama chat model")

    with OllamaChatLLM() as llm:
        answer = llm.complete(
            system="Reply with one short sentence.",
            user="Say hello to a codebase RAG learner.",
        )

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
