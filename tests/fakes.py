"""Shared test doubles (no Ollama / Docker)."""

from __future__ import annotations

from typing import Any

from app.agent.types import AgentTurn
from app.retrieval.vector_store import ScoredChunk


class FakeEmbedder:
    """Deterministic tiny vectors for unit tests."""

    model_name = "fake"
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 97
            out.append(
                [
                    float(seed % 7),
                    float((seed + 1) % 11),
                    float((seed + 2) % 13),
                    float((seed + 3) % 17),
                ]
            )
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FakeLLM:
    model_name = "fake-chat"

    def complete(self, *, system: str, user: str) -> str:
        return f"FAKE_ANSWER\n{user[:120]}"


class RecordingLLM(FakeLLM):
    def __init__(self) -> None:
        self.last_user = ""

    def complete(self, *, system: str, user: str) -> str:
        self.last_user = user
        return "Spam scoring is in compute_genuineness."


class FakeReranker:
    """Deterministic reranker: count query tokens inside chunk.text.

    This is not a cross-encoder. It exists so later slices can test the
    retrieve → rerank pipeline without loading torch or Hugging Face weights.
    """

    model_name = "fake-reranker"

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        *,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        from app.reranking import apply_scores

        stripped = query.strip()
        if not stripped:
            raise ValueError("query must be non-empty")
        tokens = [tok.lower() for tok in stripped.split() if tok]
        scores = [
            float(sum(chunk.text.lower().count(tok) for tok in tokens))
            for chunk in candidates
        ]
        return apply_scores(candidates, scores, limit=limit)


class ScriptedToolLLM:
    """Plays back ``AgentTurn`` values for CLI / agent tests (no Ollama)."""

    def __init__(self, turns: list[AgentTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[list[dict[str, Any]]] = []

    def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        self.calls.append(list(messages))
        if not self._turns:
            return AgentTurn(content="(script exhausted)")
        return self._turns.pop(0)


def fake_chunk(
    path: str = "src/scoring.py",
    start: int = 1,
    end: int = 8,
    text: str = "def compute_genuineness(job):\n    return score",
    score: float = 0.9,
) -> ScoredChunk:
    return ScoredChunk(
        path=path,
        start_line=start,
        end_line=end,
        text=text,
        score=score,
    )
