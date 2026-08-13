"""Shared test doubles (no Ollama / Docker)."""

from __future__ import annotations

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
