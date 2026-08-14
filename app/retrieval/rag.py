"""End-to-end ask path for Version 1 RAG (Slice 1d).

Flow:

    question
      → embed
      → vector search (top-k)
      → build prompt
      → LLM complete
      → RagAnswer(answer, sources)
"""

from __future__ import annotations

from dataclasses import dataclass

from app.embeddings import Embedder
from app.llm import ChatLLM
from app.llm.prompt import build_rag_messages
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk


@dataclass(frozen=True)
class RagAnswer:
    """LLM answer plus the chunks that were provided as context."""

    question: str
    answer: str
    sources: list[ScoredChunk]


def ask(
    question: str,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
    llm: ChatLLM,
    limit: int = 5,
) -> RagAnswer:
    """Retrieve relevant chunks and generate an answer.

    Args:
        question: Natural-language question about the indexed codebase.
        embedder: Used to embed the question for vector search.
        store: Qdrant store that already contains indexed chunks.
        llm: Chat model that generates the final answer.
        limit: How many chunks to retrieve (top-k).
    """
    if not question.strip():
        raise ValueError("question must be non-empty")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    query_vector = embedder.embed_one(question.strip())
    sources = store.search(query_vector, limit=limit)
    system, user = build_rag_messages(question, sources)
    answer = llm.complete(system=system, user=user)
    return RagAnswer(question=question.strip(), answer=answer, sources=sources)
