"""End-to-end ask path for hybrid Code RAG (Slice 3C).

Flow:

    question
      → rebuild BM25 from Qdrant payloads
      → hybrid_search (vector top-N + BM25 top-N → RRF top-k)
      → build prompt
      → LLM complete
      → RagAnswer(answer, sources)

Slice 4B added ``search_and_rerank``; this function still skips it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import DEFAULT_CANDIDATE_LIMIT
from app.embeddings import Embedder
from app.llm import ChatLLM
from app.llm.prompt import build_rag_messages
from app.retrieval.bm25 import Bm25Index
from app.retrieval.hybrid import hybrid_search
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
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    bm25: Bm25Index | None = None,
) -> RagAnswer:
    """Retrieve with hybrid search and generate an answer.

    Args:
        question: Natural-language question about the indexed codebase.
        embedder: Used to embed the question for vector search.
        store: Qdrant store that already contains indexed chunks.
        llm: Chat model that generates the final answer.
        limit: How many fused chunks to send to the LLM (RRF top-k).
        candidate_limit: How many hits to request from *each* retriever
            before fusion. The actual pool is ``max(candidate_limit, limit)``.
        bm25: Optional prebuilt lexical index. When omitted, rebuilt from
            ``store.list_chunks()`` so a one-shot CLI ask stays consistent
            with the last ``index`` run.
    """
    if not question.strip():
        raise ValueError("question must be non-empty")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if candidate_limit < 1:
        raise ValueError(f"candidate_limit must be >= 1, got {candidate_limit}")

    pool = max(candidate_limit, limit)
    lexical = bm25 if bm25 is not None else Bm25Index.from_store(store)
    sources = hybrid_search(
        question.strip(),
        embedder=embedder,
        store=store,
        bm25=lexical,
        vector_limit=pool,
        bm25_limit=pool,
        limit=limit,
    )
    system, user = build_rag_messages(question, sources)
    answer = llm.complete(system=system, user=user)
    return RagAnswer(question=question.strip(), answer=answer, sources=sources)
