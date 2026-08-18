"""End-to-end ask path for Code RAG (Slice 4C).

Flow:

    question
      → rebuild BM25 from Qdrant payloads
      → hybrid_search (vector top-N + BM25 top-N → RRF pool)
      → rerank the pool (unless ``reranker`` is omitted)
      → build prompt
      → LLM complete
      → RagAnswer(answer, sources)

``ask()`` does not construct models. The CLI (``main``) injects a
``CrossEncoderReranker`` by default, or skips it with ``--no-rerank``.
Tests inject ``FakeReranker`` so they never load torch.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import DEFAULT_CANDIDATE_LIMIT
from app.embeddings import Embedder
from app.llm import ChatLLM
from app.llm.prompt import build_rag_messages
from app.reranking.pipeline import search_and_rerank
from app.reranking.reranker import Reranker
from app.retrieval.bm25 import Bm25Index
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk


@dataclass(frozen=True)
class RagAnswer:
    """LLM answer plus the chunks that were provided as context."""

    question: str
    answer: str
    sources: list[ScoredChunk]
    reranked: bool = False


def ask(
    question: str,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
    llm: ChatLLM,
    limit: int = 5,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    bm25: Bm25Index | None = None,
    reranker: Reranker | None = None,
) -> RagAnswer:
    """Retrieve, optionally rerank, and generate an answer.

    Args:
        question: Natural-language question about the indexed codebase.
        embedder: Used to embed the question for vector search.
        store: Qdrant store that already contains indexed chunks.
        llm: Chat model that generates the final answer.
        limit: How many chunks to send to the LLM after ranking.
        candidate_limit: Hybrid pool size (per retriever and RRF) before
            reranking. Ignored as a final cutoff when ``reranker`` is set;
            the reranker then keeps ``limit``.
        bm25: Optional prebuilt lexical index. When omitted, rebuilt from
            ``store.list_chunks()`` so a one-shot CLI ask stays consistent
            with the last ``index`` run.
        reranker: If given, scores the hybrid pool jointly with the query
            (Slice 4B). If omitted, the LLM sees Reciprocal Rank Fusion
            order (``--no-rerank``).
    """
    if not question.strip():
        raise ValueError("question must be non-empty")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if candidate_limit < 1:
        raise ValueError(f"candidate_limit must be >= 1, got {candidate_limit}")

    stripped = question.strip()
    lexical = bm25 if bm25 is not None else Bm25Index.from_store(store)
    if reranker is not None:
        sources = search_and_rerank(
            stripped,
            embedder=embedder,
            store=store,
            bm25=lexical,
            reranker=reranker,
            candidate_limit=candidate_limit,
            limit=limit,
        )
        reranked = True
    else:
        pool = max(candidate_limit, limit)
        sources = hybrid_search(
            stripped,
            embedder=embedder,
            store=store,
            bm25=lexical,
            vector_limit=pool,
            bm25_limit=pool,
            limit=limit,
        )
        reranked = False

    system, user = build_rag_messages(question, sources)
    answer = llm.complete(system=system, user=user)
    return RagAnswer(
        question=stripped,
        answer=answer,
        sources=sources,
        reranked=reranked,
    )
