"""End-to-end ask path for Code RAG (Slice 4C; context budget in 7B).

Flow:

    question
      → retrieve_chunks (hybrid pool, optional rerank)
      → build prompt (optional context budget)
      → LLM complete
      → RagAnswer(answer, sources)

Slice 5A extracted ``retrieve_chunks`` so ``search_code`` can retrieve
without generating. ``ask()`` is still one-shot RAG, not an agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import DEFAULT_CANDIDATE_LIMIT
from app.context import BuiltContext, ContextBudget, ContextBuilder
from app.embeddings import Embedder
from app.llm import ChatLLM
from app.llm.prompt import build_rag_messages
from app.reranking.reranker import Reranker
from app.retrieval.bm25 import Bm25Index
from app.retrieval.query import retrieve_chunks
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk


@dataclass(frozen=True)
class RagAnswer:
    """LLM answer plus the chunks that were provided as context."""

    question: str
    answer: str
    sources: list[ScoredChunk]
    reranked: bool = False
    context: BuiltContext | None = None


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
    context_budget: ContextBudget | None = None,
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
        context_budget: When set, dedupe, reorder, truncate, and drop chunks
            so formatted context fits ``budget.available``. ``sources`` then
            lists only the chunks the LLM actually saw (bodies may be
            truncated). Omit for unbounded formatting (default).
    """
    if not question.strip():
        raise ValueError("question must be non-empty")

    retrieved = retrieve_chunks(
        question,
        embedder=embedder,
        store=store,
        limit=limit,
        candidate_limit=candidate_limit,
        bm25=bm25,
        reranker=reranker,
    )

    built: BuiltContext | None = None
    if context_budget is not None:
        built = ContextBuilder().build(retrieved.chunks, context_budget)
        prompt_chunks = list(built.chunks)
        system, user = build_rag_messages(
            question,
            prompt_chunks,
            context_text=built.text,
        )
    else:
        prompt_chunks = retrieved.chunks
        system, user = build_rag_messages(question, prompt_chunks)
    answer = llm.complete(system=system, user=user)
    return RagAnswer(
        question=question.strip(),
        answer=answer,
        sources=prompt_chunks,
        reranked=retrieved.reranked,
        context=built,
    )
