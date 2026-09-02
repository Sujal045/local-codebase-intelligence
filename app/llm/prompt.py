"""Build the RAG prompt from a question and retrieved code chunks.

Prompt construction is where retrieval becomes generation:

    question + top-k chunks  →  system/user strings  →  LLM

Include path, line numbers, and symbol names when retrieval stored them.
When ``context_budget`` is set, chunks are deduplicated, reordered for
lost-in-the-middle attention, and truncated to fit the budget (Slice 7B).
"""

from __future__ import annotations

from app.context.budget import ContextBudget
from app.context.builder import BLOCK_SEPARATOR, ContextBuilder, format_chunk_block
from app.retrieval.vector_store import ScoredChunk

DEFAULT_SYSTEM_PROMPT = """\
You are a local codebase assistant. Answer using ONLY the provided code \
context. If the context is insufficient, say what is missing. Cite file \
paths, line ranges, and symbol names from the context when possible.\
"""


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as readable context blocks (no budgeting)."""
    if not chunks:
        return "(no code context retrieved)"

    blocks = [
        format_chunk_block(index, chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]
    return BLOCK_SEPARATOR.join(blocks)


def build_rag_messages(
    question: str,
    chunks: list[ScoredChunk],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context_budget: ContextBudget | None = None,
    context_text: str | None = None,
) -> tuple[str, str]:
    """Return ``(system, user)`` messages for the chat LLM.

    Args:
        question: User question about the codebase.
        chunks: Ranked retrieval hits (best first).
        system_prompt: System instructions for the chat model.
        context_budget: When set, dedupe, reorder, truncate, and drop chunks
            to fit ``budget.available`` tokens. When omitted, all chunks are
            formatted without budgeting (Version 1–6 behaviour).
        context_text: Pre-rendered context block. When set, used as-is
            (``ask()`` passes this after ``ContextBuilder.build`` to avoid
            building twice).

    Raises:
        ValueError: if ``question`` is empty/whitespace.
    """
    if not question.strip():
        raise ValueError("question must be non-empty")

    if context_text is not None:
        context = context_text
    elif context_budget is None:
        context = format_context(chunks)
    else:
        context = ContextBuilder().build(chunks, context_budget).text

    user = (
        f"Code context:\n\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer:"
    )
    return system_prompt, user
