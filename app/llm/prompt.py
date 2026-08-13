"""Build the RAG prompt from a question and retrieved code chunks.

Prompt construction is where retrieval becomes generation:

    question + top-k chunks  →  system/user strings  →  LLM

We keep Version 1 simple: include path + line numbers + code, and ask the
model to cite those paths. Token budgeting and compression come later
(Version 7).
"""

from __future__ import annotations

from app.retrieval.vector_store import ScoredChunk

DEFAULT_SYSTEM_PROMPT = """\
You are a local codebase assistant. Answer using ONLY the provided code \
context. If the context is insufficient, say what is missing. Cite file \
paths and line ranges from the context when possible.\
"""


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as readable context blocks."""
    if not chunks:
        return "(no code context retrieved)"

    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] {chunk.path}:{chunk.start_line}-{chunk.end_line} "
            f"(score={chunk.score:.4f})"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_rag_messages(
    question: str,
    chunks: list[ScoredChunk],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str, str]:
    """Return ``(system, user)`` messages for the chat LLM.

    Raises:
        ValueError: if ``question`` is empty/whitespace.
    """
    if not question.strip():
        raise ValueError("question must be non-empty")

    context = format_context(chunks)
    user = (
        f"Code context:\n\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer:"
    )
    return system_prompt, user
