"""Index a repository: walk → chunk → embed → upsert.

Suffixes with a Tree-sitter extractor (Python, JS/TS, Go) use symbol
chunks. Everything else still uses naive line windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.embeddings import Embedder
from app.indexing.chunker import Chunk, chunk_file
from app.indexing.code_chunker import chunk_code_file
from app.indexing.walker import SourceFile, walk_repository
from app.parsing.languages import language_for_path
from app.retrieval.vector_store import QdrantVectorStore

DEFAULT_EMBED_BATCH_SIZE = 16


@dataclass(frozen=True)
class IndexResult:
    """Summary of one indexing run."""

    root: str
    files: int
    chunks: int
    upserted: int


def collect_chunks(
    files: list[SourceFile],
    *,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    """Chunk every source file and flatten into one list.

    Parsed suffixes go through ``chunk_code_file`` (symbol metadata).
    Other suffixes keep Version 1 line windows so we still index Markdown,
    JSON, and languages we cannot parse yet.
    """
    chunks: list[Chunk] = []
    for source in files:
        chunks.extend(
            chunk_source_file(
                source,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks


def chunk_source_file(
    source: SourceFile,
    *,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    """Choose a chunker from the file suffix."""
    if language_for_path(source.path) is not None:
        return chunk_code_file(source.path, source.content)
    return chunk_file(
        source.path,
        source.content,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def index_chunks(
    store: QdrantVectorStore,
    embedder: Embedder,
    chunks: list[Chunk],
    *,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> int:
    """Embed chunks and upsert them into Qdrant.

    Embeddings are sent in batches so a large repo does not become one
    giant Ollama request.

    Returns the number of points written.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not chunks:
        return 0

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        vectors = embedder.embed(texts)
        total += store.upsert_chunks(batch, vectors)
    return total


def index_repository(
    root: str | Path,
    *,
    store: QdrantVectorStore,
    embedder: Embedder,
    chunk_size: int = 40,
    overlap: int = 10,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    recreate: bool = True,
) -> IndexResult:
    """Walk ``root``, chunk files, embed, and store in Qdrant."""
    root_path = Path(root).resolve()
    files = walk_repository(root_path)
    chunks = collect_chunks(files, chunk_size=chunk_size, overlap=overlap)
    store.ensure_collection(recreate=recreate)
    upserted = index_chunks(store, embedder, chunks, batch_size=batch_size)
    return IndexResult(
        root=str(root_path),
        files=len(files),
        chunks=len(chunks),
        upserted=upserted,
    )
