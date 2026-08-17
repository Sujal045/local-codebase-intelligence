"""Qdrant-backed vector store for code chunks.

Conceptually, Qdrant is a specialized database for vectors:

1. **Collection** — like a table; configured with vector size + distance metric.
2. **Point** — one stored item: ``id`` + ``vector`` + optional ``payload`` (metadata).
3. **Search** — given a query vector, return the nearest points by distance.

We do not implement ANN (approximate nearest neighbor) ourselves; Qdrant
maintains an index (e.g. HNSW) so search stays fast as data grows.

For Version 1 we use cosine distance, which matches how we compare semantic
similarity when embedding vectors are normalized.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.indexing.chunker import Chunk

DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION_NAME = "code_chunks"


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk returned from vector search with a similarity score."""

    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    language: str | None = None
    symbol: str | None = None
    kind: str | None = None
    name: str | None = None
    parent: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float) -> ScoredChunk:
        return cls(
            path=str(payload["path"]),
            start_line=int(payload["start_line"]),
            end_line=int(payload["end_line"]),
            text=str(payload["text"]),
            score=score,
            language=_optional_str(payload, "language"),
            symbol=_optional_str(payload, "symbol"),
            kind=_optional_str(payload, "kind"),
            name=_optional_str(payload, "name"),
            parent=_optional_str(payload, "parent"),
        )

    def to_chunk(self) -> Chunk:
        return Chunk(
            text=self.text,
            path=self.path,
            start_line=self.start_line,
            end_line=self.end_line,
            language=self.language,
            symbol=self.symbol,
            kind=self.kind,
            name=self.name,
            parent=self.parent,
        )

    def label(self) -> str:
        """Human-readable location, with symbol when Qdrant stored one."""
        location = f"{self.path}:{self.start_line}-{self.end_line}"
        if not self.symbol:
            return location
        if self.kind:
            return f"{location} {self.symbol} ({self.kind})"
        return f"{location} {self.symbol}"


class QdrantVectorStore:
    """Store chunk embeddings in Qdrant and run similarity search.

    Args:
        collection_name: Qdrant collection to read/write.
        vector_size: Length of embedding vectors (768 for nomic-embed-text).
        url: Qdrant HTTP URL, or ``":memory:"`` for in-process tests.
        client: Optional injected client (for tests).
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        vector_size: int,
        url: str = DEFAULT_QDRANT_URL,
        client: QdrantClient | None = None,
    ) -> None:
        if vector_size < 1:
            raise ValueError(f"vector_size must be >= 1, got {vector_size}")

        self.collection_name = collection_name
        self.vector_size = vector_size
        self._url = url
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self._url == ":memory:":
                self._client = QdrantClient(":memory:")
            else:
                self._client = QdrantClient(url=self._url)
        return self._client

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection if missing (or drop + recreate when requested)."""
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Store chunks with their embedding vectors.

        Point ids are deterministic from ``path:start_line:end_line`` so the
        same chunk location can be upserted again during re-indexing later.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors length mismatch: {len(chunks)} vs {len(vectors)}"
            )
        if not chunks:
            return 0

        points: list[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Vector for {chunk.path}:{chunk.start_line} has size "
                    f"{len(vector)}, expected {self.vector_size}"
                )
            points.append(
                qmodels.PointStruct(
                    id=_chunk_point_id(chunk),
                    vector=vector,
                    payload=_chunk_payload(chunk),
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def list_chunks(self, *, page_size: int = 128) -> list[Chunk]:
        """Return every stored chunk payload (no vectors).

        Ask-time BM25 rebuilds from this list so lexical search and vector
        search share one corpus: whatever ``index`` last upserted.
        """
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")

        chunks: list[Chunk] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload is None:
                    continue
                chunks.append(ScoredChunk.from_payload(point.payload, score=0.0).to_chunk())
            if offset is None:
                break
        return chunks

    def search(self, query_vector: list[float], *, limit: int = 5) -> list[ScoredChunk]:
        """Return the nearest chunks to ``query_vector`` (higher score = closer)."""
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"query_vector size {len(query_vector)} != {self.vector_size}"
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        results: list[ScoredChunk] = []
        for hit in response.points:
            if hit.payload is None:
                continue
            results.append(ScoredChunk.from_payload(hit.payload, score=hit.score))
        return results

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection_name))

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return int(info.points_count)

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> QdrantVectorStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    """Vector is stored separately; payload is everything we want back at query time.

    Optional symbol fields are omitted when None so naive Version 1 chunks
    and code-aware chunks can share one collection.
    """
    payload: dict[str, Any] = {
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
    }
    if chunk.language is not None:
        payload["language"] = chunk.language
    if chunk.symbol is not None:
        payload["symbol"] = chunk.symbol
    if chunk.kind is not None:
        payload["kind"] = chunk.kind
    if chunk.name is not None:
        payload["name"] = chunk.name
    if chunk.parent is not None:
        payload["parent"] = chunk.parent
    return payload


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value)
    return text or None


def _chunk_point_id(chunk: Chunk) -> str:
    key = f"{chunk.path}:{chunk.start_line}:{chunk.end_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
