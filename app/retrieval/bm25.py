"""Okapi BM25 lexical search over code chunks (Slice 3A).

Vector search (Phase 1) compares *meanings* in embedding space. It can miss
an exact identifier: the query ``compute_genuineness`` is semantically close
to many scoring functions, so the right symbol may not be top-1.

BM25 is the opposite idea: count how often query *terms* appear in a
document, weighted so rare terms (identifiers) matter more than common
ones (``def``, ``return``).

    TF   — term frequency: how often a term occurs in this chunk
    IDF  — inverse document frequency: rarer in the corpus → higher weight
    |D|  — document length (token count); BM25 down-weights very long chunks
    avgdl — average |D| in the corpus

This module does not talk to Qdrant or the LLM by itself. Slice 3C rebuilds
a ``Bm25Index`` from Qdrant payloads (``from_store``) at ask time so the CLI
does not need a second on-disk index.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from app.indexing.chunker import Chunk
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk

# Okapi BM25 defaults (Robertson et al.). k1 saturates TF; b mixes in length.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Turn source into lowercase tokens useful for code search.

    Each identifier is kept whole *and* split on ``_`` / camelCase so that
    ``compute_genuineness`` matches both the exact name and ``genuineness``.
    """
    if not text:
        return []

    tokens: list[str] = []
    for ident in _IDENTIFIER.findall(text):
        tokens.extend(_identifier_tokens(ident))
    return tokens


def _identifier_tokens(ident: str) -> list[str]:
    parts: list[str] = []
    lower = ident.lower()
    if len(lower) >= 2:
        parts.append(lower)
    if "_" in ident:
        parts.extend(p.lower() for p in ident.split("_") if len(p) >= 2)
    camel = _CAMEL_PART.findall(ident)
    if len(camel) > 1:
        parts.extend(p.lower() for p in camel if len(p) >= 2)
    return parts


def document_text(chunk: Chunk) -> str:
    """Text BM25 indexes: path + symbol names + body.

    Path and symbol are duplicated into the document so a query for
    ``QdrantVectorStore.search`` can hit metadata even if a class header
    chunk does not contain the method name.
    """
    extras = [chunk.path]
    if chunk.symbol:
        extras.append(chunk.symbol)
    if chunk.name and chunk.name not in extras:
        extras.append(chunk.name)
    extras.append(chunk.text)
    return " ".join(extras)


class Bm25Index:
    """In-memory BM25 index over a list of chunks.

    Args:
        chunks: Optional corpus; passed to ``build``.
        k1: TF saturation (typical 1.2–2.0). Higher → more credit for repeats.
        b: Length normalization (0 = ignore length, 1 = full normalization).
    """

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")

        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        self._df: dict[str, int] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._n = 0

        if chunks:
            self.build(chunks)

    @classmethod
    def from_store(
        cls,
        store: QdrantVectorStore,
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> Bm25Index:
        """Build BM25 over every chunk currently stored in Qdrant."""
        return cls(store.list_chunks(), k1=k1, b=b)

    def __len__(self) -> int:
        return self._n

    def build(self, chunks: list[Chunk]) -> None:
        """Replace the corpus and rebuild DF / postings."""
        self._chunks = list(chunks)
        self._n = len(self._chunks)
        self._doc_len = []
        df: dict[str, int] = defaultdict(int)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_id, chunk in enumerate(self._chunks):
            tokens = tokenize(document_text(chunk))
            self._doc_len.append(len(tokens))
            tf: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            for term, count in tf.items():
                df[term] += 1
                postings[term].append((doc_id, count))

        self._df = dict(df)
        self._postings = dict(postings)
        total_len = sum(self._doc_len)
        self._avgdl = (total_len / self._n) if self._n else 0.0

    def search(self, query: str, *, limit: int = 5) -> list[ScoredChunk]:
        """Return the highest-scoring chunks for ``query`` (BM25, desc)."""
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if not query.strip() or self._n == 0:
            return []

        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf(term)
            if idf == 0:
                continue
            for doc_id, tf in self._postings.get(term, []):
                scores[doc_id] += idf * self._tf_norm(tf, self._doc_len[doc_id])

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._chunks[item[0]].path, self._chunks[item[0]].start_line),
        )
        hits: list[ScoredChunk] = []
        for doc_id, score in ranked[:limit]:
            if score <= 0:
                continue
            hits.append(_scored_chunk(self._chunks[doc_id], score))
        return hits

    def _idf(self, term: str) -> float:
        """Lucene-style positive IDF: ``log(1 + (N - n + 0.5) / (n + 0.5))``."""
        n_t = self._df.get(term, 0)
        if n_t == 0:
            return 0.0
        return math.log(1.0 + (self._n - n_t + 0.5) / (n_t + 0.5))

    def _tf_norm(self, tf: int, doc_len: int) -> float:
        denom_len = self._avgdl if self._avgdl > 0 else 1.0
        denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / denom_len)
        if denom == 0:
            return 0.0
        return (tf * (self.k1 + 1.0)) / denom


def _scored_chunk(chunk: Chunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        text=chunk.text,
        score=score,
        language=chunk.language,
        symbol=chunk.symbol,
        kind=chunk.kind,
        name=chunk.name,
        parent=chunk.parent,
    )
