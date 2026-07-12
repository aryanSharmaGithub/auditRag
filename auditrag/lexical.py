"""In-memory BM25 index over the chunk registry.

The corpus comes from SQLite — the canonical chunk store — not from the
vector database, so lexical and vector retrieval can never disagree about
what a chunk ID means.

Known limitation (v1): the index is built in memory from the full corpus at
retriever construction; it is not persisted or incrementally updated. Two
consequences:

* Every CLI invocation (``ask``/``search``) rebuilds the index from scratch.
  Build cost is linear in corpus size — negligible for the local document
  sets AuditRAG targets, but noticeable at tens of thousands of chunks.
* A long-running ``auditrag serve`` builds the index once and caches it, so
  it will not reflect documents ingested after startup until restarted (see
  :class:`auditrag.retrieval.Retriever`).

Persisting the index to disk and updating it incrementally on ingest is
deferred until corpus size warrants it.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization; deliberately simple for v1."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """BM25 ranking over ``(chunk_id, text)`` entries."""

    def __init__(self, entries: list[tuple[str, str]]) -> None:
        """Build the index.

        Args:
            entries: ``(chunk_id, text)`` pairs for the whole corpus; an
                empty list yields an index that returns no results.
        """
        self._ids = [chunk_id for chunk_id, _ in entries]
        self._bm25 = (
            BM25Okapi([_tokenize(text) for _, text in entries]) if entries else None
        )

    def query(self, text: str, n_results: int) -> list[str]:
        """Return chunk IDs ranked by BM25 relevance to the query.

        Only chunks with a positive score (i.e. sharing at least one term
        with the query) are returned — BM25 rank noise on non-matching
        documents must not leak into fusion.

        Args:
            text: Query text.
            n_results: Maximum number of IDs to return.

        Returns:
            Chunk IDs in descending relevance order; possibly empty.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._ids, scores), key=lambda pair: pair[1], reverse=True)
        return [chunk_id for chunk_id, score in ranked[:n_results] if score > 0]
