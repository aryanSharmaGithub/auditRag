"""Citation-tracked hybrid retrieval.

Two rankers propose chunk IDs — vector similarity (ChromaDB) and BM25 over
the registry corpus — and their rankings are combined with reciprocal rank
fusion. Fusion operates purely on chunk IDs, so it cannot mangle provenance;
the winners are then hydrated from the SQLite chunk registry, the canonical
source of truth. Downstream stages (generation, verification, evidence
reports) can never be fed text that differs from what a citation resolves to.

RRF uses the standard ``k=60`` constant, deliberately not configurable in v1:
it is the parameter-free reason RRF was chosen over score interpolation.

Hits whose registry record is missing (index/registry desync, e.g. a deleted
``chunks.db`` or an interrupted ingest) are skipped and surfaced as warnings
rather than served with unverifiable provenance.
"""

from __future__ import annotations

from collections.abc import Sequence

from chromadb.api.types import Documents, EmbeddingFunction

from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.embeddings import build_embedding_function
from auditrag.lexical import BM25Index
from auditrag.models import RetrievalResult, RetrievedChunk
from auditrag.vector_store import VectorStore

_RRF_K = 60
_CANDIDATE_POOL = 20  # per ranker, before fusion


def _rrf_fuse(rankings: Sequence[Sequence[str]], k: int = _RRF_K) -> list[tuple[str, float]]:
    """Fuse rankings with reciprocal rank fusion.

    Each document scores ``sum(1 / (k + rank_i + 1))`` over the rankings that
    contain it (rank is 0-based). Documents appearing in several rankings
    rise; ties broken by first appearance for determinism.

    Args:
        rankings: ID lists in descending relevance order, one per ranker.
        k: The RRF damping constant.

    Returns:
        ``(chunk_id, score)`` pairs, best first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class RetrievalError(RuntimeError):
    """Retrieval failed in a way the caller should surface to the user.

    The message is always actionable (what went wrong and what to try),
    suitable for direct display in CLI output or an API error body.
    """


class EmptyIndexError(RetrievalError):
    """Raised when querying before any documents have been ingested."""


class Retriever:
    """Hybrid (vector + BM25) search, hydrated from the chunk registry.

    The BM25 index is built from the registry once at construction; create
    a new retriever after ingesting to pick up new documents. A new SQLite
    connection is opened per search, so a single instance is safe to share
    across FastAPI's request threadpool.
    """

    def __init__(
        self,
        settings: Settings,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        """Create a retriever over the configured local stores.

        Args:
            settings: Loaded AuditRAG settings.
            embedding_function: Optional override, mainly for tests. When
                ``None`` the backend configured in ``settings.embedding``
                is used. Must match the function used at ingest time —
                querying with a different embedding space silently ruins
                relevance.
        """
        self._settings = settings
        if embedding_function is None:
            embedding_function = build_embedding_function(settings.embedding)
        try:
            self._vector_store = VectorStore(
                path=settings.chroma_path,
                collection=settings.storage.collection,
                embedding_function=embedding_function,
            )
        except ValueError as exc:
            # ChromaDB persists the embedding function's identity with the
            # collection and refuses to open it under a different one — the
            # typical cause is editing embedding config after ingesting.
            raise RetrievalError(
                f"Opening the vector index failed: {exc}. The embedding "
                "configuration appears to have changed since the documents "
                "were ingested. Either restore the previous embedding settings "
                f"in auditrag.yaml, or delete '{settings.data_dir}' and "
                "re-ingest with the new ones."
            ) from exc

        if settings.chunk_db_path.exists():
            with ChunkStore(settings.chunk_db_path) as store:
                corpus = store.all_chunk_texts()
        else:
            corpus = []
        self._bm25 = BM25Index(corpus)

    def search(self, query: str, top_k: int = 6) -> RetrievalResult:
        """Return the ``top_k`` most relevant chunks with full provenance.

        Vector and BM25 rankings are fused with RRF before hydration; a
        chunk found by either ranker can therefore appear in the results.

        Args:
            query: Natural-language query text.
            top_k: Maximum number of chunks to return.

        Returns:
            Ranked chunks (best fused score first) plus integrity warnings
            for any hits that could not be hydrated from the registry.

        Raises:
            EmptyIndexError: If no documents have been ingested yet.
            RetrievalError: If the embedding backend fails (unreachable
                endpoint, bad API key, unknown model, ...).
        """
        if self._vector_store.count() == 0:
            raise EmptyIndexError(
                "The index is empty — ingest documents first: auditrag ingest ./docs"
            )

        pool = max(_CANDIDATE_POOL, top_k)
        try:
            vector_hits = self._vector_store.query(query, n_results=pool)
        except EmptyIndexError:
            raise
        except Exception as exc:
            provider = self._settings.embedding.provider
            raise RetrievalError(
                f"Embedding the query failed (provider '{provider}'): {exc}. "
                "Check the embedding section of auditrag.yaml — endpoint URL, "
                "model name, and the API key environment variable."
            ) from exc

        vector_ranking = [chunk_id for chunk_id, _ in vector_hits]
        lexical_ranking = self._bm25.query(query, n_results=pool)
        fused = _rrf_fuse([vector_ranking, lexical_ranking])

        chunks: list[RetrievedChunk] = []
        warnings: list[str] = []
        with ChunkStore(self._settings.chunk_db_path) as store:
            for chunk_id, score in fused:
                if len(chunks) == top_k:
                    break
                chunk = store.get_chunk(chunk_id)
                if chunk is None:
                    warnings.append(
                        f"Chunk '{chunk_id}' was proposed by retrieval but is missing "
                        "from the chunk registry; skipped. The stores are out of sync "
                        "— re-run 'auditrag ingest' to rebuild."
                    )
                    continue
                chunks.append(RetrievedChunk(chunk=chunk, score=score, rank=len(chunks)))

        return RetrievalResult(query=query, chunks=chunks, warnings=warnings)
