"""Citation-tracked retrieval.

The vector store proposes chunk IDs; the SQLite chunk registry supplies the
content. Every retrieved chunk is hydrated from the registry — the canonical
source of truth — so downstream stages (generation, verification, evidence
reports) can never be fed text that differs from what a citation resolves to.

Hits whose registry record is missing (index/registry desync, e.g. a deleted
``chunks.db`` or an interrupted ingest) are skipped and surfaced as warnings
rather than served with unverifiable provenance.
"""

from __future__ import annotations

from chromadb.api.types import Documents, EmbeddingFunction

from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.embeddings import build_embedding_function
from auditrag.models import RetrievalResult, RetrievedChunk
from auditrag.vector_store import VectorStore


class RetrievalError(RuntimeError):
    """Retrieval failed in a way the caller should surface to the user.

    The message is always actionable (what went wrong and what to try),
    suitable for direct display in CLI output or an API error body.
    """


class EmptyIndexError(RetrievalError):
    """Raised when querying before any documents have been ingested."""


class Retriever:
    """Searches the vector index and hydrates hits from the chunk registry.

    A new SQLite connection is opened per search, so a single instance is
    safe to share across FastAPI's request threadpool.
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

    def search(self, query: str, top_k: int = 6) -> RetrievalResult:
        """Return the ``top_k`` most relevant chunks with full provenance.

        Args:
            query: Natural-language query text.
            top_k: Maximum number of chunks to return.

        Returns:
            Ranked chunks (highest similarity first) plus integrity warnings
            for any vector hits that could not be hydrated from the registry.

        Raises:
            EmptyIndexError: If no documents have been ingested yet.
            RetrievalError: If the embedding backend fails (unreachable
                endpoint, bad API key, unknown model, ...).
        """
        if self._vector_store.count() == 0:
            raise EmptyIndexError(
                "The index is empty — ingest documents first: auditrag ingest ./docs"
            )

        try:
            hits = self._vector_store.query(query, n_results=top_k)
        except EmptyIndexError:
            raise
        except Exception as exc:
            provider = self._settings.embedding.provider
            raise RetrievalError(
                f"Embedding the query failed (provider '{provider}'): {exc}. "
                "Check the embedding section of auditrag.yaml — endpoint URL, "
                "model name, and the API key environment variable."
            ) from exc

        chunks: list[RetrievedChunk] = []
        warnings: list[str] = []
        with ChunkStore(self._settings.chunk_db_path) as store:
            for chunk_id, distance in hits:
                chunk = store.get_chunk(chunk_id)
                if chunk is None:
                    warnings.append(
                        f"Chunk '{chunk_id}' is in the vector index but missing from "
                        "the chunk registry; skipped. The stores are out of sync — "
                        "re-run 'auditrag ingest' to rebuild."
                    )
                    continue
                chunks.append(
                    RetrievedChunk(chunk=chunk, score=1.0 - distance, rank=len(chunks))
                )

        return RetrievalResult(query=query, chunks=chunks, warnings=warnings)
