"""ChromaDB wrapper.

The vector store holds embeddings keyed by canonical chunk IDs plus the
minimal metadata needed for filtered retrieval. Chunk *content* provenance
lives in the SQLite chunk registry (:mod:`auditrag.chunk_store`); this store
is deliberately treated as swappable infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction

from auditrag.models import Chunk


class VectorStore:
    """Persistent ChromaDB collection of chunk embeddings."""

    def __init__(
        self,
        path: str | Path,
        collection: str,
        embedding_function: EmbeddingFunction[Documents],
    ) -> None:
        """Open (and create if needed) the persistent collection.

        Args:
            path: Directory for ChromaDB's on-disk storage.
            collection: Collection name.
            embedding_function: Function used to embed documents and queries.
        """
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert a batch of chunks, keyed by canonical chunk ID."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        """Delete all embeddings belonging to a document (used on re-ingest)."""
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        """Number of embeddings currently in the collection."""
        return self._collection.count()
