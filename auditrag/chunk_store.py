"""SQLite chunk registry: the canonical source of truth for chunk content.

Citation resolution, the BM25 corpus, and evidence reports all read from this
store — never from the vector database. This keeps the provenance chain
independent of ChromaDB internals and makes the vector store swappable.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from auditrag.models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    pages       INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    doc_name    TEXT NOT NULL,
    page        INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


class ChunkStore:
    """Typed wrapper around the SQLite chunk registry.

    Usable as a context manager::

        with ChunkStore(path) as store:
            store.upsert_chunks(chunks)
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open (and create if needed) the registry at ``db_path``."""
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> "ChunkStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def find_doc_by_path(self, path: str) -> tuple[str, str] | None:
        """Look up a previously ingested document by file path.

        Args:
            path: Absolute path of the source file.

        Returns:
            ``(doc_id, sha256)`` if the path was ingested before, else ``None``.
        """
        row = self._conn.execute(
            "SELECT doc_id, sha256 FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def upsert_document(
        self, doc_id: str, name: str, path: str, sha256: str, pages: int
    ) -> None:
        """Insert or replace a document record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, name, path, sha256, pages, ingested_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, name, path, sha256, pages, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and all of its chunks (used on re-ingest)."""
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or replace a batch of chunks."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks"
            " (chunk_id, doc_id, doc_name, page, chunk_index, text, start_char, end_char)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.doc_name,
                    c.page,
                    c.chunk_index,
                    c.text,
                    c.start_char,
                    c.end_char,
                )
                for c in chunks
            ],
        )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Fetch a single chunk by its canonical ID."""
        row = self._conn.execute(
            "SELECT chunk_id, doc_id, doc_name, page, chunk_index, text, start_char, end_char"
            " FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return Chunk(
            chunk_id=row[0],
            doc_id=row[1],
            doc_name=row[2],
            page=row[3],
            chunk_index=row[4],
            text=row[5],
            start_char=row[6],
            end_char=row[7],
        )

    def count_chunks(self) -> int:
        """Total number of chunks in the registry."""
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_documents(self) -> int:
        """Total number of documents in the registry."""
        return int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
