"""Document ingestion: loading, chunking, and indexing.

The pipeline is::

    file → pages (pypdf, or whole file as page 1 for text formats)
         → within-page chunks with exact char offsets
         → SQLite chunk registry (canonical)
         → ChromaDB (embeddings, keyed by the same chunk IDs)

Chunks never cross page boundaries, so every chunk carries exactly one
unambiguous page number — a deliberate v1 trade of slightly worse retrieval
at page breaks for exact provenance.

Ingestion is idempotent: files are identified by content hash, so re-running
``auditrag ingest`` over unchanged files is a no-op, and changed files replace
their previous chunks in both stores.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from chromadb.api.types import Documents, EmbeddingFunction

from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.embeddings import build_embedding_function
from auditrag.models import (
    Chunk,
    Document,
    FileIngestResult,
    IngestResult,
    PageText,
    mint_chunk_id,
)

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".md", ".markdown", ".txt"})

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def load_document(path: str | Path) -> Document:
    """Load a source file into a :class:`Document` with per-page text.

    PDFs are split into their real pages via pypdf. Plain-text formats
    (``.txt``, ``.md``) become single-page documents so every chunk still has
    an exact page number.

    Args:
        path: Path to a supported file.

    Returns:
        The loaded document. Pages with no extractable text are preserved
        (empty string) so page numbering stays aligned with the source file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    file_path = Path(path).resolve()
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}': {file_path}")

    raw = file_path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    doc_id = sha256[:12]

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages = [
            PageText(page=i, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages, start=1)
        ]
    else:
        pages = [PageText(page=1, text=raw.decode("utf-8", errors="replace"))]

    return Document(
        doc_id=doc_id,
        name=file_path.name,
        path=str(file_path),
        sha256=sha256,
        pages=pages,
    )


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentence-ish spans, returning (start, end) offsets.

    Boundaries are sentence-ending punctuation followed by whitespace, or
    blank lines. Spans are trimmed of surrounding whitespace and always
    satisfy ``text[start:end].strip() == text[start:end]``.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    boundaries = [m.start() for m in _SENTENCE_BOUNDARY.finditer(text)] + [len(text)]
    for boundary in boundaries:
        segment = text[cursor:boundary]
        stripped = segment.strip()
        if stripped:
            start = cursor + segment.index(stripped[0])
            spans.append((start, start + len(stripped)))
        cursor = boundary
        # Skip past the whitespace that formed the boundary.
        while cursor < len(text) and text[cursor] in " \t\r\n":
            cursor += 1
    return spans


def chunk_page(page: PageText, doc: Document, max_chars: int) -> list[Chunk]:
    """Split one page into chunks of at most ``max_chars`` characters.

    Sentences are packed greedily; a single sentence longer than ``max_chars``
    is hard-split into windows. Each chunk records its exact character span
    within the page, so ``page.text[start_char:end_char] == chunk.text``.

    Args:
        page: The page to chunk.
        doc: The document the page belongs to (for provenance fields).
        max_chars: Maximum chunk length in characters.

    Returns:
        Chunks in reading order; empty list for whitespace-only pages.
    """
    spans = _sentence_spans(page.text)
    if not spans:
        return []

    # Group sentence spans into chunk spans without crossing max_chars.
    chunk_spans: list[tuple[int, int]] = []
    group_start: int | None = None
    group_end = 0
    for start, end in spans:
        if end - start > max_chars:
            # Oversized sentence: flush the current group, then hard-split it.
            if group_start is not None:
                chunk_spans.append((group_start, group_end))
                group_start = None
            for window in range(start, end, max_chars):
                chunk_spans.append((window, min(window + max_chars, end)))
            continue
        if group_start is None:
            group_start, group_end = start, end
        elif end - group_start <= max_chars:
            group_end = end
        else:
            chunk_spans.append((group_start, group_end))
            group_start, group_end = start, end
    if group_start is not None:
        chunk_spans.append((group_start, group_end))

    return [
        Chunk(
            chunk_id=mint_chunk_id(doc.doc_id, page.page, index),
            doc_id=doc.doc_id,
            doc_name=doc.name,
            page=page.page,
            chunk_index=index,
            text=page.text[start:end],
            start_char=start,
            end_char=end,
        )
        for index, (start, end) in enumerate(chunk_spans)
    ]


def chunk_document(doc: Document, max_chars: int) -> list[Chunk]:
    """Chunk every page of a document. See :func:`chunk_page`."""
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(chunk_page(page, doc, max_chars))
    return chunks


def discover_files(target: str | Path) -> list[Path]:
    """Find supported files under a path.

    Args:
        target: A single file or a directory to walk recursively.

    Returns:
        Sorted list of supported files.

    Raises:
        FileNotFoundError: If ``target`` does not exist.
        ValueError: If ``target`` is a file of an unsupported type.
    """
    path = Path(target).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type '{path.suffix}': {path}")
        return [path]
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def ingest_path(
    target: str | Path,
    settings: Settings,
    embedding_function: EmbeddingFunction[Documents] | None = None,
) -> IngestResult:
    """Ingest a file or directory into the chunk registry and vector store.

    Args:
        target: File or directory of documents to ingest.
        settings: Loaded AuditRAG settings.
        embedding_function: Optional override, mainly for tests. When ``None``
            the backend configured in ``settings.embedding`` is used.

    Returns:
        Per-file and aggregate ingestion results.
    """
    # Imported here so a chromadb import problem surfaces at ingest time,
    # not when unrelated modules import this package.
    from auditrag.vector_store import VectorStore

    files = discover_files(target)
    result = IngestResult()

    if embedding_function is None:
        embedding_function = build_embedding_function(settings.embedding)

    vector_store = VectorStore(
        path=settings.chroma_path,
        collection=settings.storage.collection,
        embedding_function=embedding_function,
    )

    with ChunkStore(settings.chunk_db_path) as store:
        for file_path in files:
            doc = load_document(file_path)
            existing = store.find_doc_by_path(doc.path)

            if existing is not None and existing[1] == doc.sha256:
                result.files.append(
                    FileIngestResult(
                        path=doc.path, doc_id=doc.doc_id, status="skipped", chunks=0
                    )
                )
                continue

            status = "ingested"
            if existing is not None:
                # Same path, new content: replace the old document entirely.
                old_doc_id = existing[0]
                store.delete_document(old_doc_id)
                vector_store.delete_document(old_doc_id)
                status = "updated"

            chunks = chunk_document(doc, settings.chunking.max_chars)
            store.upsert_document(doc.doc_id, doc.name, doc.path, doc.sha256, len(doc.pages))
            store.upsert_chunks(chunks)
            vector_store.upsert_chunks(chunks)

            result.files.append(
                FileIngestResult(
                    path=doc.path, doc_id=doc.doc_id, status=status, chunks=len(chunks)
                )
            )

    return result
