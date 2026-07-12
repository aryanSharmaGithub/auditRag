"""Tests for the ingestion pipeline: loading, chunking, and indexing.

The vector store is exercised with a deterministic dummy embedding function
so tests run offline without downloading the local embedding model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from auditrag.chunk_store import ChunkStore
from auditrag.config import ChunkingSettings, Settings, StorageSettings
from auditrag.ingest import chunk_page, ingest_path, load_document
from auditrag.models import Document, PageText


class DummyEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic, offline stand-in for a real embedding model."""

    def __init__(self) -> None:  # noqa: D107 (chromadb requires an explicit __init__)
        pass

    def get_config(self) -> dict[str, str]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, str]) -> "DummyEmbeddingFunction":
        return DummyEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            embeddings.append([b / 255.0 for b in digest[:8]])
        return embeddings

    @staticmethod
    def name() -> str:
        return "auditrag-test-dummy"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings pointing all storage at a temporary directory."""
    return Settings(
        chunking=ChunkingSettings(max_chars=200),
        storage=StorageSettings(data_dir=str(tmp_path / "data")),
    )


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    """A directory with one .txt and one .md sample document."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text(
        "AuditRAG produces verifiable answers. Every claim cites a chunk. "
        "Citations resolve to exact page numbers. " * 3,
        encoding="utf-8",
    )
    (docs / "guide.md").write_text(
        "# Guide\n\nHybrid search combines BM25 with vectors.\n\n"
        "Faithfulness checks flag unsupported claims.",
        encoding="utf-8",
    )
    return docs


def test_chunk_ids_are_deterministic_with_exact_offsets() -> None:
    page = PageText(page=3, text="First sentence here. Second sentence follows! Third one?")
    document = Document(
        doc_id="abc123def456", name="x.txt", path="/x.txt", sha256="0" * 64, pages=[page]
    )

    chunks_a = chunk_page(page, document, max_chars=25)
    chunks_b = chunk_page(page, document, max_chars=25)

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert chunks_a[0].chunk_id == "abc123def456:3:0"
    for chunk in chunks_a:
        # Offsets must slice the page text back to the exact chunk text.
        assert page.text[chunk.start_char : chunk.end_char] == chunk.text


def test_load_document_treats_text_as_single_page(docs_dir: Path) -> None:
    doc = load_document(docs_dir / "guide.md")
    assert len(doc.pages) == 1
    assert doc.pages[0].page == 1
    assert "Hybrid search" in doc.pages[0].text
    assert len(doc.doc_id) == 12


def test_ingest_directory_populates_both_stores(settings: Settings, docs_dir: Path) -> None:
    result = ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    assert result.files_ingested == 2
    assert result.total_chunks > 0

    with ChunkStore(settings.chunk_db_path) as store:
        assert store.count_documents() == 2
        assert store.count_chunks() == result.total_chunks

        # Every chunk resolves by its canonical ID with provenance intact.
        first = result.files[0]
        chunk = store.get_chunk(f"{first.doc_id}:1:0")
        assert chunk is not None
        assert chunk.page == 1
        assert chunk.doc_name in ("guide.md", "notes.txt")

    from auditrag.vector_store import VectorStore

    vs = VectorStore(
        path=settings.chroma_path,
        collection=settings.storage.collection,
        embedding_function=DummyEmbeddingFunction(),
    )
    assert vs.count() == result.total_chunks


def test_reingest_is_idempotent(settings: Settings, docs_dir: Path) -> None:
    first = ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    second = ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    assert second.files_skipped == 2
    assert second.total_chunks == 0

    with ChunkStore(settings.chunk_db_path) as store:
        assert store.count_chunks() == first.total_chunks


def test_changed_file_replaces_old_chunks(settings: Settings, docs_dir: Path) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    (docs_dir / "guide.md").write_text("Completely new content.", encoding="utf-8")
    result = ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    updated = [f for f in result.files if f.status == "updated"]
    assert len(updated) == 1

    with ChunkStore(settings.chunk_db_path) as store:
        # Still exactly two documents: the old version was replaced, not kept.
        assert store.count_documents() == 2
        new_chunk = store.get_chunk(f"{updated[0].doc_id}:1:0")
        assert new_chunk is not None
        assert new_chunk.text == "Completely new content."
