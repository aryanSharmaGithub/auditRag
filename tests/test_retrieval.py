"""Tests for citation-tracked retrieval, including failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.ingest import ingest_path
from auditrag.lexical import BM25Index
from auditrag.retrieval import EmptyIndexError, RetrievalError, Retriever, _rrf_fuse

from .conftest import DummyEmbeddingFunction


def _ingested(settings: Settings, docs_dir: Path) -> Retriever:
    """Ingest the sample docs and return a retriever over them."""
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    return Retriever(settings, embedding_function=DummyEmbeddingFunction())


def test_search_returns_ranked_chunks_with_provenance(
    settings: Settings, docs_dir: Path
) -> None:
    retriever = _ingested(settings, docs_dir)

    result = retriever.search("faithfulness checks", top_k=3)

    assert result.query == "faithfulness checks"
    assert 0 < len(result.chunks) <= 3
    assert result.warnings == []
    for i, hit in enumerate(result.chunks):
        assert hit.rank == i
        assert hit.chunk.page == 1
        assert hit.chunk.doc_name in ("guide.md", "notes.txt")
        # ID decodes back to the provenance fields it was minted from.
        doc_id, page, index = hit.chunk.chunk_id.split(":")
        assert (doc_id, int(page), int(index)) == (
            hit.chunk.doc_id,
            hit.chunk.page,
            hit.chunk.chunk_index,
        )


def test_top_k_larger_than_index_is_clamped(settings: Settings, docs_dir: Path) -> None:
    retriever = _ingested(settings, docs_dir)
    result = retriever.search("anything", top_k=50)
    with ChunkStore(settings.chunk_db_path) as store:
        assert len(result.chunks) == store.count_chunks()


def test_empty_index_raises_actionable_error(settings: Settings) -> None:
    retriever = Retriever(settings, embedding_function=DummyEmbeddingFunction())
    with pytest.raises(EmptyIndexError, match="auditrag ingest"):
        retriever.search("anything")


def test_registry_desync_warns_and_skips(settings: Settings, docs_dir: Path) -> None:
    retriever = _ingested(settings, docs_dir)

    # Simulate a desync: a chunk exists in the vector index but its registry
    # record is gone (interrupted ingest, deleted chunks.db, ...).
    full = retriever.search("anything", top_k=50)
    victim = full.chunks[0].chunk.chunk_id
    with ChunkStore(settings.chunk_db_path) as store:
        store._conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (victim,))
        store._conn.commit()

    result = retriever.search("anything", top_k=50)

    assert len(result.warnings) == 1
    assert victim in result.warnings[0]
    assert victim not in [hit.chunk.chunk_id for hit in result.chunks]


def test_rrf_ranks_doc_found_by_both_rankers_first() -> None:
    fused = _rrf_fuse([["a", "b", "c"], ["c", "d"]])
    ids = [chunk_id for chunk_id, _ in fused]
    # "c" appears in both rankings (ranks 2 and 0); every other doc appears
    # in only one, so "c" must fuse to the top.
    assert ids[0] == "c"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_preserves_order_within_a_single_ranking() -> None:
    fused = _rrf_fuse([["a", "b", "c"]])
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]


def test_bm25_ranks_exact_term_match_first() -> None:
    index = BM25Index(
        [
            ("doc:1:0", "The retention period for customer records is seven years."),
            ("doc:2:0", "Vendors sign a data processing agreement before onboarding."),
            ("doc:3:0", "Incidents are reported within seventy-two hours."),
        ]
    )
    assert index.query("data processing agreement", n_results=3)[0] == "doc:2:0"
    # No shared terms at all: BM25 noise must not leak into fusion.
    assert index.query("zebra xylophone", n_results=3) == []


def test_hybrid_search_surfaces_exact_keyword_match(
    settings: Settings, docs_dir: Path
) -> None:
    retriever = _ingested(settings, docs_dir)

    result = retriever.search("BM25 vectors", top_k=3)

    texts = [hit.chunk.text for hit in result.chunks]
    # The dummy embeddings are hash noise, so this chunk surfacing proves the
    # lexical ranking is contributing to fusion.
    assert any("BM25 with vectors" in text for text in texts)


def test_embedding_failure_raises_retrieval_error(
    settings: Settings, docs_dir: Path
) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    class ExplodingEmbeddingFunction(DummyEmbeddingFunction):
        """Same identity as the ingest-time function, but fails on queries.

        The name must match what ChromaDB persisted at ingest time, otherwise
        the collection refuses to open at all (covered separately below).
        """

        def __call__(self, input):  # type: ignore[no-untyped-def]
            raise ConnectionError("connection refused")

    retriever = Retriever(settings, embedding_function=ExplodingEmbeddingFunction())
    with pytest.raises(RetrievalError, match="auditrag.yaml"):
        retriever.search("anything")


def test_changed_embedding_function_raises_retrieval_error(
    settings: Settings, docs_dir: Path
) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    class RenamedEmbeddingFunction(DummyEmbeddingFunction):
        """Simulates switching embedding config after ingest."""

        @staticmethod
        def name() -> str:
            return "auditrag-test-other"

    with pytest.raises(RetrievalError, match="re-ingest"):
        Retriever(settings, embedding_function=RenamedEmbeddingFunction())
