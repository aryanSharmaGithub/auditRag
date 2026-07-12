"""Shared test fixtures.

All tests run fully offline: the vector store is exercised with a
deterministic dummy embedding function instead of a real model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from auditrag.config import ChunkingSettings, Settings, StorageSettings


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
