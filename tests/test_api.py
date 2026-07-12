"""Tests for the HTTP API, including error-status mapping."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from auditrag.api import create_app
from auditrag.config import Settings
from auditrag.ingest import ingest_path

from .conftest import DummyEmbeddingFunction


def _client(settings: Settings) -> TestClient:
    app = create_app(settings, embedding_function=DummyEmbeddingFunction())
    return TestClient(app)


def test_index_serves_web_ui(settings: Settings) -> None:
    response = _client(settings).get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "AuditRAG" in response.text
    # The UI must talk to the endpoints it depends on.
    assert "/ask" in response.text and "/report" in response.text


def test_health_reports_index_stats(settings: Settings, docs_dir: Path) -> None:
    result = ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    response = _client(settings).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] == 2
    assert body["chunks"] == result.total_chunks


def test_health_ok_before_first_ingest(settings: Settings) -> None:
    response = _client(settings).get("/health")
    assert response.status_code == 200
    assert response.json()["chunks"] == 0


def test_query_returns_chunks_with_provenance(settings: Settings, docs_dir: Path) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    response = _client(settings).post(
        "/query", json={"question": "hybrid search", "top_k": 2}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == 2
    first = body["chunks"][0]["chunk"]
    assert {"chunk_id", "doc_name", "page", "text"} <= first.keys()


def test_query_on_empty_index_is_409_with_guidance(settings: Settings) -> None:
    response = _client(settings).post("/query", json={"question": "anything"})
    assert response.status_code == 409
    assert "auditrag ingest" in response.json()["detail"]


def test_query_validation_is_422(settings: Settings) -> None:
    client = _client(settings)
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/query", json={}).status_code == 422
