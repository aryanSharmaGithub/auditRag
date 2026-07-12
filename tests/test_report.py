"""Tests for PDF evidence reports.

Assertions run against text extracted from the rendered PDF (via pypdf),
so they verify what a reader of the report actually sees.
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfReader

from auditrag.answer import generate_answer
from auditrag.api import create_app
from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.ingest import ingest_path
from auditrag.models import Answer
from auditrag.report import build_evidence_report
from auditrag.retrieval import Retriever

from .conftest import DummyEmbeddingFunction, FakeLLM


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() for page in reader.pages)


def _verified_answer(settings: Settings, docs_dir: Path) -> Answer:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    retriever = Retriever(settings, embedding_function=DummyEmbeddingFunction())
    fake = FakeLLM(
        "Hybrid search combines BM25 with vectors [1]. "
        "Hybrid search was invented in 1962 [1].",
        "1: supported - stated in the evidence\n"
        "2: unsupported - the evidence says nothing about 1962",
    )
    return generate_answer(
        "How does hybrid search work?", settings, top_k=2, verify=True,
        retriever=retriever, llm_client=fake,
    )


def test_report_contains_question_verdicts_and_verbatim_evidence(
    settings: Settings, docs_dir: Path
) -> None:
    answer = _verified_answer(settings, docs_dir)

    pdf_bytes = build_evidence_report([answer], settings)
    assert pdf_bytes.startswith(b"%PDF")
    text = _pdf_text(pdf_bytes)

    assert "AuditRAG Evidence Report" in text
    assert "How does hybrid search work?" in text
    assert "SUPPORTED" in text and "UNSUPPORTED" in text
    assert "the evidence says nothing about 1962" in text
    # Verbatim registry text and full provenance for the cited chunk.
    cited_id = answer.claims[0].chunk_ids[0]
    with ChunkStore(settings.chunk_db_path) as store:
        chunk = store.get_chunk(cited_id)
    assert chunk is not None
    assert chunk.doc_name in text
    assert cited_id in text
    assert "Generated:" in text and "UTC" in text


def test_missing_registry_chunk_is_reported_as_missing(
    settings: Settings, docs_dir: Path
) -> None:
    answer = _verified_answer(settings, docs_dir)
    victim = answer.claims[0].chunk_ids[0]
    with ChunkStore(settings.chunk_db_path) as store:
        store._conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (victim,))
        store._conn.commit()

    text = _pdf_text(build_evidence_report([answer], settings))

    assert victim in text
    assert "MISSING" in text


def test_multi_answer_session_renders_in_order(
    settings: Settings, docs_dir: Path
) -> None:
    answer = _verified_answer(settings, docs_dir)
    second = answer.model_copy(deep=True)
    second.question = "A second, different question?"

    text = _pdf_text(build_evidence_report([answer, second], settings))

    assert "Q1." in text and "Q2." in text
    assert text.index("How does hybrid search work?") < text.index(
        "A second, different question?"
    )


def test_report_endpoint_round_trip(settings: Settings, docs_dir: Path) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    app = create_app(
        settings,
        embedding_function=DummyEmbeddingFunction(),
        llm_client=FakeLLM("A cited fact [1].", "1: supported - fine"),
    )
    client = TestClient(app)

    ask = client.post("/ask", json={"question": "anything", "verify": True})
    response = client.post("/report", json={"answers": [ask.json()]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "auditrag-evidence-" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")
    assert "A cited fact." in _pdf_text(response.content)


def test_report_endpoint_rejects_empty_session(settings: Settings) -> None:
    app = create_app(settings, embedding_function=DummyEmbeddingFunction())
    response = TestClient(app).post("/report", json={"answers": []})
    assert response.status_code == 422
