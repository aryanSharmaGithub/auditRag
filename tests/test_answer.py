"""Tests for cited generation: citation parsing, validation, and the pipeline.

The parser tests use hand-crafted model outputs, including malformed ones —
this is the trust boundary of the whole system, so it gets the heaviest
coverage. Pipeline and API tests use a canned fake LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auditrag.answer import generate_answer, parse_cited_answer
from auditrag.api import create_app
from auditrag.config import Settings
from auditrag.ingest import ingest_path
from auditrag.llm import LLMError
from auditrag.retrieval import Retriever

from .conftest import DummyEmbeddingFunction, FakeLLM

LABELS = {1: "aaa:1:0", 2: "bbb:2:1", 3: "ccc:3:0"}


# --- parse_cited_answer -----------------------------------------------------


def test_well_formed_citations_resolve() -> None:
    claims, warnings = parse_cited_answer(
        "Retention is seven years [1]. Backups are encrypted [2][3].", LABELS
    )
    assert warnings == []
    assert [c.text for c in claims] == [
        "Retention is seven years.",
        "Backups are encrypted.",
    ]
    assert claims[0].chunk_ids == ["aaa:1:0"]
    assert claims[1].chunk_ids == ["bbb:2:1", "ccc:3:0"]


def test_markers_after_the_period_attach_to_preceding_sentence() -> None:
    claims, warnings = parse_cited_answer(
        "Retention is seven years. [1] Backups are encrypted. [2]", LABELS
    )
    assert warnings == []
    assert claims[0].chunk_ids == ["aaa:1:0"]
    assert claims[1].chunk_ids == ["bbb:2:1"]


def test_invented_label_is_flagged_not_resolved() -> None:
    claims, warnings = parse_cited_answer("The sky is green [7].", LABELS)
    assert claims[0].chunk_ids == []
    assert claims[0].invalid_labels == [7]
    assert len(warnings) == 1
    assert "[7]" in warnings[0] and "never provided" in warnings[0]


def test_uncited_sentence_is_kept_without_citations() -> None:
    claims, warnings = parse_cited_answer(
        "The provided sources do not contain this information.", LABELS
    )
    assert warnings == []
    assert len(claims) == 1
    assert claims[0].chunk_ids == [] and claims[0].invalid_labels == []


def test_duplicate_labels_deduplicate() -> None:
    claims, _ = parse_cited_answer("Same source twice [1][1].", LABELS)
    assert claims[0].chunk_ids == ["aaa:1:0"]


def test_newline_separated_bullets_are_separate_claims() -> None:
    claims, _ = parse_cited_answer(
        "- Records kept seven years [1]\n- Incidents reported in 72 hours [2]", LABELS
    )
    assert len(claims) == 2
    assert claims[0].chunk_ids == ["aaa:1:0"]
    assert claims[1].chunk_ids == ["bbb:2:1"]


# --- generate_answer pipeline ------------------------------------------------


def _make_retriever(settings: Settings, docs_dir: Path) -> Retriever:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    return Retriever(settings, embedding_function=DummyEmbeddingFunction())


def test_generate_answer_resolves_labels_to_real_chunk_ids(
    settings: Settings, docs_dir: Path
) -> None:
    retriever = _make_retriever(settings, docs_dir)

    answer = generate_answer(
        "What does hybrid search combine?",
        settings,
        top_k=2,
        retriever=retriever,
        llm_client=FakeLLM("Hybrid search combines BM25 with vectors [1]."),
    )

    assert answer.model == "fake-model"
    assert len(answer.claims) == 1
    # Label [1] must resolve to exactly the first offered chunk.
    assert answer.claims[0].chunk_ids == [answer.chunks[0].chunk.chunk_id]
    assert answer.warnings == []


def test_generate_answer_flags_out_of_range_citation(
    settings: Settings, docs_dir: Path
) -> None:
    retriever = _make_retriever(settings, docs_dir)

    answer = generate_answer(
        "anything",
        settings,
        top_k=2,
        retriever=retriever,
        llm_client=FakeLLM("Fabricated fact [9]."),
    )

    assert answer.claims[0].invalid_labels == [9]
    assert any("never provided" in w for w in answer.warnings)


# --- /ask endpoint ------------------------------------------------------------


def test_ask_endpoint_returns_cited_answer(settings: Settings, docs_dir: Path) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    app = create_app(
        settings,
        embedding_function=DummyEmbeddingFunction(),
        llm_client=FakeLLM("Faithfulness checks flag unsupported claims [1]."),
    )

    response = TestClient(app).post("/ask", json={"question": "what gets flagged?"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-model"
    assert body["claims"][0]["chunk_ids"] == [body["chunks"][0]["chunk"]["chunk_id"]]


def test_ask_on_empty_index_is_409(settings: Settings) -> None:
    app = create_app(
        settings,
        embedding_function=DummyEmbeddingFunction(),
        llm_client=FakeLLM("irrelevant"),
    )
    response = TestClient(app).post("/ask", json={"question": "anything"})
    assert response.status_code == 409


def test_ask_llm_failure_is_502(settings: Settings, docs_dir: Path) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())

    class BrokenLLM(FakeLLM):
        def complete(self, system: str, user: str) -> str:
            raise LLMError("endpoint unreachable")

    app = create_app(
        settings,
        embedding_function=DummyEmbeddingFunction(),
        llm_client=BrokenLLM(""),
    )
    response = TestClient(app).post("/ask", json={"question": "anything"})
    assert response.status_code == 502
    assert "unreachable" in response.json()["detail"]
