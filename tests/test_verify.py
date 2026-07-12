"""Tests for the faithfulness verification pass.

Covers the verdict-line parser (lenient formats, missing/extra lines), the
verification prompt, and the full pipeline including the demo-critical case:
a claim the evidence does not support gets flagged ``unsupported``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from auditrag.answer import generate_answer
from auditrag.api import create_app
from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.ingest import ingest_path
from auditrag.models import Chunk
from auditrag.prompts import build_verification_prompt
from auditrag.retrieval import Retriever
from auditrag.verify import parse_verifier_output, verify_answer

from .conftest import DummyEmbeddingFunction, FakeLLM

# --- parse_verifier_output ----------------------------------------------------


def test_well_formed_verdict_lines_parse() -> None:
    verdicts, warnings = parse_verifier_output(
        "1: supported - the evidence states this directly\n"
        "2: unsupported - the evidence says nothing about this",
        [1, 2],
    )
    assert warnings == []
    assert verdicts[1] == ("supported", "the evidence states this directly")
    assert verdicts[2][0] == "unsupported"


def test_lenient_formats_parse() -> None:
    verdicts, warnings = parse_verifier_output(
        "1) SUPPORTED — matches the source\n2. Partial: only the first half is covered",
        [1, 2],
    )
    assert warnings == []
    assert verdicts[1][0] == "supported"
    assert verdicts[2][0] == "partial"


def test_missing_verdict_produces_warning_not_a_guess() -> None:
    verdicts, warnings = parse_verifier_output("1: supported - fine", [1, 2])
    assert 2 not in verdicts
    assert any("claim 2" in w for w in warnings)


def test_unsubmitted_claim_number_is_ignored_with_warning() -> None:
    verdicts, warnings = parse_verifier_output(
        "1: supported - fine\n9: unsupported - invented", [1]
    )
    assert list(verdicts) == [1]
    assert any("claim 9" in w for w in warnings)


def test_junk_lines_are_ignored() -> None:
    verdicts, warnings = parse_verifier_output(
        "Here are my verdicts:\n\n1: supported - fine\nThank you!", [1]
    )
    assert verdicts[1][0] == "supported"
    assert warnings == []


# --- build_verification_prompt --------------------------------------------------


def _chunk(chunk_id: str, text: str) -> Chunk:
    doc_id, page, index = chunk_id.split(":")
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name="doc.pdf",
        page=int(page),
        chunk_index=int(index),
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_shared_evidence_is_rendered_once() -> None:
    evidence = {"aaa:1:0": _chunk("aaa:1:0", "Retention is seven years.")}
    prompt = build_verification_prompt(
        [(1, "Records are kept seven years.", ["aaa:1:0"]),
         (2, "Retention lasts seven years.", ["aaa:1:0"])],
        evidence,
    )
    assert prompt.count("Retention is seven years.") == 1
    assert '1. "Records are kept seven years." (evidence: E1)' in prompt
    assert '2. "Retention lasts seven years." (evidence: E1)' in prompt


# --- verify_answer / pipeline ---------------------------------------------------


def _make_retriever(settings: Settings, docs_dir: Path) -> Retriever:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    return Retriever(settings, embedding_function=DummyEmbeddingFunction())


def test_unsupported_claim_is_flagged(settings: Settings, docs_dir: Path) -> None:
    """The demo-critical path: a fabricated claim gets an unsupported verdict."""
    retriever = _make_retriever(settings, docs_dir)
    fake = FakeLLM(
        # Generation: one real claim, one fabrication, one uncited sentence.
        "Hybrid search combines BM25 with vectors [1]. "
        "Hybrid search was invented in 1962 [1]. "
        "The sources do not cover pricing.",
        # Verification.
        "1: supported - the evidence states this\n"
        "2: unsupported - the evidence says nothing about 1962",
    )

    answer = generate_answer(
        "hybrid search?", settings, top_k=2, verify=True,
        retriever=retriever, llm_client=fake,
    )

    assert answer.verified
    assert [c.verdict for c in answer.claims] == ["supported", "unsupported", "uncited"]
    assert answer.claims[1].verdict_note == "the evidence says nothing about 1962"
    assert len(fake.calls) == 2
    # The verifier must judge against registry text, not a paraphrase.
    assert "Hybrid search combines BM25 with vectors." in fake.calls[1][1]


def test_uncited_only_answer_skips_the_verifier_call(
    settings: Settings, docs_dir: Path
) -> None:
    retriever = _make_retriever(settings, docs_dir)
    fake = FakeLLM("The provided sources do not contain this information.")

    answer = generate_answer(
        "anything", settings, top_k=2, verify=True,
        retriever=retriever, llm_client=fake,
    )

    assert answer.verified
    assert answer.claims[0].verdict == "uncited"
    assert len(fake.calls) == 1  # generation only; nothing to verify


def test_missing_registry_chunk_degrades_to_uncited(
    settings: Settings, docs_dir: Path
) -> None:
    retriever = _make_retriever(settings, docs_dir)
    answer = generate_answer(
        "anything", settings, top_k=2,
        retriever=retriever, llm_client=FakeLLM("A cited fact [1]."),
    )
    victim = answer.claims[0].chunk_ids[0]
    with ChunkStore(settings.chunk_db_path) as store:
        store._conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (victim,))
        store._conn.commit()

    verifier = FakeLLM("unused")
    verified = verify_answer(answer, settings, llm_client=verifier)

    assert verified.claims[0].verdict == "uncited"
    assert any(victim in w for w in verified.warnings)
    assert verifier.calls == []  # no evidence left to verify against


def test_verifier_without_verdict_leaves_none_and_warns(
    settings: Settings, docs_dir: Path
) -> None:
    retriever = _make_retriever(settings, docs_dir)
    fake = FakeLLM("First fact [1]. Second fact [2].", "1: supported - fine")

    answer = generate_answer(
        "anything", settings, top_k=2, verify=True,
        retriever=retriever, llm_client=fake,
    )

    assert answer.claims[0].verdict == "supported"
    assert answer.claims[1].verdict is None
    assert any("claim 2" in w for w in answer.warnings)


def test_ask_endpoint_with_verify_returns_verdicts(
    settings: Settings, docs_dir: Path
) -> None:
    ingest_path(docs_dir, settings, embedding_function=DummyEmbeddingFunction())
    app = create_app(
        settings,
        embedding_function=DummyEmbeddingFunction(),
        llm_client=FakeLLM("A cited fact [1].", "1: supported - fine"),
    )

    response = TestClient(app).post(
        "/ask", json={"question": "anything", "verify": True}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is True
    assert body["claims"][0]["verdict"] == "supported"
