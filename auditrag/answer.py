"""Cited generation: retrieve → generate → parse and validate citations.

The provenance chain through this module:

1. Retrieval returns chunks whose IDs are canonical (see
   :mod:`auditrag.retrieval`).
2. The prompt presents them as integer labels ``[1]..[k]``; the label→ID map
   exists only for the duration of the request.
3. The model's answer is split into sentence-level claims and each ``[n]``
   marker is mapped back through that table. A label that was never offered
   is recorded on the claim as invalid and surfaced as a warning — invented
   citations are detected, not resolved.

Inline bracket markers (rather than JSON structured output) are a deliberate
choice: they work on any OpenAI-compatible endpoint including small local
models, and they degrade gracefully — a malformed citation costs one
warning, not the whole parse.
"""

from __future__ import annotations

import re

from chromadb.api.types import Documents, EmbeddingFunction

from auditrag.config import Settings
from auditrag.llm import LLMClient
from auditrag.models import Answer, Claim
from auditrag.prompts import ANSWER_SYSTEM, build_answer_prompt
from auditrag.retrieval import Retriever

_MARKER = re.compile(r"\[(\d+)\]")
_LEADING_MARKERS = re.compile(r"^((?:\s*\[\d+\])+)\s*(.*)$", re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_claims(answer_text: str) -> list[str]:
    """Split an answer into sentence-level claim strings.

    Splits on sentence-final punctuation and newlines, then repairs the one
    systematic artifact: models often place markers *after* the period
    ("Foo. [1] Bar."), which strands them at the start of the next segment.
    Leading marker runs are moved back to the claim they cite.
    """
    segments = [s.strip() for s in _SENTENCE_SPLIT.split(answer_text) if s.strip()]
    claims: list[str] = []
    for segment in segments:
        match = _LEADING_MARKERS.match(segment)
        if match and claims:
            claims[-1] += " " + match.group(1).strip()
            rest = match.group(2).strip()
            if rest:
                claims.append(rest)
        else:
            claims.append(segment)
    return claims


def parse_cited_answer(
    answer_text: str, label_map: dict[int, str]
) -> tuple[list[Claim], list[str]]:
    """Parse a model answer into claims with resolved citations.

    Args:
        answer_text: Raw model output containing ``[n]`` markers.
        label_map: The request-scoped mapping from offered labels to
            canonical chunk IDs.

    Returns:
        ``(claims, warnings)``. Each claim carries the chunk IDs its valid
        labels resolved to; labels absent from ``label_map`` are recorded on
        the claim and produce one warning each. Uncited sentences are kept —
        flagging them is the faithfulness verifier's job, not the parser's.
    """
    claims: list[Claim] = []
    warnings: list[str] = []

    for raw in _split_claims(answer_text):
        labels = [int(label) for label in _MARKER.findall(raw)]
        text = _MARKER.sub("", raw)
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)  # no stranded space before punctuation
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text:
            continue

        chunk_ids: list[str] = []
        invalid: list[int] = []
        for label in labels:
            if label in label_map:
                if label_map[label] not in chunk_ids:
                    chunk_ids.append(label_map[label])
            elif label not in invalid:
                invalid.append(label)

        for label in invalid:
            warnings.append(
                f"The model cited source [{label}], which was never provided "
                f"(offered: 1–{len(label_map)}). The citation was flagged, not resolved."
            )
        claims.append(Claim(text=text, chunk_ids=chunk_ids, invalid_labels=invalid))

    return claims, warnings


def generate_answer(
    question: str,
    settings: Settings,
    top_k: int = 6,
    retriever: Retriever | None = None,
    llm_client: LLMClient | None = None,
    embedding_function: EmbeddingFunction[Documents] | None = None,
) -> Answer:
    """Answer a question with sentence-level citations.

    Args:
        question: Natural-language question.
        settings: Loaded AuditRAG settings.
        top_k: Number of chunks to offer the model as context.
        retriever: Optional pre-built retriever (reused across API requests).
        llm_client: Optional pre-built LLM client, or a test double.
        embedding_function: Optional override used only when ``retriever``
            is not supplied.

    Returns:
        The cited answer, including the offered chunks (label ``[i+1]`` is
        ``chunks[i]``) and any integrity warnings.

    Raises:
        EmptyIndexError: If no documents have been ingested yet.
        RetrievalError: If retrieval fails (see :mod:`auditrag.retrieval`).
        LLMError: If the chat endpoint fails.
    """
    if retriever is None:
        retriever = Retriever(settings, embedding_function)
    if llm_client is None:
        llm_client = LLMClient(settings.llm)

    retrieval = retriever.search(question, top_k=top_k)
    label_map = {i + 1: hit.chunk.chunk_id for i, hit in enumerate(retrieval.chunks)}

    answer_text = llm_client.complete(
        system=ANSWER_SYSTEM,
        user=build_answer_prompt(question, retrieval.chunks),
    )
    claims, parse_warnings = parse_cited_answer(answer_text, label_map)

    return Answer(
        question=question,
        answer_text=answer_text,
        claims=claims,
        chunks=retrieval.chunks,
        model=llm_client.model,
        warnings=retrieval.warnings + parse_warnings,
    )
