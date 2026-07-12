"""Prompts for cited generation.

Kept as plain, readable strings on purpose: for a tool whose pitch is
verifiability, the prompts are part of the audit surface.

The model only ever sees small integer labels ``[1]..[k]`` — never real chunk
IDs. The label→ID mapping lives in request scope on the Python side, so an
invented citation cannot resolve to anything.
"""

from __future__ import annotations

from auditrag.models import Chunk, RetrievedChunk

ANSWER_SYSTEM = """\
You answer questions using ONLY the numbered source excerpts provided. Rules:

1. Every sentence in your answer must end with the label(s) of the source(s) \
that support it, like [1] or [1][3]. Cite only labels that appear in the \
sources; never invent labels.
2. Do not use any knowledge that is not in the sources. If the sources do not \
contain the answer, reply with a single uncited sentence: "The provided \
sources do not contain this information."
3. Be concise and factual. No preamble, no summary of the sources, no \
markdown headings.
"""


VERIFIER_SYSTEM = """\
You are a strict fact-checker. You receive numbered evidence excerpts and a \
list of claims; each claim names the evidence it cites. Judge every claim \
against ONLY its cited evidence — not the other excerpts, not your own \
knowledge. Verdicts:

- supported: the evidence fully entails the claim.
- partial: the evidence supports part of the claim, but part of it goes \
beyond or qualifies what the evidence says.
- unsupported: the evidence does not support the claim, or contradicts it.

Output exactly one line per claim, in claim order, and nothing else:
<claim number>: <verdict> - <one short sentence of justification>
"""


def build_verification_prompt(
    claims: list[tuple[int, str, list[str]]],
    evidence: dict[str, Chunk],
) -> str:
    """Render the verifier's user message.

    Args:
        claims: ``(claim_number, claim_text, cited_chunk_ids)`` triples;
            claim numbers are 1-based and must match the output lines.
        evidence: Canonical chunks by ID (fetched fresh from the registry);
            each is rendered once as ``[E<i>]`` no matter how many claims
            cite it.

    Returns:
        The formatted verification prompt.
    """
    evidence_labels = {chunk_id: f"E{i}" for i, chunk_id in enumerate(evidence, start=1)}

    evidence_blocks: list[str] = []
    for chunk_id, label in evidence_labels.items():
        chunk = evidence[chunk_id]
        evidence_blocks.append(
            f"[{label}] ({chunk.doc_name}, p.{chunk.page})\n{chunk.text}"
        )

    claim_lines: list[str] = []
    for number, text, chunk_ids in claims:
        cited = ", ".join(evidence_labels[cid] for cid in chunk_ids)
        claim_lines.append(f'{number}. "{text}" (evidence: {cited})')

    return (
        "Evidence:\n\n"
        + "\n\n".join(evidence_blocks)
        + "\n\nClaims:\n\n"
        + "\n".join(claim_lines)
    )


def build_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Render the user message: numbered source blocks plus the question.

    Args:
        question: The user's question.
        chunks: Retrieved context; ``chunks[i]`` is presented as label
            ``[i+1]``, matching the positional contract on
            :class:`~auditrag.models.Answer`.

    Returns:
        The formatted user prompt.
    """
    blocks: list[str] = []
    for i, hit in enumerate(chunks, start=1):
        chunk = hit.chunk
        blocks.append(f"[{i}] ({chunk.doc_name}, p.{chunk.page})\n{chunk.text}")
    sources = "\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"
