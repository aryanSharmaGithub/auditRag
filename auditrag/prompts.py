"""Prompts for cited generation.

Kept as plain, readable strings on purpose: for a tool whose pitch is
verifiability, the prompts are part of the audit surface.

The model only ever sees small integer labels ``[1]..[k]`` — never real chunk
IDs. The label→ID mapping lives in request scope on the Python side, so an
invented citation cannot resolve to anything.
"""

from __future__ import annotations

from auditrag.models import RetrievedChunk

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
