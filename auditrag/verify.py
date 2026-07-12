"""Faithfulness verification: judge each claim against its cited evidence.

An independent LLM pass re-checks every cited claim of an answer against the
verbatim text of the chunks it cites. Design decisions that matter:

* **Evidence comes from the SQLite registry, fetched fresh by chunk ID** —
  never from the generation prompt. The verifier judges against exactly what
  a citation resolves to, which is exactly what an evidence report prints.
* **One batched verifier call** for all cited claims, not one call per claim.
  A five-sentence answer costs two LLM round-trips total, not six.
* **Uncited claims are not sent to the verifier.** There is no evidence to
  judge them against; they get the ``uncited`` verdict directly.
* The verifier replies in a plain line format (``<n>: <verdict> - <reason>``)
  parsed leniently; a malformed or missing line leaves that claim's verdict
  ``None`` and adds a warning — a broken verifier degrades loudly, never
  into silent green badges.
"""

from __future__ import annotations

import re

from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.llm import LLMClient
from auditrag.models import Answer, Chunk, Verdict
from auditrag.prompts import VERIFIER_SYSTEM, build_verification_prompt

_VERDICT_LINE = re.compile(
    r"^\s*(\d+)\s*[:.)\-]\s*(supported|partial|unsupported)\b[\s:\-—.]*(.*)$",
    re.IGNORECASE,
)


def parse_verifier_output(
    output: str, claim_numbers: list[int]
) -> tuple[dict[int, tuple[Verdict, str]], list[str]]:
    """Parse the verifier's line-per-claim reply.

    Args:
        output: Raw verifier output.
        claim_numbers: The claim numbers that were submitted; lines with
            other numbers are ignored with a warning.

    Returns:
        ``(verdicts, warnings)`` where ``verdicts`` maps claim number to
        ``(verdict, note)``. A claim number missing from the output produces
        a warning and no entry — the caller must not invent a verdict for it.
    """
    verdicts: dict[int, tuple[Verdict, str]] = {}
    warnings: list[str] = []
    expected = set(claim_numbers)

    for line in output.splitlines():
        match = _VERDICT_LINE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if number not in expected:
            warnings.append(
                f"The verifier returned a verdict for claim {number}, which was "
                "not submitted; ignored."
            )
            continue
        if number in verdicts:
            continue  # first verdict wins; models occasionally repeat lines
        verdict = match.group(2).lower()
        note = match.group(3).strip()
        verdicts[number] = (verdict, note)  # type: ignore[assignment]

    for number in claim_numbers:
        if number not in verdicts:
            warnings.append(
                f"The verifier did not return a verdict for claim {number}; "
                "its verdict is left unset rather than assumed."
            )
    return verdicts, warnings


def verify_answer(
    answer: Answer,
    settings: Settings,
    llm_client: LLMClient | None = None,
) -> Answer:
    """Run the faithfulness pass and return the answer with verdicts attached.

    Args:
        answer: A generated answer (see :func:`auditrag.answer.generate_answer`).
        settings: Loaded AuditRAG settings.
        llm_client: Optional pre-built client or test double; defaults to the
            configured ``llm`` section (same model as generation).

    Returns:
        A copy of ``answer`` with ``verified=True``, per-claim verdicts and
        notes filled in, and any verification warnings appended.

    Raises:
        LLMError: If the verifier request fails outright.
    """
    result = answer.model_copy(deep=True)
    result.verified = True

    # Fetch cited evidence verbatim from the canonical registry.
    evidence: dict[str, Chunk] = {}
    missing: set[str] = set()
    with ChunkStore(settings.chunk_db_path) as store:
        for claim in result.claims:
            for chunk_id in claim.chunk_ids:
                if chunk_id in evidence or chunk_id in missing:
                    continue
                chunk = store.get_chunk(chunk_id)
                if chunk is None:
                    missing.add(chunk_id)
                    result.warnings.append(
                        f"Cited chunk '{chunk_id}' is missing from the chunk "
                        "registry; claims citing it are treated as uncited. "
                        "Re-run 'auditrag ingest' to rebuild the stores."
                    )
                else:
                    evidence[chunk_id] = chunk

    # Split claims into verifiable (has resolvable evidence) and uncited.
    submissions: list[tuple[int, str, list[str]]] = []
    for i, claim in enumerate(result.claims):
        cited = [cid for cid in claim.chunk_ids if cid in evidence]
        if cited:
            submissions.append((i + 1, claim.text, cited))
        else:
            claim.verdict = "uncited"

    if not submissions:
        return result

    if llm_client is None:
        llm_client = LLMClient(settings.llm)

    output = llm_client.complete(
        system=VERIFIER_SYSTEM,
        user=build_verification_prompt(submissions, evidence),
    )
    verdicts, parse_warnings = parse_verifier_output(
        output, [number for number, _, _ in submissions]
    )
    result.warnings.extend(parse_warnings)

    for number, _, _ in submissions:
        if number in verdicts:
            verdict, note = verdicts[number]
            claim = result.claims[number - 1]
            claim.verdict = verdict
            claim.verdict_note = note

    return result
