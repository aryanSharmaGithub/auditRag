"""Timestamped PDF evidence reports.

A report is a self-contained audit trail for one or more Q&A exchanges:
the question, the cited answer, every claim with its faithfulness verdict,
and — crucially — the *verbatim* text of every cited chunk with its document,
page, and chunk ID.

Evidence text is always fetched from the SQLite chunk registry at render
time, by chunk ID. It is never taken from the answer payload, so a report
reproduces exactly what a citation resolves to; a cited chunk that no longer
exists in the registry is reported as missing rather than reconstructed.

fpdf2 was chosen over HTML-to-PDF tools because it is pure Python — no
system libraries, keeping the zero-setup promise. Its core fonts are
Latin-1, so text is sanitized with replacement characters where needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fpdf import FPDF

from auditrag import __version__
from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.models import Answer

_VERDICT_COLORS: dict[str | None, tuple[int, int, int]] = {
    "supported": (22, 122, 61),
    "partial": (185, 120, 20),
    "unsupported": (180, 32, 32),
    "uncited": (110, 110, 110),
    None: (110, 110, 110),
}
_TEXT = (30, 30, 30)
_MUTED = (110, 110, 110)


def _latin1(text: str) -> str:
    """Sanitize text for fpdf2's Latin-1 core fonts (lossy but safe)."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _ReportPDF(FPDF):
    """FPDF subclass adding the standard report footer."""

    def footer(self) -> None:  # noqa: D102 (fpdf2 hook)
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*_MUTED)
        self.cell(0, 8, f"AuditRAG evidence report - page {self.page_no()}/{{nb}}", align="C")


def build_evidence_report(answers: list[Answer], settings: Settings) -> bytes:
    """Render a timestamped PDF evidence report for a Q&A session.

    Args:
        answers: One or more answers, in session order. Verification is not
            required, but verdicts are printed when present.
        settings: Loaded AuditRAG settings (locates the chunk registry).

    Returns:
        The PDF as bytes.
    """
    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # --- report header ------------------------------------------------------
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    models = ", ".join(sorted({a.model for a in answers}))

    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(*_TEXT)
    pdf.cell(0, 9, "AuditRAG Evidence Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(*_MUTED)
    pdf.cell(0, 5, _latin1(f"Generated: {generated}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, _latin1(f"AuditRAG {__version__} - model(s): {models}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0, 5,
        "Evidence text is reproduced verbatim from the chunk registry at export time.",
        new_x="LMARGIN", new_y="NEXT",
    )

    with ChunkStore(settings.chunk_db_path) as store:
        for number, answer in enumerate(answers, start=1):
            _render_answer(pdf, store, number, answer)

    return bytes(pdf.output())


def _render_answer(pdf: _ReportPDF, store: ChunkStore, number: int, answer: Answer) -> None:
    """Render one Q&A exchange: question, answer, claims, evidence, warnings."""
    label_by_id = {hit.chunk.chunk_id: i + 1 for i, hit in enumerate(answer.chunks)}

    pdf.ln(6)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*_TEXT)
    pdf.multi_cell(0, 6, _latin1(f"Q{number}. {answer.question}"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin1(answer.answer_text), new_x="LMARGIN", new_y="NEXT")

    # --- claims with verdicts -------------------------------------------------
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 10)
    title = "Claims" + ("" if answer.verified else " (not verified)")
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")

    for claim in answer.claims:
        verdict = claim.verdict if answer.verified else None
        verdict_text = (claim.verdict or "no verdict") if answer.verified else "unverified"
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*_VERDICT_COLORS[verdict])
        pdf.cell(26, 5, _latin1(verdict_text.upper()))
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(*_TEXT)
        cites = "".join(f"[{label_by_id[cid]}]" for cid in claim.chunk_ids if cid in label_by_id)
        pdf.multi_cell(0, 5, _latin1(f"{claim.text} {cites}".rstrip()),
                       new_x="LMARGIN", new_y="NEXT")
        if answer.verified and claim.verdict_note:
            pdf.set_x(pdf.l_margin + 26)
            pdf.set_font("helvetica", "I", 8)
            pdf.set_text_color(*_MUTED)
            pdf.multi_cell(0, 4, _latin1(claim.verdict_note), new_x="LMARGIN", new_y="NEXT")

    # --- cited evidence, verbatim from the registry ----------------------------
    cited_ids: list[str] = []
    for claim in answer.claims:
        for chunk_id in claim.chunk_ids:
            if chunk_id not in cited_ids:
                cited_ids.append(chunk_id)

    if cited_ids:
        pdf.ln(2)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*_TEXT)
        pdf.cell(0, 6, "Cited evidence", new_x="LMARGIN", new_y="NEXT")

    for chunk_id in cited_ids:
        chunk = store.get_chunk(chunk_id)
        label = f"[{label_by_id[chunk_id]}]" if chunk_id in label_by_id else "[-]"
        pdf.set_font("helvetica", "B", 9)
        if chunk is None:
            pdf.set_text_color(*_VERDICT_COLORS["unsupported"])
            pdf.multi_cell(
                0, 5,
                _latin1(f"{label} {chunk_id} - MISSING from the chunk registry; "
                        "this evidence cannot be reproduced."),
                new_x="LMARGIN", new_y="NEXT",
            )
            continue
        pdf.set_text_color(*_TEXT)
        pdf.multi_cell(
            0, 5,
            _latin1(f"{label} {chunk.doc_name}, p.{chunk.page}  ({chunk.chunk_id})"),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.set_x(pdf.l_margin + 4)
        pdf.multi_cell(0, 4.5, _latin1(chunk.text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    # --- warnings ---------------------------------------------------------------
    if answer.warnings:
        pdf.ln(1)
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*_VERDICT_COLORS["partial"])
        pdf.cell(0, 5, "Warnings", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 8)
        for warning in answer.warnings:
            pdf.multi_cell(0, 4, _latin1(f"- {warning}"), new_x="LMARGIN", new_y="NEXT")
