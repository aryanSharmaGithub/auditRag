"""Core data models shared across the AuditRAG pipeline.

Every stage of the pipeline (ingestion, retrieval, generation, verification)
exchanges these Pydantic models rather than framework-specific objects, so
chunk provenance can never be silently dropped between stages.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


def mint_chunk_id(doc_id: str, page: int, chunk_index: int) -> str:
    """Mint the canonical, human-decodable ID for a chunk.

    The format is ``{doc_id}:{page}:{chunk_index}``, e.g. ``a3f2b1c4d5e6:14:2``.
    IDs are deterministic: re-ingesting an unchanged file yields identical IDs,
    and anyone reading an evidence report can locate the source by eye.

    Args:
        doc_id: Content-hash identifier of the source document.
        page: 1-based page number the chunk was extracted from.
        chunk_index: 0-based index of the chunk within that page.

    Returns:
        The chunk ID string.
    """
    return f"{doc_id}:{page}:{chunk_index}"


class PageText(BaseModel):
    """Extracted text of a single page of a source document."""

    page: int = Field(description="1-based page number.")
    text: str = Field(description="Raw extracted text of the page.")


class Document(BaseModel):
    """A loaded source document, split into pages.

    Plain-text formats (``.txt``, ``.md``) are treated as single-page
    documents so that every chunk, regardless of format, carries exactly one
    unambiguous page number.
    """

    doc_id: str = Field(description="First 12 hex chars of the SHA-256 of the file bytes.")
    name: str = Field(description="File name, e.g. 'report.pdf'.")
    path: str = Field(description="Absolute path the document was loaded from.")
    sha256: str = Field(description="Full SHA-256 of the file bytes.")
    pages: list[PageText] = Field(description="Extracted pages in order.")


class Chunk(BaseModel):
    """A retrievable unit of text with full provenance.

    Chunks never cross page boundaries, so ``page`` is always a single exact
    number. ``start_char``/``end_char`` are offsets into the page text, i.e.
    ``page_text[start_char:end_char] == text``.
    """

    chunk_id: str = Field(description="Canonical ID, format '{doc_id}:{page}:{chunk_index}'.")
    doc_id: str = Field(description="Identifier of the source document.")
    doc_name: str = Field(description="File name of the source document.")
    page: int = Field(description="1-based page number the chunk came from.")
    chunk_index: int = Field(description="0-based index of the chunk within its page.")
    text: str = Field(description="Verbatim chunk text.")
    start_char: int = Field(description="Start offset of the chunk within the page text.")
    end_char: int = Field(description="End offset (exclusive) within the page text.")


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its relevance score.

    The embedded :class:`Chunk` is always hydrated from the SQLite chunk
    registry — never from vector-store payloads — so downstream stages
    (generation, verification, evidence reports) see canonical content.
    """

    chunk: Chunk = Field(description="The chunk, hydrated from the canonical registry.")
    score: float = Field(
        description="Similarity score (1 - cosine distance); higher is more relevant."
    )
    rank: int = Field(description="0-based position in the ranked result list.")


class RetrievalResult(BaseModel):
    """Ranked chunks for a query, plus any integrity warnings.

    ``warnings`` is non-empty when the vector index and the chunk registry
    disagree (e.g. an embedding whose chunk is missing from SQLite). Such
    hits are skipped rather than served with unverifiable provenance.
    """

    query: str = Field(description="The query text as searched.")
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    """One sentence of a generated answer, with its resolved citations.

    ``chunk_ids`` contains only citations that resolved through the
    request-scoped label map. Labels the model invented (not offered in the
    prompt) land in ``invalid_labels`` — they are flagged, never resolved.
    """

    text: str = Field(description="The sentence with citation markers stripped.")
    chunk_ids: list[str] = Field(
        default_factory=list, description="Canonical chunk IDs this sentence cites."
    )
    invalid_labels: list[int] = Field(
        default_factory=list,
        description="Citation labels the model emitted that were never offered.",
    )


class Answer(BaseModel):
    """A cited answer to a single question.

    ``chunks[i]`` is the source that was offered to the model as label
    ``[i+1]``; that positional contract is what lets clients render the raw
    ``answer_text`` markers as links.
    """

    question: str = Field(description="The question as asked.")
    answer_text: str = Field(description="Raw model output, citation markers included.")
    claims: list[Claim] = Field(description="Parsed sentences with resolved citations.")
    chunks: list[RetrievedChunk] = Field(
        description="Context offered to the model; index i corresponds to label [i+1]."
    )
    model: str = Field(description="Chat model that generated the answer.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Integrity warnings: invented citations, store desync, ...",
    )


class FileIngestResult(BaseModel):
    """Outcome of ingesting a single file."""

    path: str = Field(description="Absolute path of the file.")
    doc_id: str = Field(description="Document ID assigned to the file.")
    status: str = Field(description="One of 'ingested', 'updated', 'skipped'.")
    chunks: int = Field(description="Number of chunks written (0 when skipped).")


class IngestResult(BaseModel):
    """Aggregate outcome of an ingestion run."""

    files: list[FileIngestResult] = Field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        """Total chunks written across all files in this run."""
        return sum(f.chunks for f in self.files)

    @property
    def files_ingested(self) -> int:
        """Number of files that were newly ingested or updated."""
        return sum(1 for f in self.files if f.status in ("ingested", "updated"))

    @property
    def files_skipped(self) -> int:
        """Number of files skipped because their content was unchanged."""
        return sum(1 for f in self.files if f.status == "skipped")
