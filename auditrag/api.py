"""AuditRAG HTTP API.

Endpoints:

* ``GET /`` — the single-page web UI.
* ``GET /health`` — index statistics and version.
* ``POST /query`` — ranked chunks with full provenance for a question.
* ``POST /ask`` — a generated answer with sentence-level citations.
* ``POST /report`` — a timestamped PDF evidence report for a Q&A session.

Error mapping: an empty index is ``409 Conflict`` (the request is fine, the
system state isn't), embedding-backend and LLM failures are ``502 Bad
Gateway``, and request validation problems are FastAPI's standard ``422``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from chromadb.api.types import Documents, EmbeddingFunction
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

_INDEX_HTML = Path(__file__).parent / "web" / "index.html"

from auditrag import __version__
from auditrag.answer import generate_answer
from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.llm import LLMClient, LLMError
from auditrag.models import Answer, RetrievalResult
from auditrag.retrieval import EmptyIndexError, RetrievalError, Retriever


class QueryRequest(BaseModel):
    """Body of ``POST /query``."""

    question: str = Field(min_length=1, description="Natural-language question.")
    top_k: int = Field(default=6, ge=1, le=50, description="Maximum chunks to return.")


class AskRequest(QueryRequest):
    """Body of ``POST /ask``."""

    verify: bool = Field(
        default=False,
        description="Run the faithfulness pass (one extra LLM call) and attach "
        "per-claim verdicts.",
    )


class ReportRequest(BaseModel):
    """Body of ``POST /report``.

    Stateless by design: the client sends back the ``Answer`` objects it
    received from ``/ask``. Claims and verdicts are reproduced as sent, but
    all evidence text is re-fetched from the chunk registry by ID at render
    time — a tampered payload cannot forge what a citation resolves to.
    """

    answers: list[Answer] = Field(
        min_length=1, description="Answers from /ask, in session order."
    )


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: str = Field(description="'ok' when the service is up.")
    version: str = Field(description="AuditRAG version.")
    documents: int = Field(description="Documents in the chunk registry.")
    chunks: int = Field(description="Chunks in the chunk registry.")


def create_app(
    settings: Settings | None = None,
    embedding_function: EmbeddingFunction[Documents] | None = None,
    llm_client: LLMClient | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Loaded settings; defaults to :meth:`Settings.load` semantics
            (``auditrag.yaml`` in the working directory, else defaults).
        embedding_function: Optional override, mainly for tests.
        llm_client: Optional override, mainly for tests.

    Returns:
        A configured FastAPI app. The retriever (and its ChromaDB handle) and
        the LLM client are created lazily on first use, so the app starts
        even when the data directory does not exist yet.
    """
    app_settings = settings if settings is not None else Settings.load()

    app = FastAPI(
        title="AuditRAG",
        version=__version__,
        description="RAG answers you can verify.",
    )
    state: dict[str, object] = {}

    def get_retriever() -> Retriever:
        if "retriever" not in state:
            state["retriever"] = Retriever(app_settings, embedding_function)
        return state["retriever"]  # type: ignore[return-value]

    def get_llm() -> LLMClient:
        if llm_client is not None:
            return llm_client
        if "llm" not in state:
            state["llm"] = LLMClient(app_settings.llm)
        return state["llm"]  # type: ignore[return-value]

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the single-page web UI."""
        return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report index statistics; usable as a liveness probe."""
        if app_settings.chunk_db_path.exists():
            with ChunkStore(app_settings.chunk_db_path) as store:
                documents, chunks = store.count_documents(), store.count_chunks()
        else:
            documents = chunks = 0
        return HealthResponse(
            status="ok", version=__version__, documents=documents, chunks=chunks
        )

    @app.post("/query", response_model=RetrievalResult)
    def query(request: QueryRequest) -> RetrievalResult:
        """Return ranked, provenance-complete chunks for a question."""
        try:
            return get_retriever().search(request.question, top_k=request.top_k)
        except EmptyIndexError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RetrievalError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/ask", response_model=Answer)
    def ask(request: AskRequest) -> Answer:
        """Return a generated answer with sentence-level citations.

        With ``verify: true``, each claim also carries a faithfulness verdict.
        """
        try:
            return generate_answer(
                request.question,
                app_settings,
                top_k=request.top_k,
                verify=request.verify,
                retriever=get_retriever(),
                llm_client=get_llm(),
            )
        except EmptyIndexError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (RetrievalError, LLMError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/report")
    def report(request: ReportRequest) -> Response:
        """Return a timestamped PDF evidence report for the given answers."""
        from auditrag.report import build_evidence_report

        pdf_bytes = build_evidence_report(request.answers, app_settings)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="auditrag-evidence-{stamp}.pdf"'
            },
        )

    return app
