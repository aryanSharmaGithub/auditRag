"""AuditRAG HTTP API.

Endpoints:

* ``GET /health`` — index statistics and version.
* ``POST /query`` — ranked chunks with full provenance for a question.

Error mapping: an empty index is ``409 Conflict`` (the request is fine, the
system state isn't), embedding-backend failures are ``502 Bad Gateway``, and
request validation problems are FastAPI's standard ``422``.
"""

from __future__ import annotations

from chromadb.api.types import Documents, EmbeddingFunction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from auditrag import __version__
from auditrag.chunk_store import ChunkStore
from auditrag.config import Settings
from auditrag.models import RetrievalResult
from auditrag.retrieval import EmptyIndexError, RetrievalError, Retriever


class QueryRequest(BaseModel):
    """Body of ``POST /query``."""

    question: str = Field(min_length=1, description="Natural-language question.")
    top_k: int = Field(default=6, ge=1, le=50, description="Maximum chunks to return.")


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: str = Field(description="'ok' when the service is up.")
    version: str = Field(description="AuditRAG version.")
    documents: int = Field(description="Documents in the chunk registry.")
    chunks: int = Field(description="Chunks in the chunk registry.")


def create_app(
    settings: Settings | None = None,
    embedding_function: EmbeddingFunction[Documents] | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Loaded settings; defaults to :meth:`Settings.load` semantics
            (``auditrag.yaml`` in the working directory, else defaults).
        embedding_function: Optional override, mainly for tests.

    Returns:
        A configured FastAPI app. The retriever (and its ChromaDB handle) is
        created lazily on first query, so the app starts even when the data
        directory does not exist yet.
    """
    app_settings = settings if settings is not None else Settings.load()

    app = FastAPI(
        title="AuditRAG",
        version=__version__,
        description="RAG answers you can verify.",
    )
    state: dict[str, Retriever] = {}

    def get_retriever() -> Retriever:
        if "retriever" not in state:
            state["retriever"] = Retriever(app_settings, embedding_function)
        return state["retriever"]

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

    return app
