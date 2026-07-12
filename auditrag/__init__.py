"""AuditRAG: RAG answers you can actually verify.

Every answer carries sentence-level citations that resolve to exact source
chunks with page numbers, and can be checked by an independent faithfulness
pass. This package currently implements the ingestion pipeline (milestone 1):
document loading, page-aware chunking, and indexing into ChromaDB with a
SQLite chunk registry as the canonical source of truth.
"""

from auditrag.config import Settings
from auditrag.models import Chunk, Document, IngestResult

__version__ = "0.1.0"

__all__ = ["Settings", "Chunk", "Document", "IngestResult", "__version__"]
