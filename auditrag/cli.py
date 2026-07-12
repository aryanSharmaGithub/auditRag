"""AuditRAG command-line interface.

Usage::

    auditrag ingest ./docs [--config auditrag.yaml]
    auditrag search "What is the retention period?" [--top-k 6]
    auditrag serve [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auditrag import __version__
from auditrag.config import Settings


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="auditrag",
        description="AuditRAG: RAG answers you can actually verify.",
    )
    parser.add_argument("--version", action="version", version=f"auditrag {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest documents (.pdf, .md, .txt) into the local index.",
        description=(
            "Load documents, chunk them with page-level provenance, and index "
            "them into the SQLite chunk registry and ChromaDB. Idempotent: "
            "unchanged files are skipped, changed files are re-indexed."
        ),
    )
    ingest.add_argument("path", type=Path, help="File or directory to ingest.")
    _add_config_arg(ingest)

    search = subparsers.add_parser(
        "search",
        help="Retrieve the most relevant chunks for a question (no LLM).",
        description=(
            "Run citation-tracked retrieval and print ranked chunks with their "
            "provenance. Useful for judging retrieval quality on its own."
        ),
    )
    search.add_argument("question", type=str, help="Natural-language question.")
    search.add_argument(
        "--top-k", type=int, default=6, help="Maximum chunks to return (default: 6)."
    )
    _add_config_arg(search)

    serve = subparsers.add_parser(
        "serve",
        help="Run the AuditRAG HTTP API.",
        description="Start the FastAPI server (uvicorn).",
    )
    serve.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    _add_config_arg(serve)

    return parser


def _add_config_arg(subparser: argparse.ArgumentParser) -> None:
    """Attach the shared --config option to a subcommand."""
    subparser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to auditrag.yaml (default: ./auditrag.yaml if present).",
    )


def _run_ingest(path: Path, config: Path | None) -> int:
    """Execute the ingest command. Returns a process exit code."""
    from auditrag.ingest import ingest_path

    settings = Settings.load(config)
    print(f"Ingesting {path} → {settings.data_dir}")

    result = ingest_path(path, settings)

    for file_result in result.files:
        label = {"ingested": "+", "updated": "~", "skipped": "="}[file_result.status]
        detail = f"{file_result.chunks} chunks" if file_result.chunks else file_result.status
        print(f"  {label} {Path(file_result.path).name}  [{file_result.doc_id}]  {detail}")

    print(
        f"Done: {result.files_ingested} file(s) indexed, "
        f"{result.files_skipped} unchanged, {result.total_chunks} chunk(s) written."
    )
    return 0


def _run_search(question: str, top_k: int, config: Path | None) -> int:
    """Execute the search command. Returns a process exit code."""
    from auditrag.retrieval import RetrievalError, Retriever

    settings = Settings.load(config)
    try:
        result = Retriever(settings).search(question, top_k=top_k)
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not result.chunks:
        print("No results.")
        return 0

    for hit in result.chunks:
        chunk = hit.chunk
        print(f"#{hit.rank + 1}  {chunk.doc_name} p.{chunk.page}  "
              f"[{chunk.chunk_id}]  score={hit.score:.3f}")
        print(f"    {chunk.text[:200].replace(chr(10), ' ')}\n")
    return 0


def _run_serve(host: str, port: int, config: Path | None) -> int:
    """Execute the serve command. Returns a process exit code."""
    import uvicorn

    from auditrag.api import create_app

    app = create_app(Settings.load(config))
    uvicorn.run(app, host=host, port=port)
    return 0


def main() -> None:
    """Console-script entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            sys.exit(_run_ingest(args.path, args.config))
        if args.command == "search":
            sys.exit(_run_search(args.question, args.top_k, args.config))
        if args.command == "serve":
            sys.exit(_run_serve(args.host, args.port, args.config))
        parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
