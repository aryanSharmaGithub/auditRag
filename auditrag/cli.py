"""AuditRAG command-line interface.

Usage::

    auditrag ingest ./docs [--config auditrag.yaml]
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
    ingest.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to auditrag.yaml (default: ./auditrag.yaml if present).",
    )
    return parser


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


def main() -> None:
    """Console-script entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            sys.exit(_run_ingest(args.path, args.config))
        parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
