# AuditRAG

**RAG answers you can actually verify.**

Most RAG stacks give you an answer and a vibe. AuditRAG gives you an answer,
sentence-level citations that resolve to exact source chunks with page
numbers, an independent faithfulness check that flags unsupported claims, and
a timestamped PDF evidence report you can hand to an auditor.

Runs locally with zero setup: SQLite + ChromaDB, any OpenAI-compatible LLM
endpoint (OpenAI, Ollama, vLLM, LM Studio, ...).

> **Status: early development.** Milestone 1 (ingestion pipeline) is
> implemented. Retrieval, cited generation, faithfulness verification, and
> evidence export are in progress — see the roadmap below.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
auditrag ingest ./docs
```

Ingests every `.pdf`, `.md`, and `.txt` under `./docs`: documents are split
into page-aware chunks and indexed into a local ChromaDB collection, with a
SQLite chunk registry as the canonical record of every chunk's provenance
(source file, page number, character offsets).

Ingestion is idempotent — unchanged files are skipped, modified files are
re-indexed in place.

### Configuration

Copy [auditrag.example.yaml](auditrag.example.yaml) to `auditrag.yaml`. All
fields are optional; the defaults use a local embedding model with no API key.
To embed via any OpenAI-compatible endpoint instead:

```yaml
embedding:
  provider: openai
  base_url: http://localhost:11434/v1   # e.g. Ollama
  model: nomic-embed-text
```

## Design notes

- **Chunk IDs are minted once and survive the whole pipeline.** Format:
  `{doc_hash}:{page}:{chunk_index}` — deterministic, human-decodable, and
  stable across re-ingestion of unchanged files.
- **SQLite is the source of truth for chunk content.** ChromaDB holds only
  embeddings keyed by the same IDs. Citation resolution and evidence reports
  never depend on vector-store internals, which also keeps the vector store
  swappable.
- **Chunks never cross page boundaries**, so every citation carries a single
  exact page number.

## Roadmap

1. ✅ Ingestion: loading, page-aware chunking, ChromaDB + SQLite indexing
2. Vector retrieval endpoint (FastAPI)
3. Cited generation: sentence-level `[n]` citations mapped to chunk IDs
4. Hybrid search (BM25 + vector, reciprocal rank fusion)
5. Faithfulness verification pass
6. Web UI with citation hover cards and verdict badges
7. Timestamped PDF evidence reports

## Development

```bash
pytest
```

Tests run fully offline using a dummy embedding function.

## License

MIT
# auditRag
