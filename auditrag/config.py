"""Configuration loading for AuditRAG.

Settings are read from a single YAML file (``auditrag.yaml`` in the working
directory by default). Every field has a sensible default so the framework
runs locally with zero configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_FILENAME = "auditrag.yaml"


class ChunkingSettings(BaseModel):
    """Controls how page text is split into chunks."""

    max_chars: int = Field(
        default=1200,
        description="Maximum characters per chunk. Sentences are packed greedily up to this size.",
    )


class EmbeddingSettings(BaseModel):
    """Controls which embedding backend indexes chunks into ChromaDB.

    ``local`` uses ChromaDB's built-in ONNX MiniLM model (zero setup, no API
    key, model downloads on first use). ``openai`` sends chunks to any
    OpenAI-compatible ``/embeddings`` endpoint (OpenAI, Ollama, vLLM, ...).
    """

    provider: Literal["local", "openai"] = Field(
        default="local",
        description="Embedding backend: 'local' (built-in model) or 'openai' (any OpenAI-compatible endpoint).",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL of the OpenAI-compatible endpoint, e.g. 'http://localhost:11434/v1'. None uses the OpenAI default.",
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name, used only when provider is 'openai'.",
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Name of the environment variable holding the API key.",
    )

    def resolve_api_key(self) -> str:
        """Return the API key from the configured environment variable.

        Falls back to a placeholder because local endpoints (Ollama, vLLM)
        accept any non-empty key.
        """
        return os.environ.get(self.api_key_env) or "not-set"


class StorageSettings(BaseModel):
    """Controls where AuditRAG persists its indexes."""

    data_dir: str = Field(
        default=".auditrag",
        description="Directory holding the SQLite chunk registry and the ChromaDB index.",
    )
    collection: str = Field(
        default="auditrag_chunks",
        description="Name of the ChromaDB collection.",
    )


class Settings(BaseModel):
    """Top-level AuditRAG settings."""

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    @property
    def data_dir(self) -> Path:
        """Resolved data directory as a :class:`~pathlib.Path`."""
        return Path(self.storage.data_dir).expanduser().resolve()

    @property
    def chunk_db_path(self) -> Path:
        """Path of the SQLite chunk registry."""
        return self.data_dir / "chunks.db"

    @property
    def chroma_path(self) -> Path:
        """Path of the persistent ChromaDB directory."""
        return self.data_dir / "chroma"

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        """Load settings from a YAML file, falling back to defaults.

        Args:
            config_path: Explicit path to a config file. When ``None``,
                ``auditrag.yaml`` in the current working directory is used if
                it exists; otherwise defaults apply.

        Returns:
            A validated :class:`Settings` instance.

        Raises:
            FileNotFoundError: If an explicit ``config_path`` does not exist.
        """
        if config_path is not None:
            path = Path(config_path)
            if not path.is_file():
                raise FileNotFoundError(f"Config file not found: {path}")
        else:
            candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
            path = candidate if candidate.is_file() else None

        if path is None:
            return cls()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
