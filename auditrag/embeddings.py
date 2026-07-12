"""Embedding backends for indexing chunks into ChromaDB.

Two backends are supported:

* ``local`` — ChromaDB's built-in ONNX MiniLM model. Zero setup, no API key;
  the model is downloaded automatically on first use.
* ``openai`` — any OpenAI-compatible ``/embeddings`` endpoint (OpenAI, Ollama,
  vLLM, LM Studio, ...), configured via ``auditrag.yaml``.
"""

from __future__ import annotations

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

from auditrag.config import EmbeddingSettings


class OpenAICompatibleEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB embedding function backed by an OpenAI-compatible endpoint."""

    def __init__(
        self, model: str, base_url: str | None, api_key: str, api_key_env: str = "OPENAI_API_KEY"
    ) -> None:
        """Create the embedding function.

        Args:
            model: Embedding model name, e.g. ``text-embedding-3-small``.
            base_url: Endpoint base URL (``None`` for api.openai.com).
            api_key: API key; local endpoints accept any non-empty value.
            api_key_env: Environment variable the key came from, recorded so
                the function can be rebuilt from persisted collection config.
        """
        # Imported lazily so the 'local' provider works without the openai
        # package being importable at module load time.
        from openai import OpenAI

        self._model = model
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a batch of texts, preserving input order."""
        response = self._client.embeddings.create(model=self._model, input=list(input))
        return [item.embedding for item in response.data]

    @staticmethod
    def name() -> str:
        """Identifier ChromaDB uses when persisting collection config."""
        return "auditrag-openai-compatible"

    def get_config(self) -> dict[str, str | None]:
        """Serializable config ChromaDB persists with the collection.

        The API key is deliberately excluded: it must always come from the
        environment, never from persisted collection metadata.
        """
        return {
            "model": self._model,
            "base_url": self._base_url,
            "api_key_env": self._api_key_env,
        }

    @staticmethod
    def build_from_config(config: dict[str, str | None]) -> "OpenAICompatibleEmbeddingFunction":
        """Rebuild the function from persisted config, reading the key from the env."""
        import os

        api_key_env = config.get("api_key_env") or "OPENAI_API_KEY"
        return OpenAICompatibleEmbeddingFunction(
            model=config["model"] or "text-embedding-3-small",
            base_url=config.get("base_url"),
            api_key=os.environ.get(api_key_env) or "not-set",
            api_key_env=api_key_env,
        )


def build_embedding_function(settings: EmbeddingSettings) -> EmbeddingFunction[Documents]:
    """Construct the embedding function selected by the configuration.

    Args:
        settings: The ``embedding`` section of the AuditRAG config.

    Returns:
        A ChromaDB-compatible embedding function.
    """
    if settings.provider == "openai":
        return OpenAICompatibleEmbeddingFunction(
            model=settings.model,
            base_url=settings.base_url,
            api_key=settings.resolve_api_key(),
            api_key_env=settings.api_key_env,
        )
    return embedding_functions.DefaultEmbeddingFunction()  # type: ignore[return-value]
