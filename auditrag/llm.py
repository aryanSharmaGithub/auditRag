"""Thin client for any OpenAI-compatible chat-completions endpoint.

Deliberately minimal: one method, no streaming, no tool use. Cited generation
needs exactly one completion per question, and the faithfulness verifier
(next milestone) will reuse this same client.
"""

from __future__ import annotations

from auditrag.config import LLMSettings


class LLMError(RuntimeError):
    """The chat endpoint failed; the message is safe to show to the user."""


class LLMClient:
    """Wrapper around the OpenAI SDK pointed at the configured endpoint."""

    def __init__(self, settings: LLMSettings) -> None:
        """Create the client from the ``llm`` section of the config."""
        from openai import OpenAI

        self._settings = settings
        self._client = OpenAI(
            base_url=settings.base_url, api_key=settings.resolve_api_key()
        )

    @property
    def model(self) -> str:
        """The configured chat model name."""
        return self._settings.model

    def complete(self, system: str, user: str) -> str:
        """Run one chat completion and return the assistant text.

        Args:
            system: System prompt.
            user: User message.

        Returns:
            The assistant's reply text.

        Raises:
            LLMError: If the request fails or the model returns no content,
                with an actionable message naming the likely config fix.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
        except Exception as exc:
            raise LLMError(
                f"Chat completion failed (model '{self._settings.model}'): {exc}. "
                "Check the llm section of auditrag.yaml — endpoint URL, model "
                "name, and the API key environment variable."
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError(
                f"The model '{self._settings.model}' returned an empty answer. "
                "Retry, or try a different model in auditrag.yaml."
            )
        return content.strip()
