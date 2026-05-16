"""
krisis/backends/anthropic.py

Anthropic Messages API backend with JSON clinical outputs.
"""

from __future__ import annotations

from typing import Any

from krisis.backends.base import BackendResponse, BaseBackend
from krisis.backends.batching import (
    build_batch_messages,
    distribute_usage_over_batch,
    parse_batch_response,
)
from krisis.backends.retry import call_with_retries
from krisis.backends.usage import usage_from_anthropic_response
from krisis.data.base import PatientRecord, Task
from krisis.prompts.base import build_messages
from krisis.tasks.base import parse_model_response

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"


def _split_system_and_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """Convert Krisis chat messages into Anthropic's system + messages shape."""
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, str]] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_parts.append(content)
        else:
            anthropic_messages.append({"role": role, "content": content})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, anthropic_messages


def _extract_text(response: Any) -> str:
    """Extract text from an Anthropic Messages API response."""
    chunks: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(str(text))
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    return "".join(chunks).strip()


class AnthropicBackend(BaseBackend):
    """
    Anthropic Messages API wrapper.

    Requires: ``pip install krisis[anthropic]``.

    Args:
        model: model id passed through to the API
        temperature: sampling temperature (0.0 recommended for evals)
        max_tokens: cap for generated tokens
        api_key: optional Anthropic API key; falls back to environment variables
        client: optional pre-built ``anthropic.Anthropic`` client for tests
        **kwargs: forwarded to ``Anthropic()`` when client is omitted
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float | None = None,
        max_tokens: int = 256,
        api_key: str | None = None,
        client: Any | None = None,
        max_retries: int = 2,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 8.0,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "AnthropicBackend requires the anthropic package. "
                    "Install with: pip install 'krisis[anthropic]'"
                ) from exc
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            self._client = Anthropic(**client_kwargs)

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        system, messages = _split_system_and_messages(build_messages(record, task))
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system is not None:
            request["system"] = system
        if self._temperature is not None:
            request["temperature"] = self._temperature

        response = call_with_retries(
            lambda: self._client.messages.create(**request),
            max_retries=self._max_retries,
            base_delay_seconds=self._retry_base_seconds,
            max_delay_seconds=self._retry_max_seconds,
        )
        raw = _extract_text(response)
        if not raw:
            raise ValueError(f"{self.name} returned an empty response.")
        parsed = parse_model_response(raw, task)
        if (
            parsed.prediction is None
            and not parsed.abstained
            and parsed.confidence is None
        ):
            raise ValueError(
                f"{self.name} returned a non-JSON response that could not be parsed."
            )
        usage = usage_from_anthropic_response(response)
        return BackendResponse(
            prediction=parsed.prediction,
            abstained=parsed.abstained,
            confidence=parsed.confidence,
            raw_response=raw,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    def evaluate_batch(
        self,
        records: list[PatientRecord],
        task: Task,
    ) -> list[BackendResponse]:
        if not records:
            return []

        system, messages = _split_system_and_messages(
            build_batch_messages(records, task)
        )
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens * len(records),
            "messages": messages,
        }
        if system is not None:
            request["system"] = system
        if self._temperature is not None:
            request["temperature"] = self._temperature

        response = call_with_retries(
            lambda: self._client.messages.create(**request),
            max_retries=self._max_retries,
            base_delay_seconds=self._retry_base_seconds,
            max_delay_seconds=self._retry_max_seconds,
        )
        raw = _extract_text(response)
        responses = parse_batch_response(raw, task, len(records))
        usage = usage_from_anthropic_response(response)
        return distribute_usage_over_batch(responses, usage)


def make_anthropic_backend(
    model: str = DEFAULT_ANTHROPIC_MODEL,
    temperature: float | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> AnthropicBackend:
    """
    Convenience factory mirroring common env-var based setup.

    ``api_key`` is passed through to the Anthropic client when provided.
    """
    return AnthropicBackend(
        model=model, temperature=temperature, api_key=api_key, **kwargs
    )
