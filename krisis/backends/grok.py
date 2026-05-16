"""
krisis/backends/grok.py

xAI Grok backend using the OpenAI-compatible Chat Completions API.
"""

from __future__ import annotations

import os
from typing import Any

from krisis.backends.base import BackendResponse, BaseBackend
from krisis.backends.batching import (
    batch_response_schema,
    build_batch_messages,
    distribute_usage_over_batch,
    parse_batch_response,
    prediction_schema_for_task,
)
from krisis.backends.retry import call_with_retries
from krisis.backends.usage import usage_from_openai_compatible_response
from krisis.data.base import PatientRecord, Task
from krisis.prompts.base import build_messages
from krisis.tasks.base import parse_model_response

DEFAULT_GROK_MODEL = "grok-4.3"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"


def _response_format_for_task(task: Task) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"krisis_{task.value}_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "abstained": {"type": "boolean"},
                    "confidence": {"type": ["number", "null"]},
                    "prediction": prediction_schema_for_task(task),
                },
                "required": ["abstained", "confidence", "prediction"],
            },
        },
    }


def _batch_response_format_for_task(task: Task) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"krisis_{task.value}_batch_evaluation",
            "strict": True,
            "schema": batch_response_schema(task),
        },
    }


class GrokBackend(BaseBackend):
    """
    xAI Grok Chat Completions API wrapper.

    Requires the OpenAI SDK because xAI exposes an OpenAI-compatible endpoint.

    Args:
        model: model id passed through to xAI
        temperature: sampling temperature (0.0 recommended for evals)
        max_tokens: optional cap for generated tokens
        api_key: optional xAI API key; falls back to ``XAI_API_KEY``
        base_url: xAI API base URL
        client: optional pre-built OpenAI-compatible client for tests
        **kwargs: forwarded to ``OpenAI()`` when client is omitted
    """

    def __init__(
        self,
        model: str = DEFAULT_GROK_MODEL,
        temperature: float | None = None,
        max_tokens: int | None = 256,
        api_key: str | None = None,
        base_url: str = DEFAULT_XAI_BASE_URL,
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
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "GrokBackend requires the openai package because xAI "
                    "uses an OpenAI-compatible API. Install with: "
                    "pip install 'krisis[grok]'"
                ) from exc

            key = api_key or os.getenv("XAI_API_KEY")
            if key is not None:
                client_kwargs["api_key"] = key
            client_kwargs.setdefault("base_url", base_url)
            self._client = OpenAI(**client_kwargs)

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds

    @property
    def name(self) -> str:
        return f"grok:{self._model}"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": build_messages(record, task),
            "response_format": _response_format_for_task(task),
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._max_tokens is not None:
            request["max_tokens"] = self._max_tokens

        completion = call_with_retries(
            lambda: self._client.chat.completions.create(**request),
            max_retries=self._max_retries,
            base_delay_seconds=self._retry_base_seconds,
            max_delay_seconds=self._retry_max_seconds,
        )
        choice = completion.choices[0].message
        raw = (choice.content or "").strip()
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
        usage = usage_from_openai_compatible_response(completion)
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

        request: dict[str, Any] = {
            "model": self._model,
            "messages": build_batch_messages(records, task),
            "response_format": _batch_response_format_for_task(task),
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._max_tokens is not None:
            request["max_tokens"] = self._max_tokens * len(records)

        completion = call_with_retries(
            lambda: self._client.chat.completions.create(**request),
            max_retries=self._max_retries,
            base_delay_seconds=self._retry_base_seconds,
            max_delay_seconds=self._retry_max_seconds,
        )
        choice = completion.choices[0].message
        raw = (choice.content or "").strip()
        responses = parse_batch_response(raw, task, len(records))
        usage = usage_from_openai_compatible_response(completion)
        return distribute_usage_over_batch(responses, usage)


def make_grok_backend(
    model: str = DEFAULT_GROK_MODEL,
    temperature: float | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> GrokBackend:
    """Convenience factory for xAI Grok setup."""
    return GrokBackend(model=model, temperature=temperature, api_key=api_key, **kwargs)
