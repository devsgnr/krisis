"""
krisis/backends/api.py

Unified API backend for model-provider routing through OpenRouter.
"""

from __future__ import annotations

import os
from typing import Any

from krisis.backends.base import (
    BackendResponse,
    BaseBackend,
    empty_response_message,
    format_messages_for_audit,
)
from krisis.backends.batching import (
    attach_prompt_metadata,
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

DEFAULT_API_MODEL = "openai/gpt-5.5"
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REASONING_EFFORT = "low"


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


class APIBackend(BaseBackend):
    """
    OpenRouter-backed API backend.

    OpenRouter exposes an OpenAI-compatible API, so Krisis can evaluate
    OpenAI, Anthropic, xAI, Google, and other routed models by changing only
    the model id, for example ``openai/gpt-5.5`` or
    ``anthropic/claude-4.7-opus``.

    Args:
        model: API model id routed through OpenRouter
        temperature: sampling temperature (0.0 recommended for evals)
        max_tokens: optional per-row cap for generated tokens
        reasoning_effort: reasoning effort; defaults to ``low``
        exclude_reasoning: keep provider reasoning out of response text
        api_key: optional key; falls back to ``OPENROUTER_API_KEY``
        base_url: API base URL
        client: optional pre-built OpenAI-compatible client for tests
        max_retries: number of retries after transient provider failures
        retry_base_seconds: initial exponential-backoff delay
        retry_max_seconds: maximum exponential-backoff delay
        **client_kwargs: forwarded to ``OpenAI()`` when client is omitted
    """

    def __init__(
        self,
        model: str = DEFAULT_API_MODEL,
        temperature: float | None = None,
        max_tokens: int | None = 1024,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        exclude_reasoning: bool = True,
        api_key: str | None = None,
        base_url: str = DEFAULT_API_BASE_URL,
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
                    "APIBackend requires the openai package because "
                    "OpenRouter exposes an OpenAI-compatible API. Install "
                    "with: pip install 'krisis[api]'"
                ) from exc

            key = api_key or os.getenv("OPENROUTER_API_KEY")
            if key is not None:
                client_kwargs["api_key"] = key
            client_kwargs.setdefault("base_url", base_url)
            self._client = OpenAI(**client_kwargs)

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._exclude_reasoning = exclude_reasoning
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds

    @property
    def name(self) -> str:
        return f"api:{self._model}"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        messages = build_messages(record, task)
        prompt = format_messages_for_audit(messages)
        request = self._request(
            messages=messages,
            response_format=_response_format_for_task(task),
            max_tokens_multiplier=1,
        )

        completion = call_with_retries(
            lambda: self._client.chat.completions.create(**request),
            max_retries=self._max_retries,
            base_delay_seconds=self._retry_base_seconds,
            max_delay_seconds=self._retry_max_seconds,
        )
        choice_obj = completion.choices[0]
        choice = choice_obj.message
        raw = (choice.content or "").strip()
        if not raw:
            raise ValueError(
                empty_response_message(
                    self.name,
                    token_cap_name="max_tokens",
                    mode="single",
                    finish_reason=getattr(choice_obj, "finish_reason", None),
                )
            )
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
            prompt=prompt,
            prompt_mode="single",
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

        messages = build_batch_messages(records, task)
        prompt = format_messages_for_audit(messages)
        request = self._request(
            messages=messages,
            response_format=_batch_response_format_for_task(task),
            max_tokens_multiplier=len(records),
        )

        completion = call_with_retries(
            lambda: self._client.chat.completions.create(**request),
            max_retries=self._max_retries,
            base_delay_seconds=self._retry_base_seconds,
            max_delay_seconds=self._retry_max_seconds,
        )
        choice_obj = completion.choices[0]
        choice = choice_obj.message
        raw = (choice.content or "").strip()
        if not raw:
            raise ValueError(
                empty_response_message(
                    self.name,
                    token_cap_name="max_tokens",
                    mode="batch",
                    finish_reason=getattr(choice_obj, "finish_reason", None),
                )
            )

        responses = parse_batch_response(raw, task, len(records))
        attach_prompt_metadata(responses, prompt=prompt, prompt_mode="batch")
        usage = usage_from_openai_compatible_response(completion)
        return distribute_usage_over_batch(responses, usage)

    def _request(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        max_tokens_multiplier: int,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "response_format": response_format,
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._max_tokens is not None:
            request["max_tokens"] = self._max_tokens * max_tokens_multiplier
        if self._reasoning_effort is not None:
            request["extra_body"] = {
                "reasoning": {
                    "effort": self._reasoning_effort,
                    "exclude": self._exclude_reasoning,
                }
            }
        return request


def make_api_backend(
    model: str = DEFAULT_API_MODEL,
    temperature: float | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    api_key: str | None = None,
    **kwargs: Any,
) -> APIBackend:
    """Convenience factory for the default API backend setup."""
    return APIBackend(
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        api_key=api_key,
        **kwargs,
    )
