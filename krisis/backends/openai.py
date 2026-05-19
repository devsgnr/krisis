"""
krisis/backends/openai.py

OpenAI Chat Completions backend with structured clinical outputs.
"""

from __future__ import annotations

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

DEFAULT_OPENAI_MODEL = "gpt-5.5"


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
                    "confidence": {
                        "type": ["number", "null"],
                    },
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


class OpenAIBackend(BaseBackend):
    """
    Chat Completions API wrapper.

    Requires: ``pip install krisis[openai]`` (brings in the ``openai`` SDK).

    Args:
        model: model id passed through to the API
        temperature: sampling temperature (0.0 recommended for evals)
        max_completion_tokens: optional cap for completion tokens
        api_key: optional OpenAI API key; falls back to environment variables
        client: optional pre-built ``openai.OpenAI`` client for injection/tests
        max_retries: number of retries after transient provider failures
        retry_base_seconds: initial exponential-backoff delay
        retry_max_seconds: maximum exponential-backoff delay
        **client_kwargs: forwarded to ``OpenAI()`` when client is omitted
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float | None = None,
        max_completion_tokens: int | None = 1024,
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
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "OpenAIBackend requires the openai package. "
                    "Install with: pip install 'krisis[openai]'"
                ) from exc
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            self._client = OpenAI(**client_kwargs)
        self._model = model
        self._temperature = temperature
        self._max_completion_tokens = max_completion_tokens
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        messages = build_messages(record, task)
        prompt = format_messages_for_audit(messages)
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "response_format": _response_format_for_task(task),
            "store": False,
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._max_completion_tokens is not None:
            request["max_completion_tokens"] = self._max_completion_tokens

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
                    token_cap_name="max_completion_tokens",
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

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "response_format": _batch_response_format_for_task(task),
            "store": False,
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._max_completion_tokens is not None:
            request["max_completion_tokens"] = self._max_completion_tokens * len(
                records
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
                    token_cap_name="max_completion_tokens",
                    mode="batch",
                    finish_reason=getattr(choice_obj, "finish_reason", None),
                )
            )
        responses = parse_batch_response(raw, task, len(records))
        attach_prompt_metadata(responses, prompt=prompt, prompt_mode="batch")
        usage = usage_from_openai_compatible_response(completion)
        return distribute_usage_over_batch(responses, usage)


def make_openai_backend(
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> OpenAIBackend:
    """
    Convenience factory mirroring common env-var based setup.

    ``api_key`` is passed through to the OpenAI client when provided.
    """
    return OpenAIBackend(
        model=model, temperature=temperature, api_key=api_key, **kwargs
    )
