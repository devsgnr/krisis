"""
krisis/backends/gemini.py

Google Gemini backend with structured clinical outputs.
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
from krisis.backends.usage import usage_from_gemini_response
from krisis.data.base import PatientRecord, Task
from krisis.prompts.base import build_messages
from krisis.tasks.base import parse_model_response

DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"


def _response_schema_for_task(task: Task) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "abstained": {"type": "boolean"},
            "confidence": {"type": ["number", "null"]},
            "prediction": prediction_schema_for_task(task),
        },
        "required": ["abstained", "confidence", "prediction"],
    }


def _split_system_and_user(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            user_parts.append(message["content"])
    system = "\n\n".join(system_parts) if system_parts else None
    return system, "\n\n".join(user_parts)


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text).strip()

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text is not None:
                chunks.append(str(part_text))
    return "".join(chunks).strip()


class GeminiBackend(BaseBackend):
    """
    Google Gemini API wrapper.

    Requires: ``pip install krisis[gemini]``.

    Args:
        model: model id passed through to the API
        temperature: sampling temperature (0.0 recommended for evals)
        max_output_tokens: optional cap for generated tokens
        api_key: optional Gemini API key; falls back to ``GEMINI_API_KEY``
        client: optional pre-built ``google.genai.Client`` for injection/tests
        max_retries: number of retries after transient provider failures
        retry_base_seconds: initial exponential-backoff delay
        retry_max_seconds: maximum exponential-backoff delay
        **client_kwargs: forwarded to ``genai.Client()`` when client is omitted
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float | None = None,
        max_output_tokens: int | None = 256,
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
                from google import genai
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "GeminiBackend requires the google-genai package. "
                    "Install with: pip install 'krisis[gemini]'"
                ) from exc

            key = api_key or os.getenv("GEMINI_API_KEY")
            if key is not None:
                client_kwargs["api_key"] = key
            self._client = genai.Client(**client_kwargs)

        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        prompt_messages = build_messages(record, task)
        prompt = format_messages_for_audit(prompt_messages)
        system, contents = _split_system_and_user(prompt_messages)
        response = self._generate_content(
            contents=contents,
            schema=_response_schema_for_task(task),
            system_instruction=system,
            max_output_tokens=self._max_output_tokens,
        )
        raw = _extract_text(response)
        if not raw:
            raise ValueError(
                empty_response_message(
                    self.name,
                    token_cap_name="max_output_tokens",
                    mode="single",
                    finish_reason=_gemini_finish_reason(response),
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
        usage = usage_from_gemini_response(response)
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

        prompt_messages = build_batch_messages(records, task)
        prompt = format_messages_for_audit(prompt_messages)
        system, contents = _split_system_and_user(prompt_messages)
        max_output_tokens = (
            self._max_output_tokens * len(records)
            if self._max_output_tokens is not None
            else None
        )
        response = self._generate_content(
            contents=contents,
            schema=batch_response_schema(task),
            system_instruction=system,
            max_output_tokens=max_output_tokens,
        )
        raw = _extract_text(response)
        if not raw:
            raise ValueError(
                empty_response_message(
                    self.name,
                    token_cap_name="max_output_tokens",
                    mode="batch",
                    finish_reason=_gemini_finish_reason(response),
                )
            )
        responses = parse_batch_response(raw, task, len(records))
        attach_prompt_metadata(responses, prompt=prompt, prompt_mode="batch")
        usage = usage_from_gemini_response(response)
        return distribute_usage_over_batch(responses, usage)

    def _generate_content(
        self,
        *,
        contents: str,
        schema: dict[str, Any],
        system_instruction: str | None,
        max_output_tokens: int | None,
    ) -> Any:
        config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }
        if system_instruction is not None:
            config["system_instruction"] = system_instruction
        if self._temperature is not None:
            config["temperature"] = self._temperature
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens

        return call_with_retries(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            ),
            max_retries=self._max_retries,
            base_delay_seconds=self._retry_base_seconds,
            max_delay_seconds=self._retry_max_seconds,
        )


def make_gemini_backend(
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> GeminiBackend:
    """Convenience factory for Google Gemini setup."""
    return GeminiBackend(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def _gemini_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    return str(finish_reason) if finish_reason is not None else None
