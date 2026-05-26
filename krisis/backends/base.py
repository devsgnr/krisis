"""
krisis/backends/base.py

Abstract interface for LLM providers. Benchmark calls evaluate_batch() over
PatientRecord chunks; backends own prompting, inference, and raw text capture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from krisis.data.base import PatientRecord, Task


class BackendResponse(BaseModel):
    """Structured output from one evaluated row."""

    prediction: int | str | None
    abstained: bool
    confidence: float | None
    raw_response: str
    prompt: list[dict[str, str]] = Field(default_factory=list)
    prompt_mode: str = ""
    input_tokens: float | None = None
    output_tokens: float | None = None
    total_tokens: float | None = None


def format_messages_for_audit(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return provider prompt messages with patient data redacted."""
    return [_redact_message_for_audit(message) for message in messages]


def empty_response_message(
    backend_name: str,
    *,
    token_cap_name: str,
    mode: str,
    finish_reason: str | None = None,
) -> str:
    """Build an actionable error for provider responses with no text content."""
    response_kind = "batched response" if mode == "batch" else "response"
    hint = (
        f"{backend_name} returned an empty {response_kind}. "
        "This can happen when the provider returns no text or the output-token "
        f"cap is too low. Increase the backend {token_cap_name} setting "
        "(or the example runner's --max-output-tokens flag)"
    )
    if mode == "batch":
        hint += " or reduce --batch-size"
    if finish_reason:
        hint += f". Provider finish_reason={finish_reason!r}"
    return f"{hint}."


def _redact_message_for_audit(message: dict[str, Any]) -> dict[str, str]:
    role = message.get("role", "")
    content = message.get("content", "")
    if role != "user":
        return {"role": role, "content": content}

    if content.startswith("Evaluate the following cases as an independent batch."):
        return {
            "role": role,
            "content": (
                "Evaluate the following cases as an independent batch.\n\n"
                "[BATCH_PATIENT_DATA_REDACTED]\n\n"
                "Return the JSON object as specified in the system message."
            ),
        }

    if content.startswith("Patient markers:\n"):
        return {
            "role": role,
            "content": (
                "Patient markers:\n"
                "[PATIENT_MARKERS_REDACTED]\n\n"
                "Return the JSON object as specified in the system message."
            ),
        }

    if "Baseline visit:" in content and "Current visit:" in content:
        return {
            "role": role,
            "content": (
                "Visits are [TRAJECTORY_INTERVAL_REDACTED] months apart.\n\n"
                "Baseline visit:\n"
                "[BASELINE_VISIT_REDACTED]\n\n"
                "Current visit:\n"
                "[CURRENT_VISIT_REDACTED]\n\n"
                "Return the JSON object as specified in the system message."
            ),
        }

    return {"role": role, "content": "[USER_PROMPT_REDACTED]"}


class BaseBackend(ABC):
    """Provider-agnostic contract for clinical benchmark inference."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging and BenchmarkResult (e.g. 'openai')."""

    @abstractmethod
    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        """
        Run the model on one patient row.

        Implementations should preserve the full model text in raw_response
        for qualitative review; abstained must be True when the model
        declines to commit to a prediction.
        """
        ...

    def evaluate_batch(
        self,
        records: list[PatientRecord],
        task: Task,
    ) -> list[BackendResponse]:
        """
        Run the model on a batch of patient rows.

        Backends can override this for provider-native batched prompts. The
        default keeps compatibility by looping over evaluate().
        """
        return [self.evaluate(record, task) for record in records]
