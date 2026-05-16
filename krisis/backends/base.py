"""
krisis/backends/base.py

Abstract interface for LLM providers. Benchmark calls evaluate_batch() over
PatientRecord chunks; backends own prompting, inference, and raw text capture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krisis.data.base import PatientRecord, Task


@dataclass
class BackendResponse:
    """Structured output from a single model call."""

    prediction: int | str | None
    abstained: bool
    confidence: float | None
    raw_response: str
    input_tokens: float | None = None
    output_tokens: float | None = None
    total_tokens: float | None = None


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
