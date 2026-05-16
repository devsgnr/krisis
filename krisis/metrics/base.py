"""
krisis/metrics/base.py

Abstract base class for all Krisis evaluation metrics.

Every metric in Krisis follows the same contract:
    - receives a list of EvaluationResult objects
    - returns a MetricScore dataclass with the computed value,
      a human-readable label, and optional breakdown by stage/class

Adding a new metric means inheriting from BaseMetric and implementing
compute(). Nothing else needs to change in the harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ── EvaluationResult ──────────────────────────────────────────────────────────


@dataclass
class EvaluationResult:
    """
    A single model evaluation result for one PatientRecord.

    prediction  — the model's answer.
                  DETECTION:   0 or 1
                  STAGING:     integer stage (1–5)
                  PROGRESSION: stage integer or direction string

    ground_truth — the correct label from PatientRecord.label

    abstained   — True if the model declined to answer.
                  A model that says "I don't have enough information
                  to make a safe clinical determination" is abstaining.
                  Abstentions are scored separately from wrong answers —
                  they represent appropriate safety behaviour.

    confidence  — optional float [0.0, 1.0] representing the model's
                  stated confidence in its prediction.
                  Used for calibration scoring.

    raw_response — the full raw text response from the model backend,
                   preserved for qualitative analysis and debugging.

    metadata    — pass-through of PatientRecord.metadata, giving metrics
                  access to egfr, ckd_stage, sex for breakdown analysis.
    """

    prediction: int | str | None
    ground_truth: int | str
    abstained: bool = False
    confidence: float | None = None
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    input_tokens: float | None = None
    output_tokens: float | None = None
    total_tokens: float | None = None


# ── MetricScore ───────────────────────────────────────────────────────────────


@dataclass
class MetricScore:
    """
    The result of computing a metric across all EvaluationResults.

    name        — metric name, e.g. 'Abstention Rate', 'Accuracy'

    value       — the primary scalar score.
                  Interpretation depends on metric:
                    accuracy        → higher is better
                    abstention_rate → context-dependent (see abstention.py)
                    ece             → lower is better

    breakdown   — optional per-class or per-stage breakdown dict.
                  e.g. {"stage_1": 0.92, "stage_2": 0.87, ...}
                  Gives researchers granular insight beyond the scalar.

    n_evaluated — number of records scored (excluding abstentions
                  where abstentions are excluded from the metric)

    n_abstained — number of records where the model abstained.
                  Always reported regardless of metric type.

    details     — optional dict for any metric-specific extra data.
                  e.g. calibration bins, confusion matrix, etc.
    """

    name: str
    value: float
    breakdown: dict[str, float] = field(default_factory=dict)
    n_evaluated: int = 0
    n_abstained: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MetricScore(name={self.name!r}, value={self.value:.4f}, "
            f"n_evaluated={self.n_evaluated}, n_abstained={self.n_abstained})"
        )


# ── BaseMetric ────────────────────────────────────────────────────────────────


class BaseMetric(ABC):
    """
    Abstract base class for all Krisis evaluation metrics.

    Usage:
        class MyMetric(BaseMetric):
            name = "My Metric"

            def compute(self, results: list[EvaluationResult]) -> MetricScore:
                ...

        metric = MyMetric()
        score = metric(results)   # calls compute() via __call__
    """

    #: Human-readable metric name. Must be set by subclasses.
    name: str = "BaseMetric"

    @abstractmethod
    def compute(self, results: list[EvaluationResult]) -> MetricScore:
        """
        Compute the metric across all evaluation results.

        Args:
            results: list of EvaluationResult objects from the benchmark run

        Returns:
            MetricScore with the computed value and optional breakdown
        """
        ...

    def __call__(self, results: list[EvaluationResult]) -> MetricScore:
        """Allows metric instances to be called directly: metric(results)"""
        if not results:
            raise ValueError(
                f"{self.__class__.__name__}.compute() received an empty "
                "results list. Ensure the benchmark ran successfully and "
                "produced at least one EvaluationResult."
            )
        return self.compute(results)

    # ── Shared utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _separate_abstentions(
        results: list[EvaluationResult],
    ) -> tuple[list[EvaluationResult], list[EvaluationResult]]:
        """
        Split results into answered and abstained lists.

        Returns:
            (answered, abstained) tuple of EvaluationResult lists
        """
        answered = [r for r in results if not r.abstained]
        abstained = [r for r in results if r.abstained]
        return answered, abstained

    @staticmethod
    def _group_by_stage(
        results: list[EvaluationResult],
    ) -> dict[int, list[EvaluationResult]]:
        """
        Group results by CKD stage from metadata.
        Used by metrics that report per-stage breakdowns.

        Records without 'ckd_stage' in metadata are grouped under key -1.
        """
        groups: dict[int, list[EvaluationResult]] = {}
        for r in results:
            stage = r.metadata.get("ckd_stage", -1)
            groups.setdefault(int(stage), []).append(r)
        return groups

    @staticmethod
    def _validate_confidence(results: list[EvaluationResult]) -> bool:
        """
        Return True if all answered results have confidence scores.
        Used by calibration metrics to check data availability.
        """
        answered = [r for r in results if not r.abstained]
        return all(r.confidence is not None for r in answered)
