"""Public metric surface and default bundles."""

from __future__ import annotations

from krisis.metrics.abstention import (
    AbstentionRate,
    AnswerRate,
    DeferralAlignment,
    SelectiveAccuracy,
    default_abstention_metrics,
)
from krisis.metrics.accuracy import Accuracy, BalancedAccuracy, default_accuracy_metrics
from krisis.metrics.base import BaseMetric, EvaluationResult, MetricScore
from krisis.metrics.calibration import (
    BrierScore,
    ExpectedCalibrationError,
    default_calibration_metrics,
)


def default_benchmark_metrics() -> list[BaseMetric]:
    """
    Full default stack: accuracy, balanced accuracy, calibration (ECE, Brier),
    then abstention and deferral diagnostics (including selective accuracy).
    """
    return [
        *default_accuracy_metrics(),
        *default_calibration_metrics(),
        *default_abstention_metrics(),
    ]


__all__ = [
    "AbstentionRate",
    "AnswerRate",
    "Accuracy",
    "BalancedAccuracy",
    "BaseMetric",
    "BrierScore",
    "DeferralAlignment",
    "EvaluationResult",
    "ExpectedCalibrationError",
    "MetricScore",
    "SelectiveAccuracy",
    "default_abstention_metrics",
    "default_accuracy_metrics",
    "default_benchmark_metrics",
    "default_calibration_metrics",
]
