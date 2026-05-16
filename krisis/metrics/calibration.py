"""
krisis/metrics/calibration.py

Calibration metrics using the optional ``confidence`` field on
:class:`~krisis.metrics.base.EvaluationResult`.

Abstentions are always excluded — there is no well-defined class probability
mass to compare against the label.
"""

from __future__ import annotations

import math
from typing import Any

from krisis.metrics.base import BaseMetric, EvaluationResult, MetricScore
from krisis.tasks.base import labels_match


def _usable_for_calibration(results: list[EvaluationResult]) -> list[EvaluationResult]:
    return [
        r
        for r in results
        if not r.abstained and r.confidence is not None and r.prediction is not None
    ]


def _bin_index(confidence: float, n_bins: int) -> int:
    c = max(0.0, min(1.0, float(confidence)))
    idx = int(c * n_bins)
    return min(idx, n_bins - 1)


class ExpectedCalibrationError(BaseMetric):
    """
    Expected Calibration Error (ECE) via equal-width bins on ``[0, 1]``.

    Each sample contributes its stated ``confidence`` (probability the model
    assigns to being correct, or the top-class score). Within bin *k*,
    compare the mean confidence to the empirical accuracy; ECE is the
    bin-size-weighted absolute gap.

    Lower is better. When no usable rows exist (all abstained or missing
    confidence), returns NaN.
    """

    name = "Expected Calibration Error"

    def __init__(self, n_bins: int = 15) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        self.n_bins = n_bins

    def compute(self, results: list[EvaluationResult]) -> MetricScore:
        n_abs = sum(1 for r in results if r.abstained)
        usable = _usable_for_calibration(results)
        if not usable:
            return MetricScore(
                name=self.name,
                value=float("nan"),
                breakdown={},
                n_evaluated=0,
                n_abstained=n_abs,
                details={"reason": "no_rows_with_confidence"},
            )

        bin_correct: list[int] = [0 for _ in range(self.n_bins)]
        bin_total: list[int] = [0 for _ in range(self.n_bins)]
        bin_conf_sum: list[float] = [0.0 for _ in range(self.n_bins)]

        for r in usable:
            c = max(0.0, min(1.0, float(r.confidence)))
            b = _bin_index(c, self.n_bins)
            bin_total[b] += 1
            bin_conf_sum[b] += c
            if labels_match(r.prediction, r.ground_truth):
                bin_correct[b] += 1

        ece = 0.0
        bin_rows: list[dict[str, Any]] = []
        n = len(usable)
        for i in range(self.n_bins):
            count = bin_total[i]
            low = i / self.n_bins
            high = (i + 1) / self.n_bins
            if count == 0:
                bin_rows.append(
                    {
                        "bin": i,
                        "low": low,
                        "high": high,
                        "count": 0,
                        "accuracy": float("nan"),
                        "mean_confidence": float("nan"),
                        "gap": float("nan"),
                    }
                )
                continue
            acc = bin_correct[i] / count
            mean_conf = bin_conf_sum[i] / count
            gap = abs(acc - mean_conf)
            ece += (count / n) * gap
            bin_rows.append(
                {
                    "bin": i,
                    "low": low,
                    "high": high,
                    "count": count,
                    "accuracy": acc,
                    "mean_confidence": mean_conf,
                    "gap": gap,
                }
            )

        breakdown: dict[str, float] = {}
        for row in bin_rows:
            if row["count"] == 0:
                continue
            gap = row["gap"]
            if isinstance(gap, float) and not math.isnan(gap):
                breakdown[f"bin_{row['bin']}_gap"] = float(gap)

        return MetricScore(
            name=self.name,
            value=float(ece),
            breakdown=breakdown,
            n_evaluated=n,
            n_abstained=n_abs,
            details={
                "n_bins": self.n_bins,
                "bins": bin_rows,
            },
        )


def _as_binary_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        return None
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


class BrierScore(BaseMetric):
    """
    Brier score for **binary** {0, 1} labels only.

    Uses ``confidence`` as the model's estimated probability *for the positive
    class* (label ``1``): if the prediction is ``1``, ``p = confidence``; if
    the prediction is ``0``, ``p = 1 - confidence``. If any row is outside the
    binary setting, returns NaN.

    Lower is better. Abstentions are excluded.
    """

    name = "Brier Score"

    def compute(self, results: list[EvaluationResult]) -> MetricScore:
        n_abs = sum(1 for r in results if r.abstained)
        usable = _usable_for_calibration(results)
        if not usable:
            return MetricScore(
                name=self.name,
                value=float("nan"),
                breakdown={},
                n_evaluated=0,
                n_abstained=n_abs,
                details={"reason": "no_rows_with_confidence"},
            )

        pairs: list[tuple[int, int, float]] = []
        for r in usable:
            y = _as_binary_int(r.ground_truth)
            pred = _as_binary_int(r.prediction)
            if y is None or pred is None:
                return MetricScore(
                    name=self.name,
                    value=float("nan"),
                    breakdown={},
                    n_evaluated=len(usable),
                    n_abstained=n_abs,
                    details={"reason": "non_binary_labels_or_predictions"},
                )
            c = max(0.0, min(1.0, float(r.confidence)))
            pairs.append((y, pred, c))

        terms = [
            ((c if pred == 1 else 1.0 - c) - float(y)) ** 2 for y, pred, c in pairs
        ]

        score = float(sum(terms) / len(terms)) if terms else float("nan")

        return MetricScore(
            name=self.name,
            value=score,
            breakdown={},
            n_evaluated=len(terms),
            n_abstained=n_abs,
            details={},
        )


def default_calibration_metrics() -> list[BaseMetric]:
    """Standard calibration diagnostics for benchmark tables."""
    return [ExpectedCalibrationError(), BrierScore()]
