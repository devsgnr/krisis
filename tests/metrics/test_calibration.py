"""Tests for calibration metrics."""

from __future__ import annotations

import math

from krisis.metrics.base import EvaluationResult
from krisis.metrics.calibration import BrierScore, ExpectedCalibrationError


def test_ece_perfect_calibration_small_bins() -> None:
    rows = [
        EvaluationResult(1, 1, abstained=False, confidence=0.9),
        EvaluationResult(0, 0, abstained=False, confidence=0.8),
        EvaluationResult(1, 1, abstained=False, confidence=0.85),
        EvaluationResult(0, 0, abstained=False, confidence=0.75),
    ]
    m = ExpectedCalibrationError(n_bins=5)
    score = m(rows)
    assert not math.isnan(score.value)
    assert score.n_evaluated == 4
    assert "bins" in score.details


def test_ece_nan_without_confidence() -> None:
    rows = [EvaluationResult(1, 1, abstained=False, confidence=None)]
    m = ExpectedCalibrationError(n_bins=5)
    score = m(rows)
    assert math.isnan(score.value)
    assert score.details.get("reason") == "no_rows_with_confidence"


def test_brier_binary() -> None:
    # pred=1, conf=1, y=1 -> (1-1)^2=0; pred=0, conf=0.8 -> p_pos=0.2, y=1 -> (0.2-1)^2
    rows = [
        EvaluationResult(1, 1, abstained=False, confidence=1.0),
        EvaluationResult(0, 1, abstained=False, confidence=0.8),
    ]
    m = BrierScore()
    score = m(rows)
    assert score.n_evaluated == 2
    expected = (0.0 + (0.2 - 1.0) ** 2) / 2
    assert abs(score.value - expected) < 1e-9


def test_brier_nan_for_multiclass() -> None:
    rows = [
        EvaluationResult(2, 2, abstained=False, confidence=0.7),
    ]
    m = BrierScore()
    score = m(rows)
    assert math.isnan(score.value)
