"""Tests for abstention and coverage metrics."""

from __future__ import annotations

from krisis.metrics.abstention import AnswerRate, SelectiveAccuracy
from krisis.metrics.base import EvaluationResult


def test_answer_rate_reports_coverage() -> None:
    rows = [
        EvaluationResult(0, 0, abstained=False, metadata={"ckd_stage": 3}),
        EvaluationResult(None, 1, abstained=True, metadata={"ckd_stage": 3}),
        EvaluationResult(1, 1, abstained=False, metadata={"ckd_stage": 2}),
    ]

    score = AnswerRate()(rows)

    assert score.name == "answer_rate"
    assert score.value == 2 / 3
    assert score.details["n_answered"] == 2
    assert score.n_abstained == 1


def test_selective_accuracy_name_clarifies_answered_only() -> None:
    rows = [
        EvaluationResult(0, 0, abstained=False),
        EvaluationResult(None, 1, abstained=True),
    ]

    score = SelectiveAccuracy()(rows)

    assert score.name == "selective_accuracy"
    assert score.value == 1.0
    assert score.n_evaluated == 1
