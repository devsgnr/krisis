"""Tests for krisis.metrics.accuracy."""

from __future__ import annotations

import math
import warnings

from krisis.metrics.accuracy import Accuracy, BalancedAccuracy
from krisis.metrics.base import EvaluationResult


def test_accuracy_counts_abstention_as_wrong() -> None:
    rows = [
        EvaluationResult(0, 0, abstained=False),
        EvaluationResult(0, 1, abstained=False),
        EvaluationResult(None, 0, abstained=True),
    ]
    m = Accuracy()
    score = m(rows)
    assert score.value == 1 / 3
    assert score.n_abstained == 1
    assert score.details["n_correct"] == 1


def test_accuracy_neutral_excludes_abstentions() -> None:
    rows = [
        EvaluationResult(0, 0, abstained=False),
        EvaluationResult(None, 0, abstained=True),
    ]
    m = Accuracy(treat_abstention_as_neutral=True)
    score = m(rows)
    assert score.value == 1.0
    assert score.n_evaluated == 1


def test_balanced_accuracy_with_abstain_encoded() -> None:
    # Class 0 twice, class 1 once; one abstention on class 1 row
    rows = [
        EvaluationResult(0, 0, abstained=False),
        EvaluationResult(0, 0, abstained=False),
        EvaluationResult(0, 1, abstained=True),
    ]
    m = BalancedAccuracy()
    score = m(rows)
    assert not math.isnan(score.value)
    assert score.n_abstained == 1


def test_balanced_accuracy_supports_progression_string_labels() -> None:
    rows = [
        EvaluationResult("stable", "stable", abstained=False),
        EvaluationResult("worsening", "worsening", abstained=False),
        EvaluationResult("stable", "improving", abstained=False),
    ]
    m = BalancedAccuracy()
    score = m(rows)
    assert not math.isnan(score.value)
    assert score.n_evaluated == 3


def test_balanced_accuracy_treats_predictions_outside_true_pool_as_wrong() -> None:
    rows = [
        EvaluationResult("improving", "stable", abstained=False),
        EvaluationResult("worsening", "worsening", abstained=False),
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = BalancedAccuracy()(rows)

    assert not math.isnan(score.value)
    assert not any(
        "y_pred contains classes not in y_true" in str(w.message) for w in caught
    )
