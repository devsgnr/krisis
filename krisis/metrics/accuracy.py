"""
krisis/metrics/accuracy.py

Standard accuracy metrics over evaluation rows.

``SelectiveAccuracy`` (in :mod:`krisis.metrics.abstention`) scores only
non-abstained predictions. The metrics here include abstentions in the
denominator by default so “overall task success” stays comparable to
classic supervised benchmarks.
"""

from __future__ import annotations

from sklearn.metrics import balanced_accuracy_score

from krisis.metrics.base import BaseMetric, EvaluationResult, MetricScore
from krisis.tasks.base import labels_match


def _as_int_label(value: int | str | None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _label_key(value: int | str) -> tuple[str, int | str]:
    int_value = _as_int_label(value)
    if int_value is not None:
        return ("int", int_value)
    return ("str", str(value).strip().lower())


def _encode_labels(
    y_true_raw: list[int | str],
    y_pred_raw: list[int | str | None],
) -> tuple[list[int], list[int]]:
    """
    Encode mixed int/string labels into integer ids for sklearn metrics.

    Predictions outside the ground-truth label pool are encoded as ``-1`` so
    the caller can convert them to an incorrect in-pool label. This prevents
    sklearn from warning about classes in ``y_pred`` that are absent from
    ``y_true``.
    """
    labels: dict[tuple[str, int | str], int] = {}

    def encode(value: int | str) -> int:
        key = _label_key(value)
        if key not in labels:
            labels[key] = len(labels)
        return labels[key]

    y_true = [encode(v) for v in y_true_raw]
    y_pred = [
        labels.get(_label_key(v), -1) if v is not None else -1 for v in y_pred_raw
    ]
    return y_true, y_pred


def _mislabel_from_pool(gt: int, pool: list[int]) -> int:
    """
    Return a label from *pool* that is not equal to *gt*.

    Used to encode abstentions (and unparsable predictions) using only
    labels that appear in the evaluated slice so sklearn does not warn about
    unseen classes in ``y_pred``.
    """
    for c in pool:
        if c != gt:
            return c
    return pool[0]


class Accuracy(BaseMetric):
    """
    Fraction of rows where the prediction matches ground truth.

    Abstentions count as **incorrect** unless ``treat_abstention_as_neutral``
    is True, in which case abstentions are excluded from both numerator and
    denominator (then identical to selective accuracy over abstentions
    excluded, but still reported with full ``n_abstained`` on the run).
    """

    name = "Accuracy"

    def __init__(self, treat_abstention_as_neutral: bool = False) -> None:
        self.treat_abstention_as_neutral = treat_abstention_as_neutral

    def compute(self, results: list[EvaluationResult]) -> MetricScore:
        n_abs = sum(1 for r in results if r.abstained)

        if self.treat_abstention_as_neutral:
            scored = [r for r in results if not r.abstained]
            if not scored:
                return MetricScore(
                    name=self.name,
                    value=float("nan"),
                    breakdown={},
                    n_evaluated=0,
                    n_abstained=n_abs,
                    details={"reason": "no_non_abstained_predictions"},
                )
            correct = sum(
                1 for r in scored if labels_match(r.prediction, r.ground_truth)
            )
            denom = len(scored)
        else:
            scored = results
            correct = 0
            for r in results:
                if r.abstained:
                    continue
                if labels_match(r.prediction, r.ground_truth):
                    correct += 1
            denom = len(results)

        value = correct / denom if denom else float("nan")

        breakdown: dict[str, float] = {}
        groups = self._group_by_stage(
            scored if self.treat_abstention_as_neutral else results
        )
        for stage, group in sorted(groups.items(), key=lambda x: x[0]):
            if not group:
                continue
            key = "unknown_stage" if stage < 0 else f"stage_{stage}"
            if self.treat_abstention_as_neutral:
                g_scored = [r for r in group if not r.abstained]
                if not g_scored:
                    continue
                hits = sum(
                    1 for r in g_scored if labels_match(r.prediction, r.ground_truth)
                )
                breakdown[key] = hits / len(g_scored)
            else:
                hits = 0
                for r in group:
                    if r.abstained:
                        continue
                    if labels_match(r.prediction, r.ground_truth):
                        hits += 1
                breakdown[key] = hits / len(group)

        return MetricScore(
            name=self.name,
            value=value,
            breakdown=breakdown,
            n_evaluated=len(scored)
            if self.treat_abstention_as_neutral
            else len(results),
            n_abstained=n_abs,
            details={
                "n_correct": correct,
                "n_incorrect": denom - correct,
                "treat_abstention_as_neutral": self.treat_abstention_as_neutral,
            },
        )


class BalancedAccuracy(BaseMetric):
    """
    sklearn ``balanced_accuracy_score`` over all rows.

    Abstentions (and non-integer predictions) are encoded as an incorrect
    label chosen from the ground-truth class set for this run, so the score
    stays well-defined without inventing unseen label ids.

    String labels are encoded internally, so progression labels such as
    ``stable`` / ``worsening`` / ``improving`` are supported.
    """

    name = "Balanced Accuracy"

    def compute(self, results: list[EvaluationResult]) -> MetricScore:
        n_abs = sum(1 for r in results if r.abstained)
        y_true, y_pred_encoded = _encode_labels(
            [r.ground_truth for r in results],
            [None if r.abstained else r.prediction for r in results],
        )

        if len(y_true) < 2 or len(set(y_true)) < 2:
            return MetricScore(
                name=self.name,
                value=float("nan"),
                breakdown={},
                n_evaluated=len(y_true),
                n_abstained=n_abs,
                details={"reason": "needs_at_least_two_classes_in_ground_truth"},
            )

        pool = sorted(set(y_true))
        y_pred: list[int] = []
        for gt, pred in zip(y_true, y_pred_encoded, strict=True):
            y_pred.append(_mislabel_from_pool(gt, pool) if pred < 0 else pred)

        score = float(balanced_accuracy_score(y_true, y_pred))

        breakdown: dict[str, float] = {}
        groups = self._group_by_stage(results)
        for stage, group in sorted(groups.items(), key=lambda x: x[0]):
            if not group:
                continue
            key = "unknown_stage" if stage < 0 else f"stage_{stage}"
            sub_true, sub_pred_encoded = _encode_labels(
                [r.ground_truth for r in group],
                [None if r.abstained else r.prediction for r in group],
            )
            if len(sub_true) < 2:
                continue
            sub_pool = sorted(set(sub_true))
            if len(sub_pool) < 2:
                continue
            sub_pred: list[int] = []
            for gt, pred in zip(sub_true, sub_pred_encoded, strict=True):
                sub_pred.append(_mislabel_from_pool(gt, sub_pool) if pred < 0 else pred)
            breakdown[key] = float(balanced_accuracy_score(sub_true, sub_pred))

        return MetricScore(
            name=self.name,
            value=score,
            breakdown=breakdown,
            n_evaluated=len(y_true),
            n_abstained=n_abs,
            details={},
        )


def default_accuracy_metrics() -> list[BaseMetric]:
    """Typical accuracy bundle to pair with abstention metrics."""
    return [Accuracy(), BalancedAccuracy()]
