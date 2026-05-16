"""
krisis/tasks/base.py

Structured-output parsing for model responses.

Backends ask models for JSON; this module turns raw text into typed fields
used to build EvaluationResult rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from krisis.data.base import Task


@dataclass
class ParsedModelOutput:
    """Fields extracted from a model response before scoring."""

    prediction: int | str | None
    abstained: bool
    confidence: float | None = None


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)


def _extract_json_blob(raw: str) -> str | None:
    raw = raw.strip()
    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    # Sometimes models prefix with prose — take first {...} span
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return None


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        c = float(value)
    except (TypeError, ValueError):
        return None
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


def _coerce_prediction_for_task(
    task: Task,
    value: Any,
    abstained: bool,
) -> int | str | None:
    if abstained:
        return None if value in (None, "", "null") else value
    if value is None or value == "null":
        return None
    if task == Task.DETECTION:
        return int(value)
    if task in (Task.STAGING, Task.PROGRESSION):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return str(value).strip().lower()
    return value


def parse_model_response(raw: str, task: Task) -> ParsedModelOutput:
    """
    Parse a model response into prediction / abstention / confidence.

    Expects a JSON object with keys:
        abstained (bool), confidence (float, optional), prediction (any)

    Falls back to light heuristics when JSON parsing fails.
    """
    blob = _extract_json_blob(raw)
    if blob:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            abstained = bool(data.get("abstained", False))
            conf = _coerce_confidence(data.get("confidence"))
            pred = _coerce_prediction_for_task(task, data.get("prediction"), abstained)
            return ParsedModelOutput(
                prediction=pred,
                abstained=abstained,
                confidence=conf,
            )

    lower = raw.lower()
    abstained = any(
        phrase in lower
        for phrase in (
            "abstain",
            "cannot answer",
            "can't answer",
            "decline to",
            "do not have enough",
            "don't have enough",
            "insufficient information",
            "unable to determine",
        )
    )
    return ParsedModelOutput(
        prediction=None if abstained else None,
        abstained=abstained,
        confidence=None,
    )


def labels_match(prediction: int | str | None, ground_truth: int | str) -> bool:
    """Equality check tolerant of int/str mismatches for numeric labels."""
    if prediction is None:
        return False
    if prediction == ground_truth:
        return True
    try:
        return int(prediction) == int(ground_truth)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(prediction).strip().lower() == str(ground_truth).strip().lower()
