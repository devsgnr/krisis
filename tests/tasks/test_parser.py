"""Tests for structured response parsing (krisis.tasks.base)."""

from __future__ import annotations

from krisis.data.base import Task
from krisis.tasks.base import labels_match, parse_model_response


def test_parse_json_detection() -> None:
    raw = '{"abstained": false, "confidence": 0.9, "prediction": 0}'
    out = parse_model_response(raw, Task.DETECTION)
    assert out.abstained is False
    assert out.confidence == 0.9
    assert out.prediction == 0


def test_parse_fenced_json_staging() -> None:
    raw = '```json\n{"abstained": true, "confidence": 0.2, "prediction": null}\n```'
    out = parse_model_response(raw, Task.STAGING)
    assert out.abstained is True
    assert out.prediction is None


def test_labels_match_int_str() -> None:
    assert labels_match("1", 1) is True
    assert labels_match(0, 1) is False


def test_parse_heuristic_abstain() -> None:
    out = parse_model_response("I decline to answer.", Task.DETECTION)
    assert out.abstained is True


def test_parse_json_progression_string_label() -> None:
    raw = '{"abstained": false, "confidence": 0.7, "prediction": "Worsening"}'
    out = parse_model_response(raw, Task.PROGRESSION)
    assert out.abstained is False
    assert out.confidence == 0.7
    assert out.prediction == "worsening"
