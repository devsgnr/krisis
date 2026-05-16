"""
krisis/backends/batching.py

Shared prompt and parsing helpers for batched provider calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

from krisis.backends.base import BackendResponse
from krisis.backends.usage import TokenUsage
from krisis.data.base import PatientRecord, Task
from krisis.prompts.base import build_messages
from krisis.tasks.base import parse_model_response

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def case_id(index: int) -> str:
    """Stable per-batch identifier returned by the model."""
    return f"case_{index}"


def prediction_schema_for_task(task: Task) -> dict[str, Any]:
    """JSON schema fragment for task-specific predictions."""
    if task == Task.PROGRESSION:
        return {
            "type": ["string", "null"],
            "enum": ["stable", "worsening", "improving", None],
        }
    return {"type": ["integer", "null"]}


def batch_response_schema(task: Task) -> dict[str, Any]:
    """OpenAI-compatible strict JSON schema for batched outputs."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "abstained": {"type": "boolean"},
                        "confidence": {"type": ["number", "null"]},
                        "prediction": prediction_schema_for_task(task),
                    },
                    "required": ["id", "abstained", "confidence", "prediction"],
                },
            }
        },
        "required": ["results"],
    }


def build_batch_messages(
    records: list[PatientRecord],
    task: Task,
) -> list[dict[str, str]]:
    """Build one prompt that evaluates several records independently."""
    if not records:
        raise ValueError("build_batch_messages() received no records.")

    first_messages = build_messages(records[0], task)
    system = "\n\n".join(
        message["content"] for message in first_messages if message["role"] == "system"
    )
    system = (
        f"{system}\n\n"
        "Batch mode rules:\n"
        "- Follow this batched output shape instead of any single-case output "
        "shape above.\n"
        "- Evaluate each case independently.\n"
        "- Do not use information from one case to answer another case.\n"
        "- Return exactly one result object for every case id.\n"
        "- Preserve each id exactly as provided.\n"
        "- Return a single JSON object only, with this shape:\n"
        '{"results":[{"id":"case_0","abstained":false,'
        '"confidence":0.82,"prediction":0}]}'
    )

    case_blocks: list[str] = []
    for i, record in enumerate(records):
        messages = build_messages(record, task)
        user_text = "\n\n".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        case_blocks.append(f"Case id: {case_id(i)}\n{user_text}")

    user = (
        "Evaluate the following cases as an independent batch.\n\n"
        + "\n\n---\n\n".join(case_blocks)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_json_blob(raw: str) -> str | None:
    raw = raw.strip()
    match = _JSON_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return None


def parse_batch_response(
    raw: str,
    task: Task,
    n_expected: int,
) -> list[BackendResponse]:
    """Parse a batched model response back into ordered BackendResponse rows."""
    blob = _extract_json_blob(raw)
    if blob is None:
        preview = raw[:300] if raw else "<empty response>"
        raise ValueError(
            "Batched model response did not contain a JSON object. "
            f"Response preview: {preview}"
        )

    data = json.loads(blob)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ValueError("Batched model response must contain a results array.")

    expected_ids = [case_id(i) for i in range(n_expected)]
    by_id: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict) or "id" not in item:
            continue
        by_id[str(item["id"])] = item

    missing = [expected_id for expected_id in expected_ids if expected_id not in by_id]
    if missing:
        raise ValueError(f"Batched model response omitted result ids: {missing}")

    parsed_rows: list[BackendResponse] = []
    for expected_id in expected_ids:
        item = dict(by_id[expected_id])
        item.pop("id", None)
        raw_item = json.dumps(item, allow_nan=False)
        parsed = parse_model_response(raw_item, task)
        parsed_rows.append(
            BackendResponse(
                prediction=parsed.prediction,
                abstained=parsed.abstained,
                confidence=parsed.confidence,
                raw_response=raw_item,
            )
        )
    return parsed_rows


def distribute_usage_over_batch(
    responses: list[BackendResponse],
    usage: TokenUsage,
) -> list[BackendResponse]:
    """Attach evenly distributed batch token usage to response rows."""
    if not responses:
        return responses

    n = len(responses)
    input_per_row = usage.input_tokens / n if usage.input_tokens is not None else None
    output_per_row = (
        usage.output_tokens / n if usage.output_tokens is not None else None
    )
    total_per_row = usage.total_tokens / n if usage.total_tokens is not None else None

    for response in responses:
        response.input_tokens = input_per_row
        response.output_tokens = output_per_row
        response.total_tokens = total_per_row
    return responses
