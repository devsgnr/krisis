"""
krisis/results/result.py

Container for a finished benchmark run.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, Field

from krisis.metrics.base import EvaluationResult, MetricScore


def _json_safe(value: Any) -> Any:
    """Recursively convert benchmark artefacts into strict JSON values."""
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


class BenchmarkResult(BaseModel):
    """All artefacts produced by ``Benchmark.run()``."""

    evaluation_results: list[EvaluationResult]
    metric_scores: dict[str, MetricScore]
    suite_description: dict[str, Any]
    backend_name: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self, *, include_results: bool = True) -> dict[str, Any]:
        """
        Return a JSON-safe dictionary representation of the run.

        ``include_results=False`` keeps only suite metadata and aggregate
        scores, which is useful for compact logs.
        """
        data: dict[str, Any] = {
            "backend_name": self.backend_name,
            "suite": self.suite_description,
            "metrics": {
                name: _json_safe(score)
                for name, score in sorted(self.metric_scores.items())
            },
            "n_evaluation_results": len(self.evaluation_results),
            "extras": self.extras,
        }
        if include_results:
            data["evaluation_results"] = self.evaluation_results
        return _json_safe(data)

    def metrics_to_dict(self) -> dict[str, Any]:
        """Return only the JSON-safe aggregate metric scores."""
        return _json_safe(
            {
                "metrics": {
                    name: score for name, score in sorted(self.metric_scores.items())
                },
                "execution": self.extras,
            }
        )

    def to_json(
        self,
        *,
        include_results: bool = True,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        """Return a strict JSON string for the benchmark run."""
        return json.dumps(
            self.to_dict(include_results=include_results),
            allow_nan=False,
            indent=indent,
            sort_keys=sort_keys,
        )

    def metrics_to_json(
        self,
        *,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        """Return strict JSON containing only aggregate metric scores."""
        return json.dumps(
            self.metrics_to_dict(),
            allow_nan=False,
            indent=indent,
            sort_keys=sort_keys,
        )
