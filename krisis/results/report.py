"""
krisis/results/report.py

Human-readable summaries of ``BenchmarkResult``.
"""

from __future__ import annotations

import math

from krisis.results.result import BenchmarkResult


def format_report(run: BenchmarkResult) -> str:
    """Return a compact multi-line text summary suitable for logs or papers."""
    lines: list[str] = [
        "Krisis benchmark",
        "================",
        f"Backend: {run.backend_name or '(unknown)'}",
        "",
        "Suite",
        "-----",
    ]
    for key, val in sorted(run.suite_description.items()):
        lines.append(f"  {key}: {val}")

    lines.extend(["", "Metrics", "-------"])
    for name, score in sorted(run.metric_scores.items()):
        val = score.value
        val_s = "nan" if isinstance(val, float) and math.isnan(val) else f"{val:.4f}"
        lines.append(
            f"  {name}: {val_s} "
            f"(n_evaluated={score.n_evaluated}, n_abstained={score.n_abstained})"
        )

    if run.extras:
        lines.extend(["", "Execution", "---------"])
        for key, val in sorted(run.extras.items()):
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")

    lines.extend(
        [
            "",
            f"Total patient rows: {len(run.evaluation_results)}",
        ]
    )
    return "\n".join(lines)


def format_json_report(run: BenchmarkResult, *, include_results: bool = True) -> str:
    """Return a strict JSON summary of a benchmark run."""
    return run.to_json(include_results=include_results)


def format_metrics_json_report(run: BenchmarkResult) -> str:
    """Return strict JSON containing only aggregate metric scores."""
    return run.metrics_to_json()
