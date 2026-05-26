"""Shared helpers for CKD example scripts."""

from __future__ import annotations

import argparse
import os

from krisis.backends.api import DEFAULT_API_MODEL, APIBackend
from krisis.backends.base import BaseBackend
from krisis.benchmark import Benchmark
from krisis.data.base import FeatureSet, SuiteConfig, Task
from krisis.data.ckd.suite import CKDSuite
from krisis.results.report import (
    format_json_report,
    format_metrics_json_report,
    format_report,
)


def _build_api_backend(
    model: str | None,
    api_key: str | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = "low",
    max_retries: int = 2,
    retry_base_seconds: float = 0.5,
    retry_max_seconds: float = 8.0,
) -> BaseBackend:
    kwargs: dict[str, object] = {}
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    return APIBackend(
        model=model or DEFAULT_API_MODEL,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        **kwargs,
    )


def run_ckd_benchmark(
    *,
    task: Task,
    model: str | None,
    api_key: str | None,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    limit: int | None,
    n_synthetic: int,
    features: FeatureSet,
    data_path: str,
    batch_size: int,
    max_concurrency: int,
    as_json: bool = False,
    metrics_only: bool = False,
    include_results: bool = False,
) -> None:
    backend = _build_api_backend(
        model,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    suite = CKDSuite(
        config=SuiteConfig(
            features=features,
            task=task,
            n_synthetic=n_synthetic,
        ),
        data_path=data_path,
    )
    records = suite.load()
    if limit is not None:
        records = records[:limit]

    result = Benchmark(
        suite,
        backend,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    ).run(records=records)
    if metrics_only:
        print(format_metrics_json_report(result))
    elif as_json:
        print(format_json_report(result, include_results=include_results))
    else:
        print(format_report(result))


def build_ckd_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenRouter model id. Defaults to openai/gpt-5.5. Examples: "
            "anthropic/claude-4.7-opus, x-ai/grok-4.3, google/gemini-3.5-flash."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=("OpenRouter API key. Defaults to OPENROUTER_API_KEY when omitted."),
    )
    parser.add_argument(
        "--data-path",
        default="datasets/ckd/ckd_full.csv",
        help="Path to a local UCI CKD CSV file.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count for transient provider errors such as 429/5xx/timeouts.",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=0.5,
        help="Initial retry backoff delay in seconds.",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=8.0,
        help="Maximum retry backoff delay in seconds.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Optional per-row output token cap. Increase this when a model "
            "returns empty or truncated JSON."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        choices=["omit", "none", "minimal", "low", "medium", "high", "xhigh"],
        default="low",
        help=(
            "Reasoning effort for supported API models. Defaults to low. Use "
            "omit to leave it unset."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for smoke tests. Omit to run all evaluation rows.",
    )
    parser.add_argument("--n-synthetic", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of patient rows per provider API call. Use 1 for strict per-row calls.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of batch API calls to run at once. Increase carefully for rate limits.",
    )
    parser.add_argument(
        "--features",
        choices=[feature_set.value for feature_set in FeatureSet],
        default=FeatureSet.FULL.value,
        help="Feature set to expose to the model. Defaults to full.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print aggregate benchmark results as JSON instead of text. "
            "Includes redacted prompt templates in extras.prompt_templates."
        ),
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print only the metrics block as JSON.",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include row-level evaluation results in JSON output.",
    )
    return parser


def run_ckd_task(task: Task, description: str) -> None:
    parser = build_ckd_parser(description)
    args = parser.parse_args()

    run_ckd_benchmark(
        task=task,
        model=args.model,
        api_key=args.api_key or os.getenv("API_KEY"),
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=None
        if args.reasoning_effort == "omit"
        else args.reasoning_effort,
        limit=args.limit,
        n_synthetic=args.n_synthetic,
        features=FeatureSet(args.features),
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        as_json=args.json,
        metrics_only=args.metrics_only,
        include_results=args.include_results,
    )
