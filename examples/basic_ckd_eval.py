"""Run any CKD benchmark task with the OpenAI backend.

Usage:
    OPENAI_API_KEY=... python examples/basic_ckd_eval.py
    OPENAI_API_KEY=... python examples/basic_ckd_eval.py --limit 10

Dedicated entrypoints are also available:
    python examples/ckd_detection.py
    python examples/ckd_progression.py
    python examples/ckd_staging.py
"""

from __future__ import annotations

import os

from _ckd_common import build_ckd_parser, run_ckd_benchmark

from krisis.data.base import FeatureSet, Task


def main() -> None:
    parser = build_ckd_parser("Run a CKD OpenAI benchmark.")
    parser.add_argument(
        "--task", choices=[t.value for t in Task], default=Task.DETECTION.value
    )
    args = parser.parse_args()

    run_ckd_benchmark(
        task=Task(args.task),
        backend_provider=args.backend,
        model=args.model,
        api_key=args.api_key or os.getenv("API_KEY"),
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        max_output_tokens=args.max_output_tokens,
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


if __name__ == "__main__":
    main()
