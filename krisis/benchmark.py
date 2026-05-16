"""
krisis/benchmark.py

End-to-end harness: data suite → backend → ``EvaluationResult`` rows → metrics.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from krisis.backends.base import BackendResponse, BaseBackend
from krisis.data.base import BaseDataSuite, PatientRecord, Task
from krisis.metrics import default_benchmark_metrics
from krisis.metrics.base import BaseMetric, EvaluationResult, MetricScore
from krisis.results.result import BenchmarkResult


class Benchmark:
    """
    Run a full evaluation pass.

    With ``metrics=None``, runs :func:`krisis.metrics.default_benchmark_metrics`
    (overall accuracy, balanced accuracy, ECE, Brier score, selective accuracy,
    abstention rate, and deferral alignment when ``should_abstain`` metadata is present).

    Typical usage::

        from krisis.data.ckd.suite import CKDSuite
        from krisis.backends.openai import OpenAIBackend
        from krisis.benchmark import Benchmark
        from krisis.results.report import format_report

        suite = CKDSuite()
        backend = OpenAIBackend(model="gpt-5.5")
        run = Benchmark(suite, backend).run()
        print(format_report(run))
    """

    def __init__(
        self,
        suite: BaseDataSuite,
        backend: BaseBackend,
        metrics: Sequence[BaseMetric] | None = None,
        batch_size: int = 8,
        max_concurrency: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        self.suite = suite
        self.backend = backend
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.metrics: list[BaseMetric] = (
            list(metrics) if metrics is not None else default_benchmark_metrics()
        )

    def run(
        self,
        records: list[PatientRecord] | None = None,
        *,
        suite_description: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """
        Execute the benchmark.

        Args:
            records: optional pre-built patient rows. When omitted, rows are
                produced via ``suite.load()`` (which may touch disk).
            suite_description: optional override for reporting metadata. When
                omitted, ``suite.describe()`` is used after loading data.
        """
        started_at = time.perf_counter()
        rows = list(records) if records is not None else self.suite.load()
        if not rows:
            raise ValueError(
                "Benchmark.run() received zero patient records. "
                "Check suite configuration, dataset paths, and split sizes."
            )

        if suite_description is not None:
            description = dict(suite_description)
        else:
            # CKDSuite.describe() reflects statistics only after load();
            # when injecting custom ``records``, pass ``suite_description``
            # if you need an accurate suite summary without calling load().
            description = self.suite.describe()

        task: Task = self.suite.config.task
        eval_rows = self._evaluate_rows(rows, task)
        elapsed_seconds = time.perf_counter() - started_at

        scores: dict[str, MetricScore] = {}
        for metric in self.metrics:
            mscore = metric(eval_rows)
            scores[metric.name] = mscore

        return BenchmarkResult(
            evaluation_results=eval_rows,
            metric_scores=scores,
            suite_description=description,
            backend_name=self.backend.name,
            extras=self._execution_extras(rows, eval_rows, elapsed_seconds),
        )

    def _evaluate_rows(
        self,
        rows: list[PatientRecord],
        task: Task,
    ) -> list[EvaluationResult]:
        chunks = [
            (start, rows[start : start + self.batch_size])
            for start in range(0, len(rows), self.batch_size)
        ]
        if self.max_concurrency == 1 or len(chunks) == 1:
            results: list[EvaluationResult | None] = [None] * len(rows)
            for start, batch in chunks:
                _, eval_batch = self._evaluate_batch(start, batch, task)
                results[start : start + len(eval_batch)] = eval_batch
            return [r for r in results if r is not None]

        results = [None] * len(rows)
        max_workers = min(self.max_concurrency, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._evaluate_batch, start, batch, task)
                for start, batch in chunks
            ]
            for future in as_completed(futures):
                start, eval_batch = future.result()
                if not eval_batch:
                    continue
                results[start : start + len(eval_batch)] = eval_batch
        return [r for r in results if r is not None]

    def _evaluate_batch(
        self,
        start: int,
        batch: list[PatientRecord],
        task: Task,
    ) -> tuple[int, list[EvaluationResult]]:
        responses = self._safe_backend_batch(batch, task)
        if len(responses) != len(batch):
            raise ValueError(
                f"{self.backend.name}.evaluate_batch() returned "
                f"{len(responses)} responses for {len(batch)} records."
            )
        eval_batch: list[EvaluationResult] = []
        for record, resp in zip(batch, responses, strict=True):
            eval_batch.append(
                EvaluationResult(
                    prediction=resp.prediction,
                    ground_truth=record.label,
                    abstained=resp.abstained,
                    confidence=resp.confidence,
                    raw_response=resp.raw_response,
                    metadata=dict(record.metadata),
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    total_tokens=resp.total_tokens,
                )
            )
        return start, eval_batch

    @staticmethod
    def _sum_optional(values: list[float | None]) -> float | None:
        usable = [value for value in values if value is not None]
        if not usable:
            return None
        return float(sum(usable))

    def _execution_extras(
        self,
        rows: list[PatientRecord],
        eval_rows: list[EvaluationResult],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        input_tokens = self._sum_optional([row.input_tokens for row in eval_rows])
        output_tokens = self._sum_optional([row.output_tokens for row in eval_rows])
        total_tokens = self._sum_optional([row.total_tokens for row in eval_rows])

        return {
            "batch_size": self.batch_size,
            "max_concurrency": self.max_concurrency,
            "n_input_records": len(rows),
            "n_api_batches": len(range(0, len(rows), self.batch_size)),
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": len(rows) / elapsed_seconds
            if elapsed_seconds > 0
            else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_total": total_tokens,
        }

    def _safe_backend_batch(
        self,
        batch: list[PatientRecord],
        task: Task,
    ) -> list[BackendResponse]:
        """
        Evaluate a batch, recursively shrinking it if the provider returns
        malformed batched JSON.

        Some frontier models are less reliable with array-shaped outputs. This
        keeps the benchmark running while still using large batches whenever
        the provider follows the requested format.
        """
        try:
            return self.backend.evaluate_batch(batch, task)
        except ValueError:
            if len(batch) == 1:
                return [self.backend.evaluate(batch[0], task)]
            midpoint = len(batch) // 2
            return self._safe_backend_batch(
                batch[:midpoint], task
            ) + self._safe_backend_batch(batch[midpoint:], task)
