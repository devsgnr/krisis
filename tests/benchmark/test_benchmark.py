"""Integration tests for the benchmark harness (no live API calls)."""

from __future__ import annotations

import json
import math
import time

from krisis.backends.base import BackendResponse, BaseBackend
from krisis.benchmark import Benchmark
from krisis.data.base import BaseDataSuite, PatientRecord, SuiteConfig, Task
from krisis.metrics.abstention import (
    AbstentionRate,
    AnswerRate,
    DeferralAlignment,
    SelectiveAccuracy,
)
from krisis.results.report import (
    format_json_report,
    format_metrics_json_report,
    format_report,
)


class _FakeSuite(BaseDataSuite):
    def __init__(
        self, records: list[PatientRecord], task: Task = Task.DETECTION
    ) -> None:
        super().__init__(SuiteConfig(task=task, seed=0, n_synthetic=0, test_size=0.2))
        self._records = records

    def load(self) -> list[PatientRecord]:
        return list(self._records)

    def describe(self) -> dict[str, str]:
        return {"domain": "fake", "n_records": str(len(self._records))}


class _EchoBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "echo"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        return BackendResponse(
            prediction=record.label,
            abstained=False,
            confidence=0.75,
            raw_response='{"abstained": false, "confidence": 0.75, "prediction": '
            + str(int(record.label))
            + "}",
            prompt=[{"role": "system", "content": "fake single prompt"}],
            prompt_mode="single",
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
        )


class _PeriodicAbstainBackend(BaseBackend):
    def __init__(self) -> None:
        self._n = 0

    @property
    def name(self) -> str:
        return "periodic"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        self._n += 1
        if self._n % 2 == 0:
            return BackendResponse(
                prediction=None,
                abstained=True,
                confidence=None,
                raw_response="abstain",
            )
        return BackendResponse(
            prediction=record.label,
            abstained=False,
            confidence=0.5,
            raw_response=(
                '{"abstained": false, "confidence": 0.5, "prediction": '
                f"{int(record.label)}"
                "}"
            ),
        )


class _SlowBatchBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "slow-batch"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        return BackendResponse(
            prediction=record.label,
            abstained=False,
            confidence=0.9,
            raw_response="single",
        )

    def evaluate_batch(
        self,
        records: list[PatientRecord],
        task: Task,
    ) -> list[BackendResponse]:
        time.sleep(float(records[0].features.get("delay", 0.0)))
        return [
            BackendResponse(
                prediction=record.label,
                abstained=False,
                confidence=0.9,
                raw_response="batch",
            )
            for record in records
        ]


class _MalformedBatchBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "malformed-batch"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        return BackendResponse(
            prediction=record.label,
            abstained=False,
            confidence=0.8,
            raw_response="single-fallback",
        )

    def evaluate_batch(
        self,
        records: list[PatientRecord],
        task: Task,
    ) -> list[BackendResponse]:
        if len(records) > 1:
            raise ValueError("bad batched JSON")
        return [self.evaluate(records[0], task)]


def test_benchmark_echo_detection() -> None:
    records = [
        PatientRecord(features={"a": 1}, label=0, metadata={"ckd_stage": 3}),
        PatientRecord(features={"a": 2}, label=1, metadata={"ckd_stage": 2}),
    ]
    suite = _FakeSuite(records, task=Task.DETECTION)
    run = Benchmark(suite, _EchoBackend(), metrics=[SelectiveAccuracy()]).run()

    assert len(run.evaluation_results) == 2
    assert run.metric_scores["Selective Accuracy (answered only)"].value == 1.0


def test_benchmark_concurrent_batches_preserve_row_order() -> None:
    records = [
        PatientRecord(features={"delay": 0.05}, label=0),
        PatientRecord(features={}, label=1),
        PatientRecord(features={"delay": 0.0}, label=1),
        PatientRecord(features={}, label=0),
    ]
    suite = _FakeSuite(records, task=Task.DETECTION)

    run = Benchmark(
        suite,
        _SlowBatchBackend(),
        metrics=[SelectiveAccuracy()],
        batch_size=2,
        max_concurrency=2,
    ).run()

    assert [r.ground_truth for r in run.evaluation_results] == [0, 1, 1, 0]
    assert [r.prediction for r in run.evaluation_results] == [0, 1, 1, 0]
    assert run.extras["batch_size"] == 2
    assert run.extras["max_concurrency"] == 2


def test_benchmark_falls_back_when_batch_response_is_malformed() -> None:
    records = [
        PatientRecord(features={"x": 1}, label=0),
        PatientRecord(features={"x": 2}, label=1),
        PatientRecord(features={"x": 3}, label=0),
    ]
    suite = _FakeSuite(records, task=Task.DETECTION)

    run = Benchmark(
        suite,
        _MalformedBatchBackend(),
        metrics=[SelectiveAccuracy()],
        batch_size=3,
    ).run()

    assert [r.prediction for r in run.evaluation_results] == [0, 1, 0]
    assert run.metric_scores["Selective Accuracy (answered only)"].value == 1.0


def test_abstention_rate_and_selective_accuracy() -> None:
    records = [
        PatientRecord(features={"x": 1}, label=0, metadata={"ckd_stage": 3}),
        PatientRecord(features={"x": 2}, label=1, metadata={"ckd_stage": 3}),
    ]
    suite = _FakeSuite(records)
    run = Benchmark(
        suite,
        _PeriodicAbstainBackend(),
        metrics=[AnswerRate(), AbstentionRate(), SelectiveAccuracy()],
    ).run()

    answer_rate = run.metric_scores["Answer Rate (Coverage)"]
    assert answer_rate.value == 0.5
    ar = run.metric_scores["Abstention Rate"]
    assert ar.value == 0.5
    sa = run.metric_scores["Selective Accuracy (answered only)"]
    assert sa.n_evaluated == 1
    assert sa.value == 1.0


def test_deferral_alignment_with_metadata() -> None:
    records = [
        PatientRecord(
            features={"x": 1},
            label=0,
            metadata={"should_abstain": True, "ckd_stage": 4},
        ),
        PatientRecord(
            features={"x": 2},
            label=1,
            metadata={"should_abstain": False, "ckd_stage": 2},
        ),
    ]
    suite = _FakeSuite(records)
    run = Benchmark(suite, _EchoBackend(), metrics=[DeferralAlignment()]).run()
    score = run.metric_scores["Deferral Alignment"]
    assert score.n_evaluated == 2
    assert score.value == 0.5


def test_format_report_contains_metrics() -> None:
    records = [PatientRecord(features={"x": 1}, label=0)]
    suite = _FakeSuite(records)
    run = Benchmark(suite, _EchoBackend(), metrics=[AbstentionRate()]).run()
    text = format_report(run)
    assert "Abstention Rate" in text
    assert "echo" in text


def test_format_json_report_contains_metrics_and_results() -> None:
    records = [PatientRecord(features={"x": 1}, label=0)]
    suite = _FakeSuite(records)
    run = Benchmark(suite, _EchoBackend(), metrics=[AbstentionRate()]).run()

    data = json.loads(format_json_report(run))

    assert data["backend_name"] == "echo"
    assert data["suite"]["domain"] == "fake"
    assert data["metrics"]["Abstention Rate"]["value"] == 0.0
    assert data["evaluation_results"][0]["ground_truth"] == 0
    assert data["evaluation_results"][0]["prompt"] == [
        {"role": "system", "content": "fake single prompt"}
    ]
    assert data["evaluation_results"][0]["prompt_mode"] == "single"


def test_benchmark_result_json_replaces_nan_with_null() -> None:
    records = [PatientRecord(features={"x": 1}, label=0)]
    suite = _FakeSuite(records)
    run = Benchmark(suite, _EchoBackend(), metrics=[DeferralAlignment()]).run()

    data = json.loads(run.to_json(include_results=False))

    assert data["metrics"]["Deferral Alignment"]["value"] is None
    assert "evaluation_results" not in data


def test_format_metrics_json_report_contains_only_metrics() -> None:
    records = [PatientRecord(features={"x": 1}, label=0)]
    suite = _FakeSuite(records)
    run = Benchmark(suite, _EchoBackend(), metrics=[AbstentionRate()]).run()

    data = json.loads(format_metrics_json_report(run))

    assert set(data) == {"metrics", "execution"}
    assert set(data["metrics"]) == {"Abstention Rate"}
    assert data["metrics"]["Abstention Rate"]["value"] == 0.0
    assert data["execution"]["batch_size"] == 8
    assert data["execution"]["max_concurrency"] == 1
    assert data["execution"]["n_input_records"] == 1
    assert data["execution"]["n_api_batches"] == 1
    assert data["execution"]["elapsed_seconds"] >= 0.0
    assert data["execution"]["input_tokens"] == 10.0
    assert data["execution"]["output_tokens"] == 2.0
    assert data["execution"]["token_total"] == 12.0
    assert data["execution"]["prompt_capture"] == "evaluation_results.prompt"
    assert data["execution"]["prompt_data_policy"] == "patient_data_redacted"
    assert data["execution"]["prompt_modes"] == ["single"]
    assert data["execution"]["n_prompts_captured"] == 1
    assert data["execution"]["prompt_templates_count"] == 1
    assert data["execution"]["prompt_templates"] == [
        {
            "prompt_mode": "single",
            "prompt": [{"role": "system", "content": "fake single prompt"}],
        }
    ]


def test_deferral_alignment_skipped_without_labels() -> None:
    records = [PatientRecord(features={"x": 1}, label=0)]
    suite = _FakeSuite(records)
    run = Benchmark(suite, _EchoBackend(), metrics=[DeferralAlignment()]).run()
    val = run.metric_scores["Deferral Alignment"].value
    assert isinstance(val, float) and math.isnan(val)
