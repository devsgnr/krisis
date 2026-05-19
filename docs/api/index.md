# API Reference

This section is generated from Krisis docstrings with `mkdocstrings`.

!!! note "Public API focus"
    The API reference emphasizes the objects most users and contributors need:
    shared data objects, suite base classes, backend base classes, metric base
    classes, benchmark execution, and reports.

!!! tip "Examples live in the guides"
    API pages show signatures and docstrings without embedding full source
    code. For narrative examples, use Getting Started and the Framework Guide.

## Main Entry Points

| Area | Objects |
|---|---|
| Benchmark | `Benchmark` |
| Data | `PatientRecord`, `SuiteConfig`, `Task`, `FeatureSet` |
| Suite | `BaseDataSuite`, `BasePreprocessor`, `BaseFeatureEngineer`, `BaseGenerator` |
| Backend | `BaseBackend`, `BackendResponse` |
| Metrics | `EvaluationResult`, `MetricScore`, `BaseMetric` |
| Results | `BenchmarkResult`, report formatters |

Use the left navigation to inspect each module.

## Operational Controls

The API Reference documents the runtime controls that matter for real
benchmarks:

| Control | Where to configure it |
|---|---|
| `batch_size` | `Benchmark(...)` |
| `max_concurrency` | `Benchmark(...)` |
| `max_retries` | provider backend |
| `retry_base_seconds` | provider backend |
| `retry_max_seconds` | provider backend |
| `token_total` | `BenchmarkResult.extras` |
| `elapsed_seconds` | `BenchmarkResult.extras` |
| `records_per_second` | `BenchmarkResult.extras` |
| `prompt` | `EvaluationResult.prompt` |
| `prompt_mode` | `EvaluationResult.prompt_mode` |
| `prompt_data_policy` | `BenchmarkResult.extras` |
| `prompt_templates` | `BenchmarkResult.extras` |

!!! note "Concrete suites live in the framework guide"
    CKD is the first implemented suite in Krisis v0.1, but the API reference
    focuses on reusable base classes. CKD-specific behavior is documented under
    Framework Guide → Suites → CKD.
