# Benchmark

`Benchmark` is the execution layer. It receives a suite, a backend, and a metric
bundle, then produces a `BenchmarkResult`.

## Constructor Controls

| Parameter | Default | What it controls |
|---|---:|---|
| `suite` | required | Data suite that produces `PatientRecord` rows |
| `backend` | required | Model backend used for inference |
| `metrics` | `None` | Optional custom metric list. `None` uses the default Krisis metric bundle |
| `batch_size` | `8` | Number of patient records sent to the backend in one provider call |
| `max_concurrency` | `1` | Number of backend batches allowed to run in parallel |

Example:

```python
result = Benchmark(
    suite,
    backend,
    batch_size=8,
    max_concurrency=2,
).run()
```

## Batch Size vs Concurrency

`batch_size` and `max_concurrency` are separate controls.

| Setting | Example | Meaning |
|---|---:|---|
| `batch_size=8` | 8 records | One API call asks the model to evaluate 8 patient rows |
| `max_concurrency=2` | 2 calls | Krisis may run two batch calls at the same time |

With `batch_size=8` and `max_concurrency=2`, up to 16 records can be in flight.
This reduces HTTP overhead and can improve throughput, but provider rate limits
still apply.

!!! tip "Start conservative"
    Use `batch_size=8` and `max_concurrency=1` or `2` for first runs. Increase
    after checking provider rate limits and structured JSON reliability.

!!! tip "When a model returns empty JSON"
    Empty responses usually mean the provider returned no text or the
    output-token cap was too low for the requested batch. In the example CLI,
    raise `--max-output-tokens`; for direct backend use, raise the provider's
    token cap (`max_completion_tokens`, `max_tokens`, or `max_output_tokens`).

## Batched JSON Fallback

Krisis asks backends to evaluate batches via `backend.evaluate_batch(...)`.

If a provider returns malformed batched JSON, `Benchmark` does not immediately
fail the whole run. It recursively splits the batch into smaller chunks. If the
batch is already size 1, it falls back to `backend.evaluate(...)`.

This protects long benchmark runs from one weak batched response while still
using batching whenever the provider follows the requested format.

!!! note "Retries are configured on backends"
    `Benchmark` controls batching and concurrency. Provider failure retries are
    configured on the backend with `max_retries`, `retry_base_seconds`, and
    `retry_max_seconds`.

## Execution Metadata

`Benchmark.run()` stores operational details in `BenchmarkResult.extras`:

| Field | Meaning |
|---|---|
| `batch_size` | Configured batch size |
| `max_concurrency` | Configured concurrency |
| `n_input_records` | Total patient records evaluated |
| `n_api_batches` | Number of planned provider batches |
| `elapsed_seconds` | Wall-clock runtime |
| `records_per_second` | Evaluation throughput |
| `input_tokens` | Total input tokens when provider usage is available |
| `output_tokens` | Total output tokens when provider usage is available |
| `token_total` | Total input + output tokens when provider usage is available |
| `prompt_capture` | Where full prompt text is stored in full JSON |
| `prompt_data_policy` | Whether patient data are included or redacted from captured prompts |
| `prompt_modes` | Prompt invocation modes observed, such as `single` or `batch` |
| `n_prompts_captured` | Number of result rows with prompt text |
| `prompt_templates_count` | Number of unique redacted prompt templates used |
| `prompt_templates` | Deduplicated redacted prompt templates used in the run |

These fields appear in text reports and JSON reports.

Redacted prompt templates are stored in `extras.prompt_templates`. Row-level
`EvaluationResult.prompt` also stores the redacted template used for that row.

::: krisis.benchmark.Benchmark
