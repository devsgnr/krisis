# Results

Benchmark runs return a Pydantic `BenchmarkResult`. It contains row-level
evaluation results, metric scores, suite metadata, backend name, and execution
metadata.

## Execution Extras

Operational fields live in `BenchmarkResult.extras` and are included in text and
JSON reports:

| Field | Meaning |
|---|---|
| `batch_size` | Configured batch size |
| `max_concurrency` | Configured maximum parallel backend batches |
| `n_input_records` | Total records evaluated |
| `n_api_batches` | Number of planned API batches |
| `elapsed_seconds` | Total wall-clock runtime |
| `records_per_second` | Throughput |
| `input_tokens` | Total input tokens when available |
| `output_tokens` | Total output tokens when available |
| `token_total` | Total token usage when available |
| `prompt_capture` | Where full prompt text is stored in full JSON |
| `prompt_data_policy` | Whether patient data are included or redacted from captured prompts |
| `prompt_modes` | Prompt invocation modes observed, such as `single` or `batch` |
| `n_prompts_captured` | Number of result rows with prompt text |
| `prompt_templates_count` | Number of unique redacted prompt templates used |
| `prompt_templates` | Deduplicated redacted prompt templates used in the run |

Example:

```python
result = Benchmark(suite, backend, batch_size=8, max_concurrency=2).run()

print(result.extras["elapsed_seconds"])
print(result.extras["token_total"])
print(result.extras["prompt_templates"][0]["prompt"])
```

Full JSON includes row-level `prompt`, `prompt_mode`, and `raw_response` fields
so prompt design and provider response quality can be audited together. `prompt`
redacts patient data and preserves the reusable instructions/output format.
Aggregate JSON also includes deduplicated templates in `extras.prompt_templates`.

## Result Container

::: krisis.results.result.BenchmarkResult

## Report Formatters

::: krisis.results.report
