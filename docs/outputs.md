# Outputs

Krisis produces human-readable reports and JSON exports from the same benchmark
result object. The goal is to make quick inspection easy while keeping structured
data available for plots, tables, and preprint appendices.

## Output Types

| Output | Best for |
|---|---|
| Text report | Terminal review, logs, quick sanity checks |
| Full JSON | Auditing every row, debugging backend responses, preserving raw outputs |
| Metrics-only JSON | Model comparison tables and plotting |

## Text Report

Use `format_report` for terminal output:

```python
from krisis.results.report import format_report

print(format_report(result))
```

Example shape:

```text
Krisis benchmark
================
Backend: openai:gpt-5.5

Suite
-----
  domain: Chronic Kidney Disease (CKD)
  task: staging
  feature_set: full
  n_total_eval_records: 160

Metrics
-------
  Accuracy: 0.7125
  Abstention Rate: 0.2000
  Answer Rate (Coverage): 0.8000
  Selective Accuracy (answered only): 0.8906
  Deferral Alignment: 1.0000

Execution
---------
  batch_size: 8
  max_concurrency: 2
  elapsed_seconds: 42.18
  records_per_second: 3.79
  token_total: 14400
```

Text reports are intentionally compact. They are good for answering "did the run
work?" and "what are the headline numbers?"

## Full JSON

Use full JSON when you need row-level results:

```python
from krisis.results.report import format_json_report

print(format_json_report(result, include_results=True))
```

Full JSON includes:

- backend name
- suite metadata
- metric scores and details
- execution metadata
- every `EvaluationResult`
- raw model responses
- row-level metadata such as stage, eGFR, and `should_abstain`

!!! warning "Full JSON can be large"
    Full JSON includes every evaluated row. It is useful for audits and
    debugging, but too noisy for most model comparison plots.

## Metrics-Only JSON

Use metrics-only JSON for model comparison:

```python
from krisis.results.report import format_metrics_json_report

print(format_metrics_json_report(result))
```

Metrics-only JSON keeps the important aggregate fields:

```json
{
  "backend_name": "openai:gpt-5.5",
  "suite": {
    "task": "staging",
    "feature_set": "full",
    "n_total_eval_records": 160
  },
  "metrics": {
    "Accuracy": {"value": 0.7125},
    "Abstention Rate": {"value": 0.2},
    "Selective Accuracy (answered only)": {"value": 0.8906}
  },
  "execution": {
    "batch_size": 8,
    "max_concurrency": 2,
    "elapsed_seconds": 42.18,
    "records_per_second": 3.79,
    "token_total": 14400,
    "prompt_capture": "evaluation_results.prompt",
    "prompt_data_policy": "patient_data_redacted",
    "prompt_modes": ["batch"],
    "prompt_templates_count": 1,
    "prompt_templates": [
      {
        "prompt_mode": "batch",
        "prompt": [
          {
            "role": "system",
            "content": "[provider-facing instructions]"
          },
          {
            "role": "user",
            "content": "[BATCH_PATIENT_DATA_REDACTED]"
          }
        ]
      }
    ]
  }
}
```

## Execution Metadata

Reports include an execution block:

```json
{
  "batch_size": 8,
  "max_concurrency": 2,
  "n_input_records": 160,
  "n_api_batches": 20,
  "elapsed_seconds": 42.18,
  "records_per_second": 3.79,
  "input_tokens": 12000,
  "output_tokens": 2400,
  "token_total": 14400,
  "prompt_capture": "evaluation_results.prompt",
  "prompt_data_policy": "patient_data_redacted",
  "prompt_modes": ["batch"],
  "n_prompts_captured": 160,
  "prompt_templates_count": 1,
  "prompt_templates": [
    {
      "prompt_mode": "batch",
      "prompt": [
        {
          "role": "system",
          "content": "[provider-facing instructions]"
        },
        {
          "role": "user",
          "content": "[BATCH_PATIENT_DATA_REDACTED]"
        }
      ]
    }
  ]
}
```

These fields let you compare model behavior and operational cost:

| Field | Meaning |
|---|---|
| `batch_size` | records per provider call |
| `max_concurrency` | parallel provider calls |
| `n_input_records` | total records passed to the benchmark |
| `n_api_batches` | number of provider batches executed |
| `elapsed_seconds` | wall-clock benchmark duration |
| `records_per_second` | throughput |
| `input_tokens` | total input tokens, where provider usage is available |
| `output_tokens` | total output tokens, where provider usage is available |
| `token_total` | combined token usage |
| `prompt_capture` | where full prompt text is stored in full JSON |
| `prompt_data_policy` | whether patient data are included or redacted from captured prompts |
| `prompt_modes` | prompt invocation modes observed, usually `single`, `batch`, or both |
| `n_prompts_captured` | number of row-level results that include prompt text |
| `prompt_templates_count` | number of unique redacted prompt templates used |
| `prompt_templates` | deduplicated redacted prompt templates used in the run |

## Prompt Audit Trail

Full JSON includes the provider-facing prompt template for each evaluated row,
with patient data redacted:

```python
from krisis.results.report import format_json_report

full = format_json_report(result, include_results=True)
```

Each row in `evaluation_results` can include:

| Field | Meaning |
|---|---|
| `prompt` | Provider-facing messages serialized as JSON, with patient data redacted |
| `prompt_mode` | `single` or `batch` |
| `raw_response` | Provider response for that row |

For batched calls, every row in the same provider batch carries the same redacted
batch prompt. This is intentional: it lets you inspect the instructions and
batched JSON shape when comparing reliability across providers, without storing
the full patient payload repeatedly.

Aggregate JSON also includes deduplicated templates under
`execution.prompt_templates`, so you can inspect the prompt shape without using
`include_results=True`.

## Plotting Model Comparisons

For bar charts across models, the most useful fields are:

| Plot | JSON path |
|---|---|
| Overall accuracy | `metrics.Accuracy.value` |
| Balanced accuracy | `metrics.Balanced Accuracy.value` |
| Answered-only accuracy | `metrics.Selective Accuracy (answered only).value` |
| Abstention rate | `metrics.Abstention Rate.value` |
| Coverage | `metrics.Answer Rate (Coverage).value` |
| Deferral alignment | `metrics.Deferral Alignment.value` |
| Calibration error | `metrics.Expected Calibration Error.value` |
| Runtime | `execution.elapsed_seconds` |
| Throughput | `execution.records_per_second` |
| Token use | `execution.token_total` |

!!! tip "Use grouped charts by task"
    Detection, staging, and progression measure different behaviors. Compare
    models within the same task instead of mixing all tasks into one score.

## Recommended Artifacts To Save

For reproducible comparisons, save:

- metrics-only JSON for each model/task run
- the full JSON for at least one audited run per model
- the exact model ID
- task name
- feature set
- seed
- synthetic record count
- batch size and max concurrency
- output-token cap
