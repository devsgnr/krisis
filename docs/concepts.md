# Core Concepts

Krisis is built from a small set of composable contracts. Understanding these
contracts makes it much easier to add suites, backends, or metrics.

## PatientRecord

`PatientRecord` is the unit of evaluation.

```python
PatientRecord(
    features={"sc": 2.4, "hemo": 10.1, "htn": 1},
    label=0,
    metadata={"ckd_stage": 3, "egfr": 42.1, "should_abstain": False},
)
```

| Field | Visible to model? | Purpose |
|---|---:|---|
| `features` | Yes | Clinical input shown to the LLM |
| `label` | No | Ground truth used for scoring |
| `metadata` | No | Extra scoring/context fields such as stage, eGFR, or deferral labels |

!!! tip "Why metadata is separate"
    Deferral labels such as `should_abstain` are intentionally hidden from the
    model. Metrics use them later to check whether the model deferred when it
    should have.

## Suite

A suite prepares a clinical benchmark.

For CKD, the suite:

1. loads a local UCI CKD CSV
2. validates the CSV schema
3. imputes missing values
4. engineers eGFR, sex, and CKD stage
5. splits real held-out records
6. optionally generates synthetic records
7. returns `PatientRecord` rows

The public suite contract is:

```python
records = suite.load()
summary = suite.describe()
```

## Backend

A backend adapts a model provider to Krisis.

Each backend receives a `PatientRecord` and returns:

```python
BackendResponse(
    prediction=...,
    abstained=...,
    confidence=...,
    raw_response=...,
    prompt=...,
    prompt_mode=...,
    input_tokens=...,
    output_tokens=...,
    total_tokens=...,
)
```

`prompt` preserves the provider-facing instructions with patient data redacted.
`prompt_mode` is `single` for one-row calls and `batch` for batched calls. This
helps compare model behavior when one provider follows batched JSON instructions
better than another, without duplicating clinical row data inside every result.

This standard shape lets the same benchmark compare OpenAI, Anthropic, Grok,
and Gemini.

## Benchmark

`Benchmark` is the execution engine.

It handles:

- loading records from a suite
- batching records into provider calls
- running batches concurrently
- retrying transient failures
- shrinking malformed batches when needed
- collecting `EvaluationResult` rows
- computing metrics
- returning a `BenchmarkResult`

!!! important "Batch fallback"
    Some models occasionally fail to return valid JSON for a batch. Krisis
    recursively shrinks the batch and eventually falls back to single-row
    evaluation, preserving benchmark progress while still using batching when
    it works.

    In practice, this has shown up with `gpt-5.5` when the OpenAI
    `max_completion_tokens` cap is too low: the provider can return an empty or
    incomplete JSON response before the model finishes. Krisis now uses a higher
    OpenAI default.

## Metric

Metrics operate on `EvaluationResult` rows. They do not call model APIs and do
not know which provider produced the outputs.

This separation keeps scoring reproducible and backend-agnostic.

## Report

`BenchmarkResult` stores all outputs. Report helpers serialize it as:

- compact text
- full JSON
- metrics-only JSON

Use metrics-only JSON when creating model comparison plots.
