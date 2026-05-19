# Backend

The backend page defines the reusable interface that provider-specific backends
implement.

!!! note "Provider implementations live in the guide"
    The API reference focuses on the common backend contract. Provider-specific
    usage for OpenAI, Anthropic, Grok, and Gemini is documented under Framework
    Guide -> Model Backends.

## Backend Base Classes

::: krisis.backends.base
    options:
      members:
        - BackendResponse
        - BaseBackend

## Provider Backend Controls

Concrete provider backends follow the same practical pattern even though each
provider SDK has slightly different parameter names.

| Control | OpenAI | Anthropic | Grok | Gemini | Purpose |
|---|---|---|---|---|---|
| `model` | yes | yes | yes | yes | Provider model ID |
| `temperature` | yes | yes | yes | yes | Sampling temperature. `0.0` or `None` is recommended for evals |
| token cap | `max_completion_tokens` | `max_tokens` | `max_tokens` | `max_output_tokens` | Caps generated output tokens |
| `api_key` | yes | yes | yes | yes | Direct provider key override |
| `client` | yes | yes | yes | yes | Prebuilt client for testing or custom setup |
| `max_retries` | yes | yes | yes | yes | Number of retries after transient failures |
| `retry_base_seconds` | yes | yes | yes | yes | Initial exponential-backoff delay |
| `retry_max_seconds` | yes | yes | yes | yes | Maximum exponential-backoff delay |

Default token caps are intentionally conservative for most providers. OpenAI
defaults to `max_completion_tokens=1024` per row because larger reasoning models
can spend part of the completion budget before producing the visible JSON.

Example:

```python
backend = OpenAIBackend(
    model="provider-model-id",
    api_key="YOUR_API_KEY",
    temperature=0.0,
    max_completion_tokens=1024,
    max_retries=2,
    retry_base_seconds=0.5,
    retry_max_seconds=8.0,
)
```

!!! note "Provider naming differs"
    `max_completion_tokens` is the OpenAI name. Anthropic and Grok use
    `max_tokens`; Gemini uses `max_output_tokens`. The example CLI exposes one
    provider-agnostic flag, `--max-output-tokens`, and maps it to the correct
    backend setting.

!!! note "Actual provider classes"
    Provider-specific setup examples live in Framework Guide -> Model Backends.
    The API Reference keeps the focus on the shared backend shape.

## Retry Behavior

Krisis retries transient provider failures, including common timeout, connection,
rate-limit, overloaded, and 5xx-style errors.

The retry controls are:

| Parameter | Default | Meaning |
|---|---:|---|
| `max_retries` | `2` | Number of retries after the first failed attempt |
| `retry_base_seconds` | `0.5` | Initial backoff delay |
| `retry_max_seconds` | `8.0` | Maximum backoff delay |

`max_retries=2` means up to three total attempts:

1. first attempt
2. first retry
3. second retry

Retry delays use exponential backoff:

```text
delay = min(retry_max_seconds, retry_base_seconds * (2 ** attempt))
```

Small jitter is added to reduce synchronized retry spikes.

## Batched Evaluation

Backends expose two methods:

| Method | Meaning |
|---|---|
| `evaluate(record, task)` | Evaluate one patient row |
| `evaluate_batch(records, task)` | Evaluate a batch of patient rows |

The base implementation of `evaluate_batch` loops over `evaluate`. Provider
backends can override it to send one prompt containing multiple patient rows and
return one response array.

`Benchmark` is responsible for deciding batch size and concurrency. Backends are
responsible for turning one batch into `BackendResponse` objects.

## Token Usage

`BackendResponse` includes prompt and usage audit fields:

- `prompt`
- `prompt_mode`
- `input_tokens`
- `output_tokens`
- `total_tokens`

When providers expose usage metadata, Krisis records it per row and aggregates
it into `BenchmarkResult.extras.token_total`. A redacted prompt template is
preserved per row in full JSON so provider behavior can be reviewed alongside
the instructions/output shape the model received.

## Shared Usage Helpers

::: krisis.backends.usage

## Retry Helpers

::: krisis.backends.retry
    options:
      members:
        - is_retryable_exception
        - call_with_retries
