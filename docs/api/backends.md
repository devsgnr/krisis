# Backend

The backend page defines the reusable interface that model backends implement.

!!! note "Provider implementations live in the guide"
    The API reference focuses on the common backend contract and the primary
    API backend. Usage is documented under Framework Guide -> Model Backends.

## Backend Base Classes

::: krisis.backends.base
    options:
      members:
        - BackendResponse
        - BaseBackend

## API Backend

::: krisis.backends.api
    options:
      members:
        - APIBackend
        - make_api_backend

## Transformers Backend

::: krisis.backends.huggingface
    options:
      members:
        - TransformersBackend
        - make_transformers_backend

## Provider Backend Controls

The primary backend is `APIBackend`.

Get an API key from [OpenRouter](https://openrouter.ai/settings/keys), then set
it as `OPENROUTER_API_KEY` or pass it through the `api_key` parameter.

| Control | Default | Purpose |
|---|---|---|
| `model` | `openai/gpt-5.5` | OpenRouter-routed model ID |
| `temperature` | `None` | Sampling temperature. `0.0` or `None` is recommended for evals |
| `max_tokens` | `1024` | Per-row output token cap |
| `reasoning_effort` | `low` | Reasoning effort for supported models |
| `exclude_reasoning` | `True` | Uses reasoning internally but keeps reasoning text out of parsed output |
| `api_key` | `OPENROUTER_API_KEY` | Direct key override or environment fallback |
| `base_url` | `https://openrouter.ai/api/v1` | API base URL |
| `client` | `None` | Prebuilt OpenAI-compatible client for testing or custom setup |
| `max_retries` | `2` | Number of retries after transient failures |
| `retry_base_seconds` | `0.5` | Initial exponential-backoff delay |
| `retry_max_seconds` | `8.0` | Maximum exponential-backoff delay |

For local Hugging Face models, the experimental `TransformersBackend` accepts a
`model_id` and defaults to `device="cpu"`. Use `device="cuda"` in GPU notebooks
such as Colab or Deepnote. For gated models, pass `hf_token` directly or set
`HF_TOKEN`.

| Control | Default | Purpose |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model ID |
| `device` | `cpu` | Runtime device, such as `cpu` or `cuda` |
| `dtype` | `None` | Optional torch dtype, such as `bfloat16` |
| `max_new_tokens` | `1024` | Per-row generated token cap |
| `hf_token` | `HF_TOKEN` | Access token for gated models |
| `trust_remote_code` | `False` | Allows custom model code when required |

Default token caps are intentionally conservative. `APIBackend` defaults to
`1024` output tokens per row because larger reasoning models can spend part of
the completion budget before producing the visible JSON.

Example:

```python
backend = APIBackend(
    model="openai/gpt-5.5",
    api_key="YOUR_OPENROUTER_API_KEY",
    temperature=0.0,
    max_tokens=1024,
    reasoning_effort="low",
    max_retries=2,
    retry_base_seconds=0.5,
    retry_max_seconds=8.0,
)
```

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
