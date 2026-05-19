# Model Backends

Model backends adapt provider APIs into one standard Krisis interface. This is
what makes it possible to run the same clinical task across OpenAI, Anthropic,
Grok, and Gemini while receiving comparable result objects.

## Supported Providers

| Provider | Backend | Default model | Install extra |
|---|---|---|---|
| OpenAI | `OpenAIBackend` | `gpt-5.5` | `krisis[openai]` |
| Anthropic | `AnthropicBackend` | `claude-opus-4-7` | `krisis[anthropic]` |
| Grok | `GrokBackend` | `grok-4.3` | `krisis[grok]` |
| Google Gemini | `GeminiBackend` | `gemini-3-pro-preview` | `krisis[gemini]` |

Install only the provider clients you need:

```bash
pip install "krisis[openai]"
pip install "krisis[anthropic]"
pip install "krisis[grok]"
pip install "krisis[gemini]"
```

Or install all provider extras:

```bash
pip install "krisis[all]"
```

## Backend Contract

Every backend returns the same response shape:

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

Field meanings:

| Field | Meaning |
|---|---|
| `prediction` | Parsed model prediction, or `None` when the model abstained |
| `abstained` | Whether the model declined to answer |
| `confidence` | Model-reported confidence between 0 and 1 |
| `raw_response` | Raw provider text or JSON for auditability |
| `prompt` | Prompt/messages with patient data redacted |
| `prompt_mode` | `single` for one-row calls, `batch` for batched calls |
| `input_tokens` | Provider-reported input token count when available |
| `output_tokens` | Provider-reported output token count when available |
| `total_tokens` | Combined token count when available |

## API Keys

You can pass provider keys directly:

```python
from krisis.backends.openai import OpenAIBackend

backend = OpenAIBackend(api_key="YOUR_OPENAI_KEY")
```

Provider examples:

```python
from krisis.backends.anthropic import AnthropicBackend
from krisis.backends.gemini import GeminiBackend
from krisis.backends.grok import GrokBackend

anthropic_backend = AnthropicBackend(api_key="YOUR_ANTHROPIC_KEY")
gemini_backend = GeminiBackend(api_key="YOUR_GEMINI_KEY")
grok_backend = GrokBackend(api_key="YOUR_XAI_KEY")
```

The example runner supports the generic `API_KEY` environment variable and
provider-native environment variables where available.

!!! warning "Do not commit API keys"
    Keep API keys in environment variables, local secret managers, or CI
    secrets. Never put provider keys in benchmark scripts committed to GitHub.

## Batching And Concurrency

Krisis can evaluate records in provider batches:

```python
from krisis.benchmark import Benchmark

result = Benchmark(
    suite,
    backend,
    batch_size=8,
    max_concurrency=2,
).run()
```

`batch_size` and `max_concurrency` control different things:

| Setting | Meaning |
|---|---|
| `batch_size` | How many patient records are sent in one provider call |
| `max_concurrency` | How many provider calls may run at the same time |
| `max_output_tokens` | Per-row output-token cap passed to the provider backend |

Example: `batch_size=8` and `max_concurrency=2` can evaluate up to 16 records in
flight, split across two API calls.

!!! tip "Start conservative"
    Use `batch_size=8` and `max_concurrency=1` or `2` for first runs. Increase
    only after checking provider rate limits and structured JSON reliability.

!!! tip "Empty provider responses"
    If a provider returns an empty response or truncated JSON, increase the
    example runner's `--max-output-tokens` value. For larger frontier models,
    also try a smaller `--batch-size` so each call has less JSON to produce.

!!! note "OpenAI token cap"
    The OpenAI backend defaults to `max_completion_tokens=1024` per row. This is
    higher than the visible JSON usually needs because OpenAI reasoning models
    may consume part of the completion budget before emitting the final JSON.

## Structured Output

Backends request structured JSON when the provider supports it. The expected
single-row response is:

```json
{
  "abstained": false,
  "confidence": 0.82,
  "prediction": 0
}
```

Batched responses use:

```json
{
  "results": [
    {
      "id": "case_0",
      "abstained": false,
      "confidence": 0.82,
      "prediction": 0
    }
  ]
}
```

If a batched response is malformed, Krisis can shrink the batch and retry smaller
groups before falling back to single-row evaluation.

## Failure Handling

Backend hardening includes:

- retry with backoff for transient provider failures
- parsing safeguards for markdown-wrapped or malformed JSON
- recursive batch shrinking when batched JSON fails
- single-row fallback for difficult cases
- actionable empty-response errors when output-token caps are too low
- raw response preservation for debugging
- token usage aggregation in benchmark execution metadata

!!! note "Batching reduces calls, not reasoning time"
    Large frontier models may still take time to reason over batched clinical
    cases. Batching usually reduces HTTP overhead and rate-limit pressure, but
    it does not make the model itself free.

## Choosing A Model

For fast smoke tests, use a lightweight provider model and small synthetic
counts. For publishable comparisons, use fixed seeds, fixed task settings, and
the same batch/concurrency settings across models where provider limits allow.

Always report:

- provider
- model ID
- task
- feature set
- batch size
- max concurrency
- output-token cap
- elapsed seconds
- token total
