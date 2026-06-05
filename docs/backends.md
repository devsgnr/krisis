# Model Backends

Model backends adapt model APIs into one standard Krisis interface. In v0.2,
Krisis uses a single `APIBackend` powered by OpenRouter: the same clinical task
can run across OpenAI, Anthropic, Grok, Gemini, and other routed models by
changing only the model id.

Krisis also includes an experimental `TransformersBackend` for local Hugging
Face models. It defaults to CPU and can use GPU runtimes by passing
`device="cuda"`.

## API Backend

| Backend | Default model | API key |
|---|---|---|
| `APIBackend` | `openai/gpt-5.5` | `OPENROUTER_API_KEY` |

Example model ids:

| Provider route | Example model id |
|---|---|
| OpenAI | `openai/gpt-5.5` |
| Anthropic | `anthropic/claude-opus-4.7` |
| Grok/xAI | `x-ai/grok-4.3` |
| Google Gemini | `google/gemini-3.5-flash` |

```python
from krisis.backends.api import APIBackend

backend = APIBackend(
    model="anthropic/claude-opus-4.7",
    reasoning_effort="low",
)
```

## Local Transformers Backend

Install the Hugging Face extra:

```bash
pip install "krisis[hf]"
```

!!! warning "Experimental backend"
    `TransformersBackend` is experimental in v0.2.7. It is useful for Colab,
    Deepnote, and local GPU experiments, but full benchmark runs on CPU will be
    very slow and the backend API may still change as it is hardened.

!!! warning "Text-generation models only"
    The Hugging Face backend only supports causal text-generation models that
    can be loaded with `AutoModelForCausalLM`. Classifier, embedding,
    masked-language, seq2seq, and multimodal-only models are not supported and
    will raise an error during backend initialization.

Then provide a Hugging Face model id:

```python
from krisis.backends.huggingface import TransformersBackend

backend = TransformersBackend(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
    max_new_tokens=512,
)
```

Gated Hugging Face models can use either the `HF_TOKEN` environment variable or
the explicit `hf_token` argument:

```python
backend = TransformersBackend(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    hf_token="<your-hugging-face-token>",
)
```

For Colab, Deepnote, or another GPU runtime:

```python
backend = TransformersBackend(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    device="cuda",
    dtype="bfloat16",
    max_new_tokens=512,
)
```

!!! warning "Local model performance"
    CPU inference can be very slow. For serious benchmarking with local
    Transformers models, use a GPU runtime and start with a small batch size.

Example script:

```bash
python examples/basic_ckd_hf_eval.py --limit 3 --batch-size 1
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

Create an API key from [OpenRouter](https://openrouter.ai/settings/keys), then
set `OPENROUTER_API_KEY` or pass the key directly:

```python
from krisis.backends.api import APIBackend

backend = APIBackend(api_key="YOUR_OPENROUTER_KEY")
```

The example runner accepts `--api-key`, reads `OPENROUTER_API_KEY` through the
backend, and still supports `API_KEY` as a local convenience when passed from
the examples.

One OpenRouter key can be used with routed model ids such as
`openai/gpt-5.5`, `anthropic/claude-opus-4.7`, `x-ai/grok-4.3`, and
`google/gemini-3.5-flash`, subject to model availability and your OpenRouter
account settings.

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
| `max_output_tokens` | Per-row output-token cap passed to the API backend |
| `reasoning_effort` | API reasoning effort for supported models; defaults to `low` |

Example: `batch_size=8` and `max_concurrency=2` can evaluate up to 16 records in
flight, split across two API calls.

!!! tip "Start conservative"
    Use `batch_size=8` and `max_concurrency=1` or `2` for first runs. Increase
    only after checking provider rate limits and structured JSON reliability.

!!! tip "Empty provider responses"
    If a provider returns an empty response or truncated JSON, increase the
    example runner's `--max-output-tokens` value. For larger frontier models,
    also try a smaller `--batch-size` so each call has less JSON to produce.

!!! note "API token caps"
    `APIBackend` defaults to `1024` output tokens per row. This
    is higher than the visible JSON usually needs because larger reasoning
    models may consume part of the completion budget before emitting the final
    JSON.

!!! note "Reasoning effort"
    `APIBackend` defaults to `reasoning_effort="low"` for faster,
    lower-token schema-constrained evaluation. Pass `reasoning_effort=None` to
    omit the provider parameter entirely. In the example CLI, use
    `--reasoning-effort omit` to omit it.

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
