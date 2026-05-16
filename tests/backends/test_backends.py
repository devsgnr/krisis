"""Backend tests that avoid live provider calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from krisis.backends.anthropic import DEFAULT_ANTHROPIC_MODEL, AnthropicBackend
from krisis.backends.gemini import DEFAULT_GEMINI_MODEL, GeminiBackend
from krisis.backends.grok import DEFAULT_GROK_MODEL, GrokBackend
from krisis.backends.openai import DEFAULT_OPENAI_MODEL, OpenAIBackend
from krisis.backends.retry import is_retryable_exception
from krisis.data.base import PatientRecord, Task


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        schema_name = (
            kwargs.get("response_format", {}).get("json_schema", {}).get("name", "")
        )
        if "batch" in schema_name:
            message = SimpleNamespace(
                content=(
                    '{"results":['
                    '{"id":"case_1","abstained":true,"confidence":0.4,"prediction":null},'
                    '{"id":"case_0","abstained":false,"confidence":0.82,"prediction":0}'
                    "]}"
                )
            )
            usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=usage,
            )
        message = SimpleNamespace(
            content='{"abstained": false, "confidence": 0.82, "prediction": 0}'
        )
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeRateLimitError(Exception):
    status_code = 429


class _FakeAuthError(Exception):
    status_code = 401


class _FlakyCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise _FakeRateLimitError("slow down")
        message = SimpleNamespace(
            content='{"abstained": false, "confidence": 0.7, "prediction": 0}'
        )
        usage = SimpleNamespace(prompt_tokens=40, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FlakyOpenAIClient:
    def __init__(self) -> None:
        self.completions = _FlakyCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_backend_uses_structured_chat_completion_request() -> None:
    client = _FakeOpenAIClient()
    backend = OpenAIBackend(client=client)
    record = PatientRecord(features={"sc": 2.4, "htn": 1.0}, label=0)

    response = backend.evaluate(record, Task.DETECTION)

    assert response.prediction == 0
    assert response.abstained is False
    assert response.confidence == 0.82
    assert response.input_tokens == 50
    assert response.output_tokens == 10
    assert response.total_tokens == 60

    request = client.completions.last_request
    assert request is not None
    assert request["model"] == DEFAULT_OPENAI_MODEL
    assert "temperature" not in request
    assert request["max_completion_tokens"] == 256
    assert request["store"] is False
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["messages"][0]["role"] == "system"


def test_openai_backend_progression_schema_uses_string_predictions() -> None:
    client = _FakeOpenAIClient()
    backend = OpenAIBackend(client=client)
    record = PatientRecord(
        features={
            "trajectory_months": 6,
            "baseline": {"sc": 1.0},
            "current": {"sc": 1.4},
        },
        label="worsening",
    )

    backend.evaluate(record, Task.PROGRESSION)

    request = client.completions.last_request
    assert request is not None
    schema = request["response_format"]["json_schema"]["schema"]
    prediction = schema["properties"]["prediction"]
    assert prediction["enum"] == ["stable", "worsening", "improving", None]


def test_openai_backend_batches_records_with_stable_ids() -> None:
    client = _FakeOpenAIClient()
    backend = OpenAIBackend(client=client)
    records = [
        PatientRecord(features={"sc": 2.4}, label=0),
        PatientRecord(features={"sc": 1.0}, label=1),
    ]

    responses = backend.evaluate_batch(records, Task.DETECTION)

    assert [r.prediction for r in responses] == [0, None]
    assert [r.abstained for r in responses] == [False, True]
    assert [r.input_tokens for r in responses] == [50, 50]
    assert [r.output_tokens for r in responses] == [10, 10]

    request = client.completions.last_request
    assert request is not None
    assert request["max_completion_tokens"] == 512
    schema = request["response_format"]["json_schema"]["schema"]
    assert "results" in schema["properties"]
    assert "case_0" in request["messages"][1]["content"]
    assert "case_1" in request["messages"][1]["content"]


def test_openai_backend_retries_transient_provider_errors() -> None:
    client = _FlakyOpenAIClient()
    backend = OpenAIBackend(
        client=client,
        max_retries=1,
        retry_base_seconds=0.0,
    )
    record = PatientRecord(features={"sc": 2.4}, label=0)

    response = backend.evaluate(record, Task.DETECTION)

    assert response.prediction == 0
    assert client.completions.calls == 2


def test_retry_classifier_does_not_retry_auth_errors() -> None:
    assert is_retryable_exception(_FakeRateLimitError("slow down")) is True
    assert is_retryable_exception(_FakeAuthError("bad key")) is False


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.last_request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        first_message = kwargs.get("messages", [{}])[0].get("content", "")
        if "Case id: case_0" in first_message:
            block = SimpleNamespace(
                text=(
                    '{"results":['
                    '{"id":"case_0","abstained":false,"confidence":0.77,"prediction":1},'
                    '{"id":"case_1","abstained":true,"confidence":0.5,"prediction":null}'
                    "]}"
                )
            )
            usage = SimpleNamespace(input_tokens=80, output_tokens=16)
            return SimpleNamespace(content=[block], usage=usage)
        block = SimpleNamespace(
            text='{"abstained": false, "confidence": 0.77, "prediction": 1}'
        )
        usage = SimpleNamespace(input_tokens=40, output_tokens=8)
        return SimpleNamespace(content=[block], usage=usage)


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


def test_anthropic_backend_uses_messages_request() -> None:
    client = _FakeAnthropicClient()
    backend = AnthropicBackend(client=client)
    record = PatientRecord(features={"sc": 0.9, "htn": 0.0}, label=1)

    response = backend.evaluate(record, Task.DETECTION)

    assert response.prediction == 1
    assert response.abstained is False
    assert response.confidence == 0.77
    assert response.input_tokens == 40
    assert response.output_tokens == 8

    request = client.messages.last_request
    assert request is not None
    assert request["model"] == DEFAULT_ANTHROPIC_MODEL
    assert request["max_tokens"] == 256
    assert "temperature" not in request
    assert request["system"].startswith("You are a careful clinical assistant")
    assert request["messages"][0]["role"] == "user"


def test_anthropic_backend_batches_records() -> None:
    client = _FakeAnthropicClient()
    backend = AnthropicBackend(client=client)
    records = [
        PatientRecord(features={"sc": 0.9}, label=1),
        PatientRecord(features={"sc": 3.1}, label=0),
    ]

    responses = backend.evaluate_batch(records, Task.DETECTION)

    assert [r.prediction for r in responses] == [1, None]
    assert [r.abstained for r in responses] == [False, True]
    assert [r.input_tokens for r in responses] == [40, 40]
    assert [r.output_tokens for r in responses] == [8, 8]

    request = client.messages.last_request
    assert request is not None
    assert request["max_tokens"] == 512
    assert "Batch mode rules" in request["system"]
    assert "case_0" in request["messages"][0]["content"]
    assert "case_1" in request["messages"][0]["content"]


def test_grok_backend_uses_openai_compatible_request() -> None:
    client = _FakeOpenAIClient()
    backend = GrokBackend(client=client)
    record = PatientRecord(features={"sc": 2.4, "htn": 1.0}, label=0)

    response = backend.evaluate(record, Task.DETECTION)

    assert response.prediction == 0
    assert response.abstained is False
    assert response.confidence == 0.82
    assert response.input_tokens == 50
    assert response.output_tokens == 10

    request = client.completions.last_request
    assert request is not None
    assert request["model"] == DEFAULT_GROK_MODEL
    assert request["max_tokens"] == 256
    assert request["response_format"]["type"] == "json_schema"
    assert request["messages"][0]["role"] == "system"
    assert "store" not in request


def test_grok_backend_batches_records() -> None:
    client = _FakeOpenAIClient()
    backend = GrokBackend(client=client)
    records = [
        PatientRecord(features={"sc": 2.4}, label=0),
        PatientRecord(features={"sc": 1.0}, label=1),
    ]

    responses = backend.evaluate_batch(records, Task.DETECTION)

    assert [r.prediction for r in responses] == [0, None]
    assert [r.abstained for r in responses] == [False, True]

    request = client.completions.last_request
    assert request is not None
    assert request["max_tokens"] == 512
    assert (
        "results" in request["response_format"]["json_schema"]["schema"]["properties"]
    )
    assert "case_0" in request["messages"][1]["content"]
    assert "case_1" in request["messages"][1]["content"]


class _FakeGeminiModels:
    def __init__(self) -> None:
        self.last_request: dict[str, Any] | None = None

    def generate_content(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        schema = kwargs.get("config", {}).get("response_json_schema", {})
        if "results" in schema.get("properties", {}):
            usage = SimpleNamespace(prompt_token_count=90, candidates_token_count=18)
            return SimpleNamespace(
                text=(
                    '{"results":['
                    '{"id":"case_0","abstained":false,"confidence":0.81,"prediction":0},'
                    '{"id":"case_1","abstained":true,"confidence":0.43,"prediction":null}'
                    "]}"
                ),
                usage_metadata=usage,
            )
        usage = SimpleNamespace(prompt_token_count=45, candidates_token_count=9)
        return SimpleNamespace(
            text='{"abstained": false, "confidence": 0.81, "prediction": 0}',
            usage_metadata=usage,
        )


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.models = _FakeGeminiModels()


def test_gemini_backend_uses_structured_output_request() -> None:
    client = _FakeGeminiClient()
    backend = GeminiBackend(client=client)
    record = PatientRecord(features={"sc": 2.4, "htn": 1.0}, label=0)

    response = backend.evaluate(record, Task.DETECTION)

    assert response.prediction == 0
    assert response.abstained is False
    assert response.confidence == 0.81
    assert response.input_tokens == 45
    assert response.output_tokens == 9
    assert response.total_tokens == 54

    request = client.models.last_request
    assert request is not None
    assert request["model"] == DEFAULT_GEMINI_MODEL
    assert request["config"]["response_mime_type"] == "application/json"
    assert "temperature" not in request["config"]
    assert request["config"]["max_output_tokens"] == 256
    assert "system_instruction" in request["config"]
    schema = request["config"]["response_json_schema"]
    assert schema["properties"]["prediction"]["type"] == ["integer", "null"]


def test_gemini_backend_batches_records() -> None:
    client = _FakeGeminiClient()
    backend = GeminiBackend(client=client)
    records = [
        PatientRecord(features={"sc": 2.4}, label=0),
        PatientRecord(features={"sc": 1.0}, label=1),
    ]

    responses = backend.evaluate_batch(records, Task.DETECTION)

    assert [r.prediction for r in responses] == [0, None]
    assert [r.abstained for r in responses] == [False, True]
    assert [r.input_tokens for r in responses] == [45, 45]
    assert [r.output_tokens for r in responses] == [9, 9]

    request = client.models.last_request
    assert request is not None
    assert request["config"]["max_output_tokens"] == 512
    assert "results" in request["config"]["response_json_schema"]["properties"]
    assert "case_0" in request["contents"]
    assert "case_1" in request["contents"]
