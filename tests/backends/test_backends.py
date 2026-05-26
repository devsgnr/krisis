"""Backend tests that avoid live provider calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from krisis.backends.api import DEFAULT_API_MODEL, APIBackend
from krisis.backends.batching import parse_batch_response
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


class _FakeAPIClient:
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


class _FlakyAPIClient:
    def __init__(self) -> None:
        self.completions = _FlakyCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_api_backend_uses_structured_chat_completion_request() -> None:
    client = _FakeAPIClient()
    backend = APIBackend(client=client)
    record = PatientRecord(features={"sc": 2.4, "htn": 1.0}, label=0)

    response = backend.evaluate(record, Task.DETECTION)

    assert response.prediction == 0
    assert response.abstained is False
    assert response.confidence == 0.82
    assert response.input_tokens == 50
    assert response.output_tokens == 10
    assert response.total_tokens == 60
    assert response.prompt_mode == "single"
    assert response.prompt[1]["role"] == "user"
    assert "[PATIENT_MARKERS_REDACTED]" in response.prompt[1]["content"]
    assert "2.4" not in str(response.prompt)

    request = client.completions.last_request
    assert request is not None
    assert request["model"] == DEFAULT_API_MODEL
    assert "temperature" not in request
    assert request["max_tokens"] == 1024
    assert request["extra_body"] == {"reasoning": {"effort": "low", "exclude": True}}
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["messages"][0]["role"] == "system"


def test_api_backend_progression_schema_uses_string_predictions() -> None:
    client = _FakeAPIClient()
    backend = APIBackend(client=client)
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


def test_api_backend_batches_records_with_stable_ids() -> None:
    client = _FakeAPIClient()
    backend = APIBackend(client=client)
    records = [
        PatientRecord(features={"sc": 2.4}, label=0),
        PatientRecord(features={"sc": 1.0}, label=1),
    ]

    responses = backend.evaluate_batch(records, Task.DETECTION)

    assert [r.prediction for r in responses] == [0, None]
    assert [r.abstained for r in responses] == [False, True]
    assert [r.input_tokens for r in responses] == [50, 50]
    assert [r.output_tokens for r in responses] == [10, 10]
    assert all(r.prompt_mode == "batch" for r in responses)
    assert all(
        "[BATCH_PATIENT_DATA_REDACTED]" in r.prompt[1]["content"] for r in responses
    )

    request = client.completions.last_request
    assert request is not None
    assert request["max_tokens"] == 2048
    assert request["extra_body"]["reasoning"]["effort"] == "low"
    schema = request["response_format"]["json_schema"]["schema"]
    assert "results" in schema["properties"]
    assert "case_0" in request["messages"][1]["content"]
    assert "case_1" in request["messages"][1]["content"]


def test_api_backend_can_omit_reasoning_config() -> None:
    client = _FakeAPIClient()
    backend = APIBackend(client=client, reasoning_effort=None)
    record = PatientRecord(features={"sc": 2.4}, label=0)

    backend.evaluate(record, Task.DETECTION)

    request = client.completions.last_request
    assert request is not None
    assert "extra_body" not in request


def test_api_backend_retries_transient_provider_errors() -> None:
    client = _FlakyAPIClient()
    backend = APIBackend(
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


def test_batch_parser_repairs_missing_commas_between_objects() -> None:
    raw = (
        '{"results":['
        '{"id":"case_0","abstained":false,"confidence":0.81,"prediction":0}'
        '{"id":"case_1","abstained":true,"confidence":0.43,"prediction":null}'
        "]}"
    )

    responses = parse_batch_response(raw, Task.DETECTION, 2)

    assert [r.prediction for r in responses] == [0, None]
    assert [r.abstained for r in responses] == [False, True]
