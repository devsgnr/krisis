"""
Hugging Face Transformers backend.

This experimental backend runs local/open-weight models through ``transformers``
instead of calling a hosted API. It defaults to CPU so it works in notebooks and
local environments, while allowing users to pass ``device="cuda"`` on GPU
runtimes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from krisis.backends.base import (
    BackendResponse,
    BaseBackend,
    format_messages_for_audit,
)
from krisis.backends.batching import (
    attach_prompt_metadata,
    build_batch_messages,
    distribute_usage_over_batch,
    parse_batch_response,
)
from krisis.backends.usage import TokenUsage
from krisis.data.base import PatientRecord, Task
from krisis.prompts.base import build_messages
from krisis.tasks.base import parse_model_response

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HF_BACKEND_EXPERIMENTAL = True


class TransformersBackend(BaseBackend):
    """
    Experimental local Hugging Face Transformers backend.

    This backend is intended for GPU notebooks and local experimentation. CPU
    runs are supported for smoke tests, but full CKD benchmark runs are expected
    to be slow without GPU acceleration.

    Args:
        model_id: Hugging Face model id, e.g. ``Qwen/Qwen2.5-7B-Instruct``.
        device: execution device. Defaults to ``cpu``. Use ``cuda`` in GPU
            runtimes such as Colab or Deepnote.
        dtype: optional torch dtype string (``float16``, ``bfloat16``,
            ``float32``) or torch dtype object.
        max_new_tokens: maximum generated tokens per row.
        temperature: optional decoding temperature. Leave as ``None`` for
            deterministic eval defaults.
        do_sample: whether to sample. Defaults to ``False``.
        trust_remote_code: passed to Hugging Face loaders.
        hf_token: optional Hugging Face access token for gated models. Defaults
            to ``HF_TOKEN`` when set in the environment.
        tokenizer: optional preloaded tokenizer, mainly for tests/custom setup.
        model: optional preloaded model, mainly for tests/custom setup.
        generator: optional callable test hook that receives prompts and returns
            raw model strings.
        tokenizer_kwargs: extra kwargs passed to ``AutoTokenizer.from_pretrained``.
        model_kwargs: extra kwargs passed to ``AutoModelForCausalLM.from_pretrained``.
    """

    experimental = HF_BACKEND_EXPERIMENTAL

    def __init__(
        self,
        model_id: str = DEFAULT_HF_MODEL,
        device: str = "cpu",
        dtype: str | Any | None = None,
        max_new_tokens: int = 1024,
        temperature: float | None = None,
        do_sample: bool = False,
        trust_remote_code: bool = False,
        hf_token: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        generator: Callable[[list[str]], list[str]] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample
        self._trust_remote_code = trust_remote_code
        self._hf_token = hf_token or os.getenv("HF_TOKEN")
        self._generator = generator

        self._tokenizer = tokenizer
        self._model = model
        if self._generator is None and (self._tokenizer is None or self._model is None):
            self._load_transformers_objects(
                tokenizer_kwargs=tokenizer_kwargs or {},
                model_kwargs=model_kwargs or {},
            )

    @property
    def name(self) -> str:
        return f"hf:{self._model_id}"

    def evaluate(self, record: PatientRecord, task: Task) -> BackendResponse:
        messages = build_messages(record, task)
        prompt = format_messages_for_audit(messages)
        raw, usage = self._generate_from_messages([messages])
        raw_text = raw[0].strip()
        parsed = parse_model_response(raw_text, task)
        if (
            parsed.prediction is None
            and not parsed.abstained
            and parsed.confidence is None
        ):
            raise ValueError(
                f"{self.name} returned a non-JSON response that could not be parsed."
            )

        return BackendResponse(
            prediction=parsed.prediction,
            abstained=parsed.abstained,
            confidence=parsed.confidence,
            raw_response=raw_text,
            prompt=prompt,
            prompt_mode="single",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    def evaluate_batch(
        self,
        records: list[PatientRecord],
        task: Task,
    ) -> list[BackendResponse]:
        if not records:
            return []

        messages = build_batch_messages(records, task)
        prompt = format_messages_for_audit(messages)
        raw, usage = self._generate_from_messages(
            [messages],
            max_new_tokens_multiplier=len(records),
        )
        raw_text = raw[0].strip()
        responses = parse_batch_response(raw_text, task, len(records))
        attach_prompt_metadata(responses, prompt=prompt, prompt_mode="batch")
        return distribute_usage_over_batch(responses, usage)

    def _load_transformers_objects(
        self,
        *,
        tokenizer_kwargs: dict[str, Any],
        model_kwargs: dict[str, Any],
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "TransformersBackend requires Hugging Face dependencies. "
                "Install with: pip install 'krisis[hf]'"
            ) from exc

        tokenizer_kwargs.setdefault("trust_remote_code", self._trust_remote_code)
        model_kwargs.setdefault("trust_remote_code", self._trust_remote_code)
        if self._hf_token is not None:
            tokenizer_kwargs.setdefault("token", self._hf_token)
            model_kwargs.setdefault("token", self._hf_token)
        if self._dtype is not None:
            model_kwargs.setdefault("torch_dtype", self._resolve_torch_dtype(torch))

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            **tokenizer_kwargs,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            **model_kwargs,
        )
        if hasattr(self._model, "to") and self._device:
            self._model.to(self._device)
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = getattr(self._tokenizer, "eos_token", None)

    def _resolve_torch_dtype(self, torch_module: Any) -> Any:
        if not isinstance(self._dtype, str):
            return self._dtype
        dtype_map = {
            "float16": torch_module.float16,
            "float32": torch_module.float32,
            "bfloat16": torch_module.bfloat16,
        }
        try:
            return dtype_map[self._dtype]
        except KeyError as exc:
            raise ValueError(
                "dtype must be one of: float16, float32, bfloat16, or a torch dtype."
            ) from exc

    def _generate_from_messages(
        self,
        prompts_as_messages: list[list[dict[str, str]]],
        max_new_tokens_multiplier: int = 1,
    ) -> tuple[list[str], TokenUsage]:
        prompts = [self._messages_to_prompt(messages) for messages in prompts_as_messages]
        if self._generator is not None:
            return self._generator(prompts), TokenUsage()

        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        encoded = self._move_encoded_to_device(encoded)
        input_token_counts = self._input_token_counts(encoded)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens * max_new_tokens_multiplier,
            "do_sample": self._do_sample,
        }
        if self._temperature is not None:
            generation_kwargs["temperature"] = self._temperature

        generated = self._model.generate(**encoded, **generation_kwargs)
        raw_texts, output_token_counts = self._decode_generated(
            generated,
            encoded,
            input_token_counts,
        )
        usage = TokenUsage(
            input_tokens=float(sum(input_token_counts)),
            output_tokens=float(sum(output_token_counts)),
        )
        return raw_texts, usage

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self._tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return self._plain_text_prompt(messages)

    @staticmethod
    def _plain_text_prompt(messages: list[dict[str, str]]) -> str:
        blocks = [f"{m['role'].upper()}:\n{m['content']}" for m in messages]
        return "\n\n".join(blocks) + "\n\nASSISTANT:\n"

    def _move_encoded_to_device(self, encoded: Any) -> Any:
        if hasattr(encoded, "to"):
            return encoded.to(self._device)
        for key, value in list(encoded.items()):
            if hasattr(value, "to"):
                encoded[key] = value.to(self._device)
        return encoded

    @staticmethod
    def _input_token_counts(encoded: Any) -> list[int]:
        attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
        if attention_mask is not None:
            return [int(row.sum().item()) for row in attention_mask]

        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "shape") and len(input_ids.shape) == 2:
            return [int(input_ids.shape[1])] * int(input_ids.shape[0])
        return [len(row) for row in input_ids]

    def _decode_generated(
        self,
        generated: Any,
        encoded: Any,
        input_token_counts: list[int],
    ) -> tuple[list[str], list[int]]:
        assert self._tokenizer is not None
        input_ids = encoded["input_ids"]
        input_width = int(input_ids.shape[1]) if hasattr(input_ids, "shape") else 0
        output_ids = []
        output_counts: list[int] = []
        for row in generated:
            completion_ids = row[input_width:]
            output_ids.append(completion_ids)
            output_counts.append(self._sequence_length(completion_ids))
        return (
            self._tokenizer.batch_decode(output_ids, skip_special_tokens=True),
            output_counts,
        )

    @staticmethod
    def _sequence_length(sequence: Any) -> int:
        if hasattr(sequence, "numel"):
            return int(sequence.numel())
        return len(sequence)


def make_transformers_backend(
    model_id: str = DEFAULT_HF_MODEL,
    device: str = "cpu",
    **kwargs: Any,
) -> TransformersBackend:
    """Convenience factory for local Hugging Face Transformers inference."""
    return TransformersBackend(model_id=model_id, device=device, **kwargs)
