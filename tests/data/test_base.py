"""Shared data-contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from krisis.data.base import FeatureSet, SuiteConfig, Task


def test_suite_config_coerces_enum_values() -> None:
    config = SuiteConfig(features="full", task="staging")

    assert config.features is FeatureSet.FULL
    assert config.task is Task.STAGING


def test_suite_config_rejects_invalid_synthetic_count() -> None:
    with pytest.raises(ValidationError):
        SuiteConfig(n_synthetic=-1)


def test_suite_config_rejects_invalid_test_size() -> None:
    with pytest.raises(ValidationError):
        SuiteConfig(test_size=1.0)
