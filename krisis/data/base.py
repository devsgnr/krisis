"""
krisis/data/base.py

Abstract base classes for the Krisis data layer.
All domain-specific data modules (CKD, Hypertension, Diabetes)
inherit from these contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# ── Enums ─────────────────────────────────────────────────────────────────────


class FeatureSet(str, Enum):
    """
    Controls which feature set a suite exposes to the benchmark.

    FULL    — all features in the raw dataset, including weak signals
              and clinically ambiguous markers. Use this to stress-test
              models on messy, real-world input.

    REDUCED — the validated subset derived from feature selection
              (e.g. Pearson correlation). Tighter signal, less noise.
              Use this for focused clinical reasoning evaluation.
    """

    FULL = "full"
    REDUCED = "reduced"


class Task(str, Enum):
    """
    The clinical reasoning task the benchmark will evaluate.

    DETECTION   — binary: condition present vs. not present.
                  The simplest task. Most models handle this reasonably.

    STAGING     — multiclass: assign the correct disease stage
                  (e.g. CKD Stage 1–5). Harder. Requires understanding
                  of clinical thresholds.

    PROGRESSION — temporal: given a patient trajectory across multiple
                  time points, predict direction of disease progression.
                  The hardest task. Requires longitudinal reasoning.
    """

    DETECTION = "detection"
    STAGING = "staging"
    PROGRESSION = "progression"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class PatientRecord:
    """
    A single patient record passed to a model backend for evaluation.

    features    — the clinical markers for this patient as a dict
                  e.g. {"sc": 1.2, "hemo": 10.4, "htn": "yes", ...}

    label       — the ground truth label for this record.
                  For DETECTION: 0 or 1
                  For STAGING: integer stage (1–5)
                  For PROGRESSION: direction string ("stable", "worsening",
                  "improving") or next-stage integer

    metadata    — optional dict for anything that shouldn't be passed to
                  the model but is useful for result analysis
                  e.g. {"egfr": 42.3, "ckd_stage": 3, "sex": "female"}
    """

    features: dict[str, Any]
    label: int | str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteConfig:
    """
    Configuration passed to a BaseDataSuite at instantiation.

    features    — FULL or REDUCED feature set
    task        — DETECTION, STAGING, or PROGRESSION
    seed        — random seed for reproducibility across all
                  stochastic operations (imputation, generation, splits)
    n_synthetic — number of synthetic patient records to generate.
                  Set to 0 to use only real records from the source dataset.
    test_size   — fraction of records held out for evaluation (0.0–1.0)
    """

    features: FeatureSet = FeatureSet.REDUCED
    task: Task = Task.DETECTION
    seed: int = 42
    n_synthetic: int = 200
    test_size: float = 0.2


# ── Abstract base classes ─────────────────────────────────────────────────────


class BasePreprocessor(ABC):
    """
    Cleans and imputes raw domain data.

    Each domain implements this to handle its own encoding,
    imputation strategy, and scaling. The contract is simple:
    fit_transform takes a raw DataFrame and returns a clean one.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._is_fitted = False

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing on df and return the transformed DataFrame.
        Sets self._is_fitted = True on completion.
        """
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply already-fitted preprocessing to new data.
        Raises RuntimeError if called before fit_transform.
        """
        ...

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling "
                "transform(). Call fit_transform() first."
            )


class BaseFeatureEngineer(ABC):
    """
    Derives new clinically meaningful features from preprocessed data.

    This is where domain-specific engineering happens:
    - CKD: eGFR computation, sex generation, stage derivation
    - Hypertension: MAP, pulse pressure, BP stage
    - Diabetes: HbA1c staging, insulin resistance markers

    The engineer sits between the preprocessor and the generator —
    it operates on clean data and produces an enriched DataFrame
    that the generator can sample from.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features and return the enriched DataFrame.
        """
        ...

    @abstractmethod
    def get_feature_names(self, feature_set: FeatureSet) -> list[str]:
        """
        Return the list of feature column names for the given feature set.
        Used by the suite to select the right columns before passing
        records to the model backend.
        """
        ...


class BaseGenerator(ABC):
    """
    Generates synthetic patient records from a fitted distribution.

    Synthetic generation in Krisis is stage-aware — records are
    generated along physiologically plausible disease progression arcs,
    not sampled randomly. This ensures the benchmark tests models on
    clinically coherent inputs rather than statistical noise.

    The generator is seeded for reproducibility. Two researchers running
    the same suite with the same seed get identical synthetic patients.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseGenerator:
        """
        Fit the generator on an engineered DataFrame.
        Learns the statistical distribution of each feature per stage.
        Returns self for chaining.
        """
        ...

    @abstractmethod
    def generate(self, n: int) -> pd.DataFrame:
        """
        Generate n synthetic patient records.
        Returns a DataFrame with the same schema as the fitted data.
        Raises RuntimeError if called before fit().
        """
        ...

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling "
                "generate(). Call fit() first."
            )


class BaseDataSuite(ABC):
    """
    The top-level data contract that Benchmark receives.

    A suite is the public API of the data layer. It wires together
    a preprocessor, feature engineer, and generator, and exposes
    a clean list of PatientRecord objects ready for evaluation.

    Example:
        ```python
        suite = MyClinicalSuite(config=SuiteConfig(task=Task.STAGING))
        records = suite.load()
        ```

    The suite handles train/test splitting internally.
    Benchmark always receives the test split only.
    """

    def __init__(self, config: SuiteConfig | None = None) -> None:
        self.config = config or SuiteConfig()

    @abstractmethod
    def load(self) -> list[PatientRecord]:
        """
        Run the full data pipeline and return test-split PatientRecords.

        Pipeline order:
            1. Load raw source data
            2. Preprocess (encode, impute, scale)
            3. Engineer features (domain-specific derivations)
            4. Generate synthetic records (if n_synthetic > 0)
            5. Merge real + synthetic
            6. Split → return test split as PatientRecord list
        """
        ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """
        Return a summary of the suite configuration and data statistics.
        Used by results.report() to document what was evaluated.

        Should include at minimum:
            - domain name
            - feature set (full/reduced)
            - task type
            - n_real records
            - n_synthetic records
            - label distribution
            - seed
        """
        ...

    @property
    def domain(self) -> str:
        """Human-readable domain name. e.g. 'CKD', 'Hypertension'"""
        return self.__class__.__name__.replace("Suite", "")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"features={self.config.features.value!r}, "
            f"task={self.config.task.value!r}, "
            f"seed={self.config.seed}, "
            f"n_synthetic={self.config.n_synthetic})"
        )
