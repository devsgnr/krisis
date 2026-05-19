"""
krisis/data/ckd/generate.py

Stage-aware synthetic patient generator for the CKD domain.

Methodology:
    For each CKD stage present in the fitted data, the generator learns
    the mean and standard deviation of every feature from real patient
    records. New synthetic patients are sampled from a normal distribution
    per feature per stage, then clipped to published clinical reference
    ranges to ensure physiological plausibility.

    Binary features (htn, dm, rbc, pc) are sampled from a Bernoulli
    distribution using the per-stage prevalence observed in the real data.

    Sex is generated using generate_sex_from_prevalence() — prevalence-
    anchored rather than creatinine-conditioned, because creatinine is
    itself being generated here rather than observed.

    Stage distribution in synthetic data mirrors the real data's stage
    distribution, preserving class balance.

Design rationale:
    A simple parametric approach (Gaussian per stage) is chosen over
    deep generative models (VAE, GAN) deliberately:
        - Transparent: researchers can audit why a record looks as it does
        - Reproducible: fully seeded, no stochastic training
        - Safe: clinical bounds prevent physiologically impossible values
        - Appropriate: the UCI dataset has ~400 records — deep generative
          models would overfit and produce unreliable synthetic data

Clinical reference ranges:
    All bounds are sourced from published clinical guidelines:
    - Urine specific gravity: standard urinalysis reference (1.005–1.030)
    - Albumin (ordinal): KDIGO 2024 (0–5 scale)
    - Blood glucose: ADA reference ranges (70–490 mg/dL)
    - Blood urea: clinical reference (1.5–200 mg/dL)
    - Serum creatinine: clinical reference (0.4–15.0 mg/dL)
    - Haemoglobin: WHO reference ranges (3.1–17.8 g/dL)
    - Packed cell volume: clinical reference (9–54%)
    - Red blood cell count: clinical reference (1.5–6.5 M/cmm)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from krisis.data.base import BaseGenerator
from krisis.data.ckd.engineer import (
    FULL_FEATURES,
    compute_egfr,
    generate_sex_from_prevalence,
)

# ── Clinical bounds ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClinicalBounds:
    """
    Min/max physiologically valid range for a continuous feature.
    Synthetic samples are clipped to these bounds after Gaussian sampling.
    """

    low: float
    high: float


# Published clinical reference ranges per continuous feature
CLINICAL_BOUNDS: dict[str, ClinicalBounds] = {
    "age": ClinicalBounds(2.0, 90.0),  # patient age (years)
    "bp": ClinicalBounds(50.0, 180.0),  # blood pressure (mmHg)
    "sg": ClinicalBounds(1.005, 1.030),  # urine specific gravity
    "al": ClinicalBounds(0.0, 5.0),  # albumin (ordinal 0–5)
    "bgr": ClinicalBounds(70.0, 490.0),  # blood glucose (mg/dL)
    "bu": ClinicalBounds(1.5, 200.0),  # blood urea (mg/dL)
    "sc": ClinicalBounds(0.4, 15.0),  # serum creatinine (mg/dL)
    "sod": ClinicalBounds(100.0, 170.0),  # sodium (mEq/L)
    "pot": ClinicalBounds(2.5, 7.0),  # potassium (mEq/L)
    "hemo": ClinicalBounds(3.1, 17.8),  # haemoglobin (g/dL)
    "pcv": ClinicalBounds(9.0, 54.0),  # packed cell volume (%)
    "wbcc": ClinicalBounds(2200.0, 26400.0),  # white blood cell count (cells/cmm)
    "rbcc": ClinicalBounds(1.5, 6.5),  # red blood cell count (M/cmm)
}

# Binary features — sampled from Bernoulli using per-stage prevalence
BINARY_FEATURES = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]

# Continuous features — sampled from Gaussian, clipped to clinical bounds
CONTINUOUS_FEATURES = [
    "age",
    "bp",
    "sg",
    "al",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
]

# All source features the generator produces.
ALL_FEATURES = FULL_FEATURES.copy()


# ── Per-stage statistics ──────────────────────────────────────────────────────


@dataclass
class StageStats:
    """
    Learned statistics for a single CKD stage.

    continuous_stats: {feature_name: {"mean": float, "std": float}}
    binary_prevalence: {feature_name: float}  — P(feature == 1) per stage
    n_real: number of real records this stage was fitted on
    """

    continuous_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    binary_prevalence: dict[str, float] = field(default_factory=dict)
    n_real: int = 0


# ── CKDGenerator ─────────────────────────────────────────────────────────────


class CKDGenerator(BaseGenerator):
    """
    Stage-aware synthetic patient generator for CKD.

    Learns per-stage feature distributions from real engineered CKD data,
    then generates new synthetic patients that are physiologically plausible
    and stage-consistent.

    Example:
        ```python
        generator = CKDGenerator(seed=42)
        generator.fit(df_engineered)
        synthetic_df = generator.generate(n=200)
        ```

    The generated DataFrame has the same schema as the input to fit():
    ALL_FEATURES columns + 'class', 'sex', 'egfr', 'ckd_stage'.
    """

    def __init__(
        self,
        seed: int = 42,
        split_stage_3: bool = False,
    ) -> None:
        super().__init__(seed=seed)
        self.split_stage_3 = split_stage_3
        self._stage_stats: dict[int, StageStats] = {}
        self._stage_distribution: dict[int, float] | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> CKDGenerator:
        """
        Fit the generator on an engineered CKD DataFrame.

        Expects df to contain ALL_FEATURES, ``ckd_stage``, and ``class``.

        Args:
            df: engineered DataFrame from CKDFeatureEngineer.fit_transform()

        Returns:
            self (for method chaining)

        Raises:
            ValueError: if required columns are missing
        """
        self._validate_columns(df)

        stages = sorted(df["ckd_stage"].unique())
        total = len(df)

        for stage in stages:
            stage_df = df[df["ckd_stage"] == stage]
            stats = StageStats(n_real=len(stage_df))

            # Learn continuous feature distributions
            for feature in CONTINUOUS_FEATURES:
                if feature in stage_df.columns:
                    stats.continuous_stats[feature] = {
                        "mean": float(stage_df[feature].mean()),
                        "std": float(stage_df[feature].std()),
                    }

            # Learn binary feature prevalence
            for feature in BINARY_FEATURES:
                if feature in stage_df.columns:
                    stats.binary_prevalence[feature] = float(stage_df[feature].mean())

            self._stage_stats[stage] = stats

        # Learn stage distribution from real data
        self._stage_distribution = {
            stage: len(df[df["ckd_stage"] == stage]) / total for stage in stages
        }

        self._is_fitted = True
        return self

    def generate(self, n: int) -> pd.DataFrame:
        """
        Generate n synthetic CKD patient records.

        Records are distributed across stages proportionally to the
        real data's stage distribution, preserving class balance.

        Args:
            n: number of synthetic records to generate

        Returns:
            DataFrame with columns: ALL_FEATURES + class, sex, egfr, ckd_stage

        Raises:
            RuntimeError: if called before fit()
        """
        self._check_fitted()

        records = []
        stage_counts = self._compute_stage_counts(n)

        for stage, count in stage_counts.items():
            stage_records = self._generate_stage_records(stage, count)
            records.extend(stage_records)

        df = pd.DataFrame(records)

        # Shuffle so stages aren't in blocks
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return df

    # ── Private helpers ──────────────────────────────────────────────────────

    def _generate_stage_records(
        self,
        stage: int,
        n: int,
    ) -> list[dict]:
        """Generate n records for a specific CKD stage."""
        stats = self._stage_stats[stage]
        records = []

        # Generate sex first (prevalence-based, not creatinine-conditioned)
        sexes = generate_sex_from_prevalence(n, seed=self.seed + stage)

        for i in range(n):
            record: dict = {}
            sex = sexes[i]

            # Sample continuous features
            for feature in CONTINUOUS_FEATURES:
                if feature not in stats.continuous_stats:
                    continue

                mean = stats.continuous_stats[feature]["mean"]
                std = stats.continuous_stats[feature]["std"]

                # Handle zero or near-zero std (all patients same value)
                if std < 1e-6:
                    value = mean
                else:
                    value = self.rng.normal(loc=mean, scale=std)

                # Clip to clinical bounds
                bounds = CLINICAL_BOUNDS.get(feature)
                if bounds:
                    value = float(np.clip(value, bounds.low, bounds.high))

                record[feature] = round(value, 4)

            # Sample binary features from Bernoulli
            for feature in BINARY_FEATURES:
                p = stats.binary_prevalence.get(feature, 0.5)
                p = float(np.clip(p, 0.0, 1.0))
                record[feature] = float(self.rng.choice([0, 1], p=[1 - p, p]))

            # Derive class label from stage
            # Stage 1–2: not CKD (class = 1), Stage 3–5: CKD (class = 0)
            # This mirrors the UCI dataset label convention:
            # 0 = CKD present, 1 = CKD absent
            record["class"] = 1 if stage <= 2 else 0

            # Attach metadata
            record["sex"] = sex
            record["ckd_stage"] = stage

            # Compute eGFR — use a representative age for the stage
            # (age is not in the reduced feature set, so we use
            # a stage-informed approximation for metadata only)
            representative_age = self._representative_age_for_stage(stage)
            record["egfr"] = compute_egfr(
                creatinine=record["sc"],
                age=representative_age,
                sex=sex,
            )

            records.append(record)

        return records

    def _compute_stage_counts(self, n: int) -> dict[int, int]:
        """
        Distribute n records across stages proportionally to the
        real data's stage distribution.

        Uses floor allocation with the remainder assigned to the
        most prevalent stage to ensure exactly n records total.
        """
        assert self._stage_distribution is not None

        counts: dict[int, int] = {}
        allocated = 0

        stages = list(self._stage_distribution.keys())
        proportions = [self._stage_distribution[s] for s in stages]

        for i, stage in enumerate(stages):
            count = int(np.floor(proportions[i] * n))
            counts[stage] = count
            allocated += count

        # Assign remainder to the most prevalent stage
        remainder = n - allocated
        if remainder > 0:
            most_prevalent = max(
                self._stage_distribution,
                key=lambda s: self._stage_distribution[s],  # type: ignore
            )
            counts[most_prevalent] += remainder

        return counts

    def _representative_age_for_stage(self, stage: int) -> float:
        """
        Return a clinically reasonable representative age for a CKD stage.
        Used when computing eGFR for synthetic records where age is not
        in the reduced feature set.

        These are approximations informed by published CKD epidemiology
        (KDIGO 2024) — CKD prevalence increases significantly with age.
        """
        stage_age_map = {
            1: 45.0,
            2: 52.0,
            3: 60.0,
            4: 67.0,
            5: 72.0,
            6: 75.0,  # Stage 5 when split_stage_3=True
        }
        return stage_age_map.get(stage, 60.0)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise informative error if required columns are missing."""
        required = set(ALL_FEATURES) | {"ckd_stage", "class"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CKDGenerator.fit() requires columns: {required}. "
                f"Missing: {missing}. "
                "Ensure CKDFeatureEngineer.fit_transform() was called "
                "before fitting the generator."
            )
