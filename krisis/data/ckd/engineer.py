"""
krisis/data/ckd/engineer.py

Feature engineering for the CKD domain.

Derives three new clinically meaningful features from preprocessed data:

    sex       — synthetically generated from serum creatinine distribution,
                conditioned on published CKD epidemiology (55% male / 45%
                female, KDIGO 2024). See generate_sex_from_creatinine() for
                full methodology and assumptions.

    egfr      — estimated Glomerular Filtration Rate, computed using the
                CKD-EPI 2021 (race-free) equation from serum creatinine,
                age, and sex. This is the clinical gold standard for
                assessing kidney function.

    ckd_stage — CKD stage (1–5) derived from eGFR using KDIGO 2024
                staging thresholds. Transforms the dataset from binary
                classification to a clinically meaningful multiclass task.

Clinical sources:
    - KDIGO 2024 CKD Guidelines (staging thresholds)
    - CKD-EPI 2021 Collaboration (race-free eGFR equation)
    - Global Burden of Disease 2019 (sex prevalence in CKD cohorts)

Documented assumption:
    The UCI CKD dataset does not include patient sex. Sex is synthetically
    generated using a creatinine-conditioned probabilistic model. This is
    a deliberate, documented assumption — not a data quality issue.
    Users with sex-labelled datasets can bypass this via the
    sex_column parameter on CKDSuite.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from krisis.data.base import BaseFeatureEngineer, FeatureSet

# ── Constants ─────────────────────────────────────────────────────────────────

# CKD-EPI 2021 equation constants
# Source: Inker et al., NEJM 2021
CKD_EPI_KAPPA_FEMALE = 0.7
CKD_EPI_KAPPA_MALE = 0.9
CKD_EPI_ALPHA_FEMALE = -0.241
CKD_EPI_ALPHA_MALE = -0.302
CKD_EPI_AGE_FACTOR = 0.9938
CKD_EPI_BASE = 142
CKD_EPI_SEX_FACTOR_FEMALE = 1.012  # multiplier applied for female patients
MIN_SERUM_CREATININE = 0.4
MIN_AGE = 0.0

# KDIGO 2024 eGFR staging thresholds (mL/min/1.73m²)
# Source: KDIGO 2024 CKD Clinical Practice Guidelines
KDIGO_THRESHOLDS = {
    1: (90, float("inf")),  # Stage 1 — normal or high
    2: (60, 90),  # Stage 2 — mildly decreased
    3: (30, 60),  # Stage 3 — mildly to moderately decreased
    #   Stage 3 can be split into 3a (45–59) and 3b (30–44).
    #   We use the combined Stage 3 range here for simplicity.
    #   Set SPLIT_STAGE_3 = True below to enable 3a/3b splitting.
    4: (15, 30),  # Stage 4 — severely decreased
    5: (0, 15),  # Stage 5 — kidney failure
}

# Set to True to split Stage 3 into 3a and 3b
# When True, stage values will be: 1, 2, 3 (3a), 4 (3b), 5, 6
# When False (default), stage values are: 1, 2, 3, 4, 5
SPLIT_STAGE_3 = False

# Sex generation — population prevalence in CKD cohorts
# Source: Global Burden of Disease 2019, KDIGO 2024
P_MALE_CKD = 0.55
P_FEMALE_CKD = 0.45

# Creatinine thresholds for sex inference
# Based on CKD-EPI kappa values: 0.7 (female), 0.9 (male)
SC_FEMALE_UPPER = 0.7  # sc <= 0.7 → likely female
SC_AMBIGUOUS_UPPER = 0.9  # 0.7 < sc <= 0.9 → ambiguous
# sc > 0.9 → likely male

# Feature names exposed per feature set
REDUCED_FEATURES = [
    "htn",
    "dm",
    "sg",
    "hemo",
    "pcv",
    "rbcc",
    "rbc",
    "pc",
    "al",
    "bgr",
    "sc",
    "bu",
]

FULL_FEATURES = [
    "age",
    "bp",
    "al",
    "bgr",
    "bu",
    "sc",
    "sg",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
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

# Engineered features added by this module
ENGINEERED_FEATURES = ["sex", "egfr", "ckd_stage"]


# ── eGFR computation ──────────────────────────────────────────────────────────


def compute_egfr(
    creatinine: float,
    age: float,
    sex: str,
) -> float:
    """
    Compute estimated Glomerular Filtration Rate using the CKD-EPI 2021
    race-free equation.

    Args:
        creatinine: Serum creatinine in mg/dL (sc column)
        age: Patient age in years
        sex: 'male' or 'female'

    Returns:
        eGFR in mL/min/1.73m²

    Source:
        Inker LA et al. New Creatinine- and Cystatin C–Based Equations to
        Estimate GFR without Race. NEJM 2021; 385:1737-1749.
    """
    creatinine = _bounded_float(
        creatinine,
        lower=MIN_SERUM_CREATININE,
        fallback=MIN_SERUM_CREATININE,
    )
    age = _bounded_float(age, lower=MIN_AGE, fallback=MIN_AGE)

    if sex == "female":
        kappa = CKD_EPI_KAPPA_FEMALE
        alpha = CKD_EPI_ALPHA_FEMALE
    else:
        kappa = CKD_EPI_KAPPA_MALE
        alpha = CKD_EPI_ALPHA_MALE

    ratio = creatinine / kappa

    if ratio < 1:
        egfr = CKD_EPI_BASE * (ratio**alpha) * (CKD_EPI_AGE_FACTOR**age)
    else:
        egfr = CKD_EPI_BASE * (ratio**-1.200) * (CKD_EPI_AGE_FACTOR**age)

    if sex == "female":
        egfr *= CKD_EPI_SEX_FACTOR_FEMALE

    return round(egfr, 2)


def _bounded_float(value: float, *, lower: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric):
        return fallback
    return max(numeric, lower)


def assign_ckd_stage(egfr: float, split_stage_3: bool = SPLIT_STAGE_3) -> int:
    """
    Assign CKD stage from eGFR using KDIGO 2024 thresholds.

    Args:
        egfr: eGFR value in mL/min/1.73m²
        split_stage_3: if True, splits Stage 3 into 3a (returned as 3)
                       and 3b (returned as 4), shifting Stage 4 → 5
                       and Stage 5 → 6.

    Returns:
        Integer CKD stage.
        Default: 1, 2, 3, 4, 5
        With split_stage_3=True: 1, 2, 3 (3a), 4 (3b), 5, 6
    """
    if split_stage_3:
        if egfr >= 90:
            return 1
        if egfr >= 60:
            return 2
        if egfr >= 45:
            return 3
        if egfr >= 30:
            return 4
        if egfr >= 15:
            return 5
        return 6

    if egfr >= 90:
        return 1
    if egfr >= 60:
        return 2
    if egfr >= 30:
        return 3
    if egfr >= 15:
        return 4
    return 5


# ── Sex generation ────────────────────────────────────────────────────────────


def generate_sex_from_creatinine(
    sc: pd.Series,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate sex probabilistically, conditioned on serum creatinine.

    Methodology:
        Women physiologically have lower serum creatinine than men at
        equivalent kidney function due to lower muscle mass. The CKD-EPI
        equation accounts for this with different kappa thresholds
        (0.7 female / 0.9 male). We use these thresholds to condition
        sex assignment probabilities on the observed creatinine value:

            sc <= 0.7  → P(female) = 0.75, P(male) = 0.25
            sc <= 0.9  → P(female) = 0.50, P(male) = 0.50  (ambiguous)
            sc >  0.9  → P(female) = 0.25, P(male) = 0.75

        The overall distribution converges to approximately 45% female /
        55% male, consistent with published CKD epidemiology.

    Args:
        sc: Series of serum creatinine values (mg/dL)
        seed: random seed for reproducibility

    Returns:
        numpy array of 'male' / 'female' strings, same length as sc

    Note:
        This is a documented assumption. The UCI CKD dataset does not
        include patient sex. See docs/datasets/ckd.md for full disclosure.
    """
    rng = np.random.default_rng(seed)
    result = []

    for creatinine in sc:
        if creatinine <= SC_FEMALE_UPPER:
            p_female = 0.75
        elif creatinine <= SC_AMBIGUOUS_UPPER:
            p_female = 0.50
        else:
            p_female = 0.25

        result.append(rng.choice(["female", "male"], p=[p_female, 1.0 - p_female]))

    return np.array(result)


def generate_sex_from_prevalence(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate sex from population prevalence only, without conditioning
    on creatinine. Used by the synthetic patient generator when generating
    new records from scratch (not transforming existing UCI records).

    Args:
        n: number of records to generate
        seed: random seed for reproducibility

    Returns:
        numpy array of 'male' / 'female' strings of length n
    """
    rng = np.random.default_rng(seed)
    return rng.choice(
        ["male", "female"],
        size=n,
        p=[P_MALE_CKD, P_FEMALE_CKD],
    )


# ── CKDFeatureEngineer ────────────────────────────────────────────────────────


class CKDFeatureEngineer(BaseFeatureEngineer):
    """
    Derives eGFR, sex, and CKD stage from preprocessed CKD data.

    The engineer expects a DataFrame that has already passed through
    CKDPreprocessor.fit_transform() — i.e. clean, imputed, scaled data
    with at minimum the columns: sc, age (for eGFR computation).

    Usage:
        engineer = CKDFeatureEngineer(seed=42, split_stage_3=False)
        df_engineered = engineer.fit_transform(df_preprocessed)
        # df_engineered now has additional columns: sex, egfr, ckd_stage

    Note:
        fit_transform and transform are identical here — the engineer
        is stateless (no fitting required). Both methods are provided
        to maintain interface consistency with BaseFeatureEngineer.
    """

    def __init__(
        self,
        seed: int = 42,
        split_stage_3: bool = SPLIT_STAGE_3,
        sex_column: str | None = None,
    ) -> None:
        """
        Args:
            seed: random seed passed to sex generation
            split_stage_3: whether to split Stage 3 into 3a/3b
            sex_column: if the dataset already contains a sex column,
                        provide its name here and sex generation will
                        be skipped. Column must contain 'male'/'female'.
        """
        super().__init__(seed=seed)
        self.split_stage_3 = split_stage_3
        self.sex_column = sex_column

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer eGFR, sex, and CKD stage features.

        Args:
            df: preprocessed CKD DataFrame (output of CKDPreprocessor)

        Returns:
            df with three additional columns: sex, egfr, ckd_stage
        """
        return self._engineer(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identical to fit_transform — engineer is stateless."""
        return self._engineer(df)

    def get_feature_names(self, feature_set: FeatureSet) -> list[str]:
        """
        Return feature column names for the given feature set,
        including the engineered features.
        """
        base = REDUCED_FEATURES if feature_set == FeatureSet.REDUCED else FULL_FEATURES
        return base + ENGINEERED_FEATURES

    # ── Private ──────────────────────────────────────────────────────────────

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        self._validate_columns(df)

        # Step 1 — generate or use existing sex column
        if self.sex_column and self.sex_column in df.columns:
            df["sex"] = df[self.sex_column].values
        else:
            df["sex"] = generate_sex_from_creatinine(df["sc"], seed=self.seed)

        # Step 2 — compute eGFR row-wise
        df["egfr"] = df.apply(
            lambda row: compute_egfr(
                creatinine=row["sc"],
                age=row["age"],
                sex=row["sex"],
            ),
            axis=1,
        )

        # Step 3 — derive CKD stage from eGFR
        df["ckd_stage"] = df["egfr"].apply(
            lambda e: assign_ckd_stage(e, split_stage_3=self.split_stage_3)
        )

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise informative errors if required columns are missing."""
        required = {"sc", "age"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CKDFeatureEngineer requires columns {required}. "
                f"Missing: {missing}. "
                "Ensure CKDPreprocessor.fit_transform() was called first "
                "and that FeatureSet.FULL is used (age is not in the "
                "REDUCED feature set — pass the full imputed DataFrame "
                "to the engineer before applying feature selection)."
            )
