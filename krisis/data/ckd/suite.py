"""
krisis/data/ckd/suite.py

CKDSuite is the public API of the CKD data domain.

It wires together CKDPreprocessor, CKDFeatureEngineer, and CKDGenerator
into a single pipeline that produces a list of PatientRecord objects
ready for evaluation by the Benchmark harness.

Usage:
    from krisis.data.ckd.suite import CKDSuite
    from krisis.data.base import FeatureSet, Task, SuiteConfig

    suite = CKDSuite(
        config=SuiteConfig(
            features=FeatureSet.REDUCED,
            task=Task.STAGING,
            seed=42,
            n_synthetic=200,
            test_size=0.2,
        ),
        data_path="datasets/ckd/ckd_full.csv",
    )

    records = suite.load()
    # records → List[PatientRecord]

Pipeline order (internal):
    1. Load raw CSV from data_path
    2. CKDPreprocessor.fit_transform() → fit encoders/imputer/scaler
    3. CKDPreprocessor.get_imputed_dataframe() → clean clinical-unit DataFrame
    4. CKDFeatureEngineer.fit_transform() → adds sex, egfr, ckd_stage
    5. Train/test split (stratified by ckd_stage for STAGING/PROGRESSION,
       stratified by class for DETECTION)
    6. CKDGenerator.fit(train split) → learns stage distributions
    7. CKDGenerator.generate(n_synthetic) → synthetic records
    8. Merge test split + synthetic records
    9. Build PatientRecord list from merged DataFrame

Synthetic records in the test split:
    Synthetic records are merged into the test split intentionally.
    The benchmark evaluates models on both real held-out records and
    synthetic ones. This increases evaluation set size and stress-tests
    models on edge cases that may be underrepresented in the small
    UCI dataset (400 records). The describe() output reports real vs
    synthetic counts separately for full transparency.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from krisis.data.base import (
    BaseDataSuite,
    FeatureSet,
    PatientRecord,
    SuiteConfig,
    Task,
)
from krisis.data.ckd.engineer import CKDFeatureEngineer
from krisis.data.ckd.generate import CKDGenerator
from krisis.data.ckd.preprocess import FINAL_COLUMN_ORDER, CKDPreprocessor
from krisis.data.ckd.validate import validate_ckd_csv

# Default path — relative to the project root
DEFAULT_DATA_PATH = os.path.join("datasets", "ckd", "ckd_full.csv")


class CKDSuite(BaseDataSuite):
    """
    Full CKD data pipeline — from raw CSV to PatientRecord list.

    Args:
        config: SuiteConfig controlling features, task, seed,
                n_synthetic, and test_size. Defaults to SuiteConfig()
                which uses REDUCED features, DETECTION task, seed=42,
                200 synthetic records, 20% test split.

        data_path: path to the raw UCI CKD CSV file.
                   Defaults to 'datasets/ckd/ckd_full.csv'.

        sex_column: if your dataset already contains a sex column,
                    pass its name here and sex generation is skipped.
                    Column must contain 'male' / 'female' strings.

        scaler_path: optional path to persist the fitted MinMaxScaler
                     as a .pkl file for reuse outside of Krisis.

        split_stage_3: whether to split CKD Stage 3 into 3a and 3b.
                       Default False. When True, stage values are
                       1, 2, 3 (3a), 4 (3b), 5, 6.
    """

    def __init__(
        self,
        config: SuiteConfig | None = None,
        data_path: str = DEFAULT_DATA_PATH,
        sex_column: str | None = None,
        scaler_path: str | None = None,
        split_stage_3: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.data_path = data_path
        self.sex_column = sex_column
        self.scaler_path = scaler_path
        self.split_stage_3 = split_stage_3

        # Internal state — populated during load()
        self._n_real: int = 0
        self._n_synthetic: int = 0
        self._n_should_abstain: int = 0
        self._progression_distribution: dict[str, int] = {}
        self._label_distribution: dict[str, int] = {}
        self._stage_distribution: dict[int, int] = {}
        self._raw_row_count: int = 0
        self._raw_missing_values: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> list[PatientRecord]:
        """
        Run the full CKD pipeline and return PatientRecord list.

        Returns:
            List of PatientRecord objects for the test split,
            including both real held-out records and synthetic records.

        Raises:
            FileNotFoundError: if data_path does not exist
            ValueError: if the dataset is missing required columns
        """
        # Step 1 — load raw data
        raw_df = self._load_raw()

        # Step 2 — preprocess
        preprocessor = CKDPreprocessor(
            feature_set=self.config.features,
            seed=self.config.seed,
            scaler_path=self.scaler_path,
        )
        preprocessor.fit_transform(raw_df)
        clinical_df = preprocessor.get_imputed_dataframe()

        # Step 3 — feature engineering
        # Engineer runs on the full unscaled clinical DataFrame. It needs age
        # and serum creatinine in real units for CKD-EPI eGFR computation.
        # For REDUCED feature set, age is used only for metadata derivation.
        engineer = CKDFeatureEngineer(
            seed=self.config.seed,
            split_stage_3=self.split_stage_3,
            sex_column=self.sex_column,
        )
        engineered_df = engineer.fit_transform(clinical_df)

        # Step 4 — train/test split
        stratify_col = self._get_stratify_column()
        train_df, test_df = train_test_split(
            engineered_df,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=engineered_df[stratify_col],
        )

        self._n_real = len(test_df)

        # Step 5 — fit generator on train split
        generator = CKDGenerator(
            seed=self.config.seed,
            split_stage_3=self.split_stage_3,
        )
        generator.fit(train_df)

        # Step 6 — generate synthetic records
        synthetic_df = pd.DataFrame()
        if self.config.n_synthetic > 0:
            synthetic_df = generator.generate(self.config.n_synthetic)
            self._n_synthetic = len(synthetic_df)

        # Step 7 — merge test split + synthetic
        if not synthetic_df.empty:
            eval_df = pd.concat([test_df, synthetic_df], ignore_index=True)
            eval_df = eval_df.sample(frac=1, random_state=self.config.seed).reset_index(
                drop=True
            )
        else:
            eval_df = test_df.copy()

        # Step 8 — record distributions for describe()
        self._label_distribution = eval_df["class"].value_counts().to_dict()
        self._stage_distribution = eval_df["ckd_stage"].value_counts().to_dict()

        # Step 9 — build PatientRecord list
        return self._build_records(eval_df)

    def describe(self) -> dict[str, Any]:
        """
        Return a summary of suite configuration and data statistics.
        Called by results.report() to document what was evaluated.
        """
        return {
            "domain": "Chronic Kidney Disease (CKD)",
            "source": "UCI ML Repository — CKD Dataset (400 records)",
            "data_path": self.data_path,
            "n_raw_records": self._raw_row_count,
            "n_raw_missing_values": self._raw_missing_values,
            "feature_set": self.config.features.value,
            "task": self.config.task.value,
            "seed": self.config.seed,
            "test_size": self.config.test_size,
            "n_real_test_records": self._n_real,
            "n_synthetic_records": self._n_synthetic,
            "n_total_eval_records": self._n_real + self._n_synthetic,
            "n_should_abstain_records": self._n_should_abstain,
            "progression_distribution": self._progression_distribution,
            "label_distribution": {
                "ckd_present (class=0)": self._label_distribution.get(0, 0),
                "ckd_absent  (class=1)": self._label_distribution.get(1, 0),
            },
            "stage_distribution": {
                f"stage_{k}": v for k, v in sorted(self._stage_distribution.items())
            },
            "split_stage_3": self.split_stage_3,
            "sex_generation": (
                f"from column '{self.sex_column}'"
                if self.sex_column
                else "synthetic (creatinine-conditioned, KDIGO 2024)"
            ),
            "clinical_sources": [
                "KDIGO 2024 CKD Clinical Practice Guidelines",
                "CKD-EPI 2021 (Inker et al., NEJM 2021)",
                "Global Burden of Disease 2019",
                "UCI ML Repository — CKD Dataset",
            ],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_raw(self) -> pd.DataFrame:
        """Load raw CSV from data_path with informative error on failure."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"CKD dataset not found at '{self.data_path}'. "
                "Krisis does not bundle the UCI CKD data in the Python package. "
                "Download ckd_full.csv locally, then pass its path with "
                "CKDSuite(data_path='...') or the example --data-path option."
            )
        raw_df = pd.read_csv(self.data_path)
        validation = validate_ckd_csv(raw_df, data_path=self.data_path)
        self._raw_row_count = validation.n_rows
        self._raw_missing_values = validation.n_missing_values
        return validation.dataframe

    def _get_stratify_column(self) -> str:
        """
        Return the column to stratify the train/test split on.

        DETECTION   → stratify by 'class' (preserve CKD/not CKD balance)
        STAGING     → stratify by 'ckd_stage' (preserve stage distribution)
        PROGRESSION → stratify by 'ckd_stage' (preserve stage distribution)
        """
        if self.config.task == Task.DETECTION:
            return "class"
        return "ckd_stage"

    def _get_feature_columns(self) -> list[str]:
        """
        Return the feature columns to include in PatientRecord.features.

        REDUCED → 12 validated features (no engineered metadata)
        FULL    → all available columns except class, sex, egfr, ckd_stage
        """
        if self.config.features == FeatureSet.REDUCED:
            return FINAL_COLUMN_ORDER
        else:
            # FULL: all columns except metadata and target
            exclude = {"class", "sex", "egfr", "ckd_stage"}
            return [
                c
                for c in CKDFeatureEngineer().get_feature_names(FeatureSet.FULL)
                if c not in exclude
            ]

    def _get_label(self, row: pd.Series) -> int | str:
        """
        Return the ground truth label for a record based on the task.

        DETECTION   → int: 0 (CKD present) or 1 (CKD absent)
        STAGING     → int: CKD stage (1–5, or 1–6 if split_stage_3)
        PROGRESSION → str: 'stable', 'worsening', 'improving'
        """
        if self.config.task == Task.DETECTION:
            return int(row["class"])
        elif self.config.task == Task.STAGING:
            return int(row["ckd_stage"])
        return str(row["progression_direction"])

    def _build_records(self, df: pd.DataFrame) -> list[PatientRecord]:
        """
        Convert the evaluation DataFrame into a list of PatientRecord objects.

        Features go into PatientRecord.features (what the model sees).
        Engineered metadata (sex, egfr, ckd_stage) goes into
        PatientRecord.metadata (what Krisis uses to score the model).
        """
        feature_cols = self._get_feature_columns()
        records = []

        for i, (_, row) in enumerate(df.iterrows()):
            progression_metadata: dict[str, Any] = {}
            if self.config.task == Task.PROGRESSION:
                progression = self._build_progression_record(row, feature_cols, i)
                features = progression["features"]
                progression_metadata = progression["metadata"]
                row = row.copy()
                row["progression_direction"] = progression["label"]
            else:
                features = {col: row[col] for col in feature_cols if col in row}
                if self.config.task == Task.STAGING:
                    egfr = float(row.get("egfr"))
                    nearest, margin = self._nearest_egfr_threshold(egfr)
                    features["egfr"] = egfr
                    features["nearest_egfr_stage_threshold"] = nearest
                    features["egfr_threshold_margin"] = margin

            metadata: dict[str, Any] = {
                "egfr": row.get("egfr"),
                "ckd_stage": row.get("ckd_stage"),
                "sex": row.get("sex"),
                "class": row.get("class"),
            }
            metadata.update(self._deferral_metadata(row))
            metadata.update(progression_metadata)

            label = self._get_label(row)

            records.append(
                PatientRecord(
                    features=features,
                    label=label,
                    metadata=metadata,
                )
            )

        self._n_should_abstain = sum(
            1 for record in records if record.metadata.get("should_abstain")
        )
        self._progression_distribution = {}
        if self.config.task == Task.PROGRESSION:
            for record in records:
                label = str(record.label)
                self._progression_distribution[label] = (
                    self._progression_distribution.get(label, 0) + 1
                )
        return records

    def _build_progression_record(
        self,
        row: pd.Series,
        feature_cols: list[str],
        index: int,
    ) -> dict[str, Any]:
        direction = self._progression_direction(row, index)
        current = {col: float(row[col]) for col in feature_cols if col in row}
        is_ambiguous = self._is_ambiguous_progression(row, index)

        if is_ambiguous:
            label = direction
            baseline = self._ambiguous_baseline_from_current(current, index)
            metadata = {
                "should_abstain": True,
                "progression_ambiguous": True,
                "deferral_reason": "ambiguous_progression_trajectory",
            }
        else:
            label = direction
            baseline = self._baseline_from_current(current, direction)
            metadata = {
                "progression_ambiguous": False,
            }

        features: dict[str, Any] = {
            "trajectory_months": 6,
            "baseline": baseline,
            "current": current,
        }
        return {"features": features, "label": label, "metadata": metadata}

    def _progression_direction(self, row: pd.Series, index: int) -> str:
        """
        Derive a reproducible synthetic progression label.

        UCI CKD is cross-sectional, so Krisis creates a plausible six-month
        trajectory label without claiming longitudinal source data.
        """
        stage = int(row["ckd_stage"])
        htn = float(row.get("htn", 0.0))
        dm = float(row.get("dm", 0.0))
        albumin = float(row.get("al", 0.0))
        hemo = float(row.get("hemo", 12.0))
        risk = stage + htn + dm + min(albumin, 5.0) / 2.0

        bucket = (self.config.seed + index * 17 + stage * 13) % 10
        if risk >= 5.5 or (stage >= 3 and bucket < 6):
            return "worsening"
        if stage <= 2 and htn < 0.5 and dm < 0.5 and albumin <= 1 and bucket < 3:
            return "stable"
        if hemo >= 11.5 and bucket in {7, 8, 9}:
            return "improving"
        return "stable"

    def _is_ambiguous_progression(self, row: pd.Series, index: int) -> bool:
        """
        Select progression cases where a cautious model should abstain.

        The source data are cross-sectional, so hard progression cases simulate
        ambiguous follow-up: tiny changes, mixed markers, and borderline stages.
        """
        stage = int(row["ckd_stage"])
        egfr = float(row["egfr"])
        bucket = (self.config.seed + index * 19 + stage * 7) % 10
        near_stage_threshold = any(
            abs(egfr - threshold) <= 5.0 for threshold in (15.0, 30.0, 60.0, 90.0)
        )
        return bucket < 3 or near_stage_threshold

    def _baseline_from_current(
        self,
        current: dict[str, float],
        direction: str,
    ) -> dict[str, float]:
        baseline = current.copy()

        if direction == "worsening":
            baseline["sc"] = max(0.4, current["sc"] * 0.86)
            baseline["bu"] = max(1.5, current["bu"] * 0.88)
            baseline["al"] = max(0.0, current["al"] - 1.0)
            baseline["hemo"] = min(17.8, current["hemo"] + 0.7)
            baseline["pcv"] = min(54.0, current["pcv"] + 2.0)
        elif direction == "improving":
            baseline["sc"] = min(15.0, current["sc"] * 1.16)
            baseline["bu"] = min(200.0, current["bu"] * 1.12)
            baseline["al"] = min(5.0, current["al"] + 1.0)
            baseline["hemo"] = max(3.1, current["hemo"] - 0.6)
            baseline["pcv"] = max(9.0, current["pcv"] - 2.0)
        else:
            baseline["sc"] = current["sc"] * 0.99
            baseline["bu"] = current["bu"] * 0.98

        return {key: round(value, 4) for key, value in baseline.items()}

    def _ambiguous_baseline_from_current(
        self,
        current: dict[str, float],
        index: int,
    ) -> dict[str, float]:
        """
        Build a mixed-signal baseline for abstention-oriented progression rows.

        The resulting trajectory has small renal-marker movement and opposing
        supporting markers, so "stable", "worsening", and "improving" are all
        underdetermined from the available evidence.
        """
        baseline = current.copy()
        sign = -1.0 if index % 2 else 1.0

        baseline["sc"] = max(0.4, min(15.0, current["sc"] * (1.0 + sign * 0.035)))
        baseline["bu"] = max(1.5, min(200.0, current["bu"] * (1.0 - sign * 0.055)))
        baseline["al"] = max(0.0, min(5.0, current["al"] + sign * 0.5))
        baseline["hemo"] = max(3.1, min(17.8, current["hemo"] - sign * 0.35))
        baseline["pcv"] = max(9.0, min(54.0, current["pcv"] + sign * 1.0))

        return {key: round(value, 4) for key, value in baseline.items()}

    def _deferral_metadata(self, row: pd.Series) -> dict[str, Any]:
        """
        Mark records where a clinically cautious model should defer.

        These labels are used only for deferral-alignment scoring; they are
        not exposed in PatientRecord.features. A record should be deferred
        when the binary CKD label conflicts with eGFR-derived stage, or when
        eGFR sits close to a KDIGO staging threshold where small measurement
        variation can change the clinical interpretation.
        """
        egfr = float(row.get("egfr"))
        stage = int(row.get("ckd_stage"))
        class_label = int(round(float(row.get("class"))))

        label_stage_conflict = (class_label == 1 and stage >= 3) or (
            class_label == 0 and stage <= 2
        )
        nearest_threshold, threshold_margin = self._nearest_egfr_threshold(egfr)
        near_thresholds = [nearest_threshold] if threshold_margin <= 3.0 else []

        should_abstain = bool(near_thresholds)
        if self.config.task != Task.STAGING:
            should_abstain = should_abstain or label_stage_conflict
        reasons: list[str] = []
        if label_stage_conflict and self.config.task != Task.STAGING:
            reasons.append("label_stage_conflict")
        if near_thresholds:
            joined = ",".join(str(int(t)) for t in near_thresholds)
            reasons.append(f"egfr_near_threshold:{joined}")

        return {
            "should_abstain": should_abstain,
            "deferral_reason": ";".join(reasons) if reasons else "none",
        }

    @staticmethod
    def _nearest_egfr_threshold(egfr: float) -> tuple[float, float]:
        threshold = min((15.0, 30.0, 60.0, 90.0), key=lambda t: abs(egfr - t))
        return threshold, round(abs(egfr - threshold), 2)
