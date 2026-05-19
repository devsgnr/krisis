"""
krisis/data/ckd/preprocess.py

Formalises the CKD preprocessing pipeline originally developed in impute.ipynb.

Pipeline steps (in order):
    1. Load raw UCI CKD dataset (ckd_full.csv)
    2. Split into nominal and numerical feature groups
    3. OrdinalEncode nominal features with explicit clinical mappings,
       preserving NaN rather than filling
    4. LabelEncode the target column (ckd → 0, notckd → 1)
       Label convention: 0 = CKD present, 1 = CKD absent.
       This is the convention from the original UCI dataset encoding
       and is preserved here for fidelity. Downstream metric code
       must account for this inverted convention.
    5. MICE imputation via IterativeImputer + GradientBoostingRegressor
    6. Pearson correlation feature selection
       (positive threshold ≥ 0.5, negative threshold ≤ -0.4)
    7. Preserve a full, unscaled imputed clinical DataFrame for feature
       engineering and LLM-facing records
    8. Append sc and bu (required for eGFR computation)
    9. MinMaxScaler on selected features for classical ML workflows
    10. Binary ceiling applied back to htn, dm, rbc, pc after scaling

Output columns (in order):
    htn, dm, sg, hemo, pcv, rbcc, rbc, pc, al, bgr, sc, bu, class
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from krisis.data.base import BasePreprocessor, FeatureSet

# ── Constants ─────────────────────────────────────────────────────────────────

# Nominal columns and their explicit clinical ordinal mappings.
# NaN is preserved — not filled — prior to imputation.
NOMINAL_COLUMNS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

ORDINAL_MAPPINGS = [
    {"col": "htn", "mapping": {"no": 0, "yes": 1, np.nan: np.nan}},
    {"col": "dm", "mapping": {"no": 0, "yes": 1, np.nan: np.nan}},
    {"col": "cad", "mapping": {"no": 0, "yes": 1, np.nan: np.nan}},
    {"col": "pe", "mapping": {"no": 0, "yes": 1, np.nan: np.nan}},
    {"col": "ane", "mapping": {"no": 0, "yes": 1, np.nan: np.nan}},
    {"col": "appet", "mapping": {"poor": 0, "good": 1, np.nan: np.nan}},
    {"col": "pc", "mapping": {"normal": 0, "abnormal": 1, np.nan: np.nan}},
    {"col": "pcc", "mapping": {"notpresent": 0, "present": 1, np.nan: np.nan}},
    {"col": "ba", "mapping": {"notpresent": 0, "present": 1, np.nan: np.nan}},
    {"col": "rbc", "mapping": {"normal": 0, "abnormal": 1, np.nan: np.nan}},
]

NUMERICAL_COLUMNS = [
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
]

# Pearson correlation thresholds used in the original feature selection
POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.4

# These two features are always appended regardless of correlation,
# because they are required for eGFR computation downstream.
ALWAYS_INCLUDE = ["sc", "bu"]

# Final output column order (matches original notebook)
FINAL_COLUMN_ORDER = [
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

# Features that get binary ceiling applied after MinMax scaling
BINARY_FEATURES = ["htn", "dm", "rbc", "pc"]


# ── CKDPreprocessor ───────────────────────────────────────────────────────────


class CKDPreprocessor(BasePreprocessor):
    """
    Preprocesses the raw UCI Chronic Kidney Disease dataset.

    This class formalises the pipeline originally developed in impute.ipynb.
    It is designed to be fitted once on the full dataset, then reused via
    transform() on new data without re-fitting.

    Example:
        ```python
        preprocessor = CKDPreprocessor(seed=42)
        df_processed = preprocessor.fit_transform(df_raw)
        ```

        `df_processed` contains feature columns plus the encoded `class`
        column, scaled and ready for feature engineering.

    Attributes:
        feature_set: FeatureSet.FULL or FeatureSet.REDUCED
            - FULL: all 24 input features after encoding and imputation
            - REDUCED: the 12 features selected by Pearson correlation
              (plus sc and bu for eGFR), matching the original notebook
        seed: random seed passed to IterativeImputer and scikit-learn
        scaler_path: optional path to persist/load the fitted MinMaxScaler
    """

    def __init__(
        self,
        feature_set: FeatureSet = FeatureSet.REDUCED,
        seed: int = 42,
        scaler_path: str | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.feature_set = feature_set
        self.scaler_path = scaler_path

        # Internal state — set during fit_transform
        self._ordinal_encoder: OrdinalEncoder | None = None
        self._label_encoder: LabelEncoder | None = None
        self._imputer: IterativeImputer | None = None
        self._scaler: MinMaxScaler | None = None
        self._selected_features: list[str] | None = None
        self._imputed_df: pd.DataFrame | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the full preprocessing pipeline on df and return the
        transformed DataFrame.

        The returned DataFrame contains:
            - 12 feature columns (REDUCED) or 24 feature columns (FULL)
            - 1 target column: 'class' (0 = CKD, 1 = not CKD)

        Sets self._is_fitted = True on completion.
        """
        df = df.copy()

        # Step 1 — separate features and target
        features = df.iloc[:, 1:25]  # columns 1-24 are features
        target = df.iloc[:, -1]  # last column is the class label

        nominal = features[NOMINAL_COLUMNS]
        numerical = features[NUMERICAL_COLUMNS]

        # Step 2 — OrdinalEncode nominal features
        self._ordinal_encoder = OrdinalEncoder(
            cols=NOMINAL_COLUMNS,
            handle_missing="return_nan",
            mapping=ORDINAL_MAPPINGS,
        )
        nominal = self._ordinal_encoder.fit_transform(nominal)
        nominal = pd.DataFrame(nominal, columns=NOMINAL_COLUMNS).replace(-1, np.nan)

        # Step 3 — LabelEncode target
        self._label_encoder = LabelEncoder()
        target_encoded = self._label_encoder.fit_transform(
            pd.DataFrame(target, dtype=str).values.ravel()
        )
        target_df = pd.DataFrame(target_encoded, columns=["class"], dtype="int64")

        # Step 4 — merge and MICE imputation
        merged = pd.concat([nominal, numerical, target_df], axis=1)

        self._imputer = IterativeImputer(
            estimator=GradientBoostingRegressor(),
            random_state=self.seed,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed_values = self._imputer.fit_transform(merged)

        imputed_df = (
            pd.DataFrame(
                imputed_values,
                columns=merged.columns,
                dtype=float,
            )
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
        )
        imputed_df["class"] = imputed_df["class"].round().astype("int64")
        self._imputed_df = imputed_df.copy()

        # Step 5 — feature selection
        if self.feature_set == FeatureSet.REDUCED:
            # The original notebook derived this canonical reduced schema via
            # correlation analysis. Keep the public schema stable across runs
            # and datasets instead of recomputing a potentially different
            # feature set on every fit.
            selected_features = FINAL_COLUMN_ORDER.copy()
        else:
            # FULL: all features except the target
            selected_features = [c for c in imputed_df.columns if c != "class"]

        self._selected_features = selected_features

        # Step 6 — build model-ready DataFrame
        model_df = imputed_df[selected_features + ["class"]].copy()

        # Reorder to the canonical column order (REDUCED only)
        if self.feature_set == FeatureSet.REDUCED:
            model_df = pd.DataFrame(model_df[FINAL_COLUMN_ORDER + ["class"]])

        # Step 7 — MinMax scaling (features only, not target)
        feature_cols = [c for c in model_df.columns if c != "class"]
        X = model_df[feature_cols]

        self._scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self._scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        # Persist scaler if path provided
        if self.scaler_path:
            scaler_dir = os.path.dirname(self.scaler_path)
            if scaler_dir:
                os.makedirs(scaler_dir, exist_ok=True)
            joblib.dump(self._scaler, self.scaler_path)

        # Step 8 — binary ceiling on categorical features (REDUCED only)
        if self.feature_set == FeatureSet.REDUCED:
            for col in BINARY_FEATURES:
                if col in X_scaled_df.columns:
                    X_scaled_df[col] = np.ceil(X_scaled_df[col])

        # Reassemble
        result = X_scaled_df.copy()
        result["class"] = model_df["class"].values

        self._is_fitted = True
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the already-fitted pipeline to new data.
        Raises RuntimeError if called before fit_transform().
        """
        self._check_fitted()
        df = df.copy()

        features = df.iloc[:, 1:25]
        target = df.iloc[:, -1]

        nominal = features[NOMINAL_COLUMNS]
        numerical = features[NUMERICAL_COLUMNS]

        # Encode
        nominal = self._ordinal_encoder.transform(nominal)
        nominal = pd.DataFrame(nominal, columns=NOMINAL_COLUMNS).replace(-1, np.nan)

        target_encoded = self._label_encoder.transform(
            pd.DataFrame(target, dtype=str).values.ravel()
        )
        target_df = pd.DataFrame(target_encoded, columns=["class"], dtype="int64")

        merged = pd.concat([nominal, numerical, target_df], axis=1)

        # Impute
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed_values = self._imputer.transform(merged)

        imputed_df = pd.DataFrame(
            imputed_values,
            columns=merged.columns,
            dtype=float,
        )
        imputed_df["class"] = imputed_df["class"].round().astype("int64")

        # Select features
        model_df = imputed_df[self._selected_features + ["class"]].copy()

        if self.feature_set == FeatureSet.REDUCED:
            model_df = pd.DataFrame(model_df[FINAL_COLUMN_ORDER + ["class"]])

        # Scale
        feature_cols = [c for c in model_df.columns if c != "class"]
        X_scaled = self._scaler.transform(model_df[feature_cols])
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        if self.feature_set == FeatureSet.REDUCED:
            for col in BINARY_FEATURES:
                if col in X_scaled_df.columns:
                    X_scaled_df[col] = np.ceil(X_scaled_df[col])

        result = X_scaled_df.copy()
        result["class"] = model_df["class"].values
        return result

    def get_imputed_dataframe(self) -> pd.DataFrame:
        """
        Return the full encoded/imputed clinical DataFrame before scaling.

        CKDFeatureEngineer and CKDGenerator need real clinical units
        (age in years, creatinine in mg/dL, etc.). The scaled output from
        fit_transform() is kept for classical ML compatibility, but the
        Krisis LLM suite should use this unscaled frame.
        """
        self._check_fitted()
        if self._imputed_df is None:
            raise RuntimeError("No imputed DataFrame is available.")
        return self._imputed_df.copy()

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names for the fitted feature set."""
        self._check_fitted()
        return self._selected_features  # type: ignore[return-value]

    def get_label_classes(self) -> list[str]:
        """
        Return the label encoder classes in order.
        Index 0 = 'ckd' (CKD present), Index 1 = 'notckd' (CKD absent).

        This is the inverted convention from the UCI dataset.
        Label 0 = CKD present. Label 1 = CKD absent.
        """
        self._check_fitted()
        return list(self._label_encoder.classes_)  # type: ignore[union-attr]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _select_features_by_correlation(self, df: pd.DataFrame) -> list[str]:
        """
        Select features using Pearson correlation against the target column.

        Thresholds (from original notebook):
            positive: correlation >= POSITIVE_THRESHOLD (0.5)
            negative: correlation <= NEGATIVE_THRESHOLD (-0.4)

        sc and bu are always included (required for eGFR computation),
        even if they fall below the thresholds.
        """
        corr = df.corr(method="pearson")["class"].drop("class")
        high_positive = corr[corr >= POSITIVE_THRESHOLD].index.tolist()
        high_negative = corr[corr <= NEGATIVE_THRESHOLD].index.tolist()
        selected = high_positive + high_negative

        # Always include sc and bu
        for col in ALWAYS_INCLUDE:
            if col not in selected and col in df.columns:
                selected.append(col)

        return selected
