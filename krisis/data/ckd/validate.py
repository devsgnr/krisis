"""
krisis/data/ckd/validate.py

Validation for the UCI Chronic Kidney Disease CSV schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RAW_ID_COLUMN = "id"
RAW_TARGET_COLUMN = "class"
RAW_FEATURE_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]
RAW_COLUMN_ORDER = [RAW_ID_COLUMN, *RAW_FEATURE_COLUMNS, RAW_TARGET_COLUMN]

RAW_NUMERIC_COLUMNS = [
    "id",
    "age",
    "bp",
    "sg",
    "al",
    "su",
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

RAW_CATEGORICAL_VALUES = {
    "rbc": {"normal", "abnormal"},
    "pc": {"normal", "abnormal"},
    "pcc": {"present", "notpresent"},
    "ba": {"present", "notpresent"},
    "htn": {"yes", "no"},
    "dm": {"yes", "no"},
    "cad": {"yes", "no"},
    "appet": {"good", "poor"},
    "pe": {"yes", "no"},
    "ane": {"yes", "no"},
    "class": {"ckd", "notckd"},
}


@dataclass(frozen=True)
class CKDValidationResult:
    """Clean UCI CKD DataFrame plus lightweight validation metadata."""

    dataframe: pd.DataFrame
    n_rows: int
    n_missing_values: int


def validate_ckd_csv(
    df: pd.DataFrame, *, data_path: str = "CKD CSV"
) -> CKDValidationResult:
    """
    Validate and normalize a raw UCI CKD CSV DataFrame.

    The returned DataFrame is reordered to the canonical UCI schema and has
    object values lowercased/trimmed so downstream encoders see stable labels.
    """
    if df.empty:
        raise ValueError(f"{data_path} is empty. Expected the UCI CKD CSV schema.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            f"{data_path} has duplicate columns: {duplicate_columns}. "
            "Column names must be unique."
        )

    actual = set(df.columns)
    required = set(RAW_COLUMN_ORDER)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        raise ValueError(
            f"{data_path} is missing required UCI CKD columns: {missing}. "
            f"Expected columns: {RAW_COLUMN_ORDER}."
        )
    if extra:
        raise ValueError(
            f"{data_path} has unexpected columns: {extra}. "
            "Krisis CKDSuite currently supports only the UCI CKD CSV schema."
        )

    clean = df.loc[:, RAW_COLUMN_ORDER].copy()
    clean = clean.replace(r"^\s*$", pd.NA, regex=True)
    _normalize_text_columns(clean)
    _validate_numeric_columns(clean, data_path)
    _validate_identifier_column(clean, data_path)
    _normalize_known_source_anomalies(clean)
    _validate_categorical_columns(clean, data_path)
    _validate_target_column(clean, data_path)

    return CKDValidationResult(
        dataframe=clean,
        n_rows=len(clean),
        n_missing_values=int(clean.isna().sum().sum()),
    )


def _normalize_text_columns(df: pd.DataFrame) -> None:
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(
            df[column]
        ):
            df[column] = df[column].map(
                lambda value: value.strip().lower() if isinstance(value, str) else value
            )


def _validate_numeric_columns(df: pd.DataFrame, data_path: str) -> None:
    for column in RAW_NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = df[column].notna() & coerced.isna()
        if invalid_mask.any():
            examples = df.loc[invalid_mask, column].head(3).tolist()
            raise ValueError(
                f"{data_path} column '{column}' contains non-numeric values: "
                f"{examples}. Missing values are allowed, but present values "
                "must be numeric."
            )
        df[column] = coerced


def _normalize_known_source_anomalies(df: pd.DataFrame) -> None:
    """
    Treat rare UCI-derived column-shift artefacts as missing categorical values.

    Some published CKD CSV copies contain a small number of appetite/oedema
    cells with labels from the neighbouring column vocabulary. Missingness is
    already handled by the imputer, so preserving them as unknown is safer than
    encoding them as real clinical states.
    """
    df["appet"] = df["appet"].replace({"yes": pd.NA, "no": pd.NA})
    df["pe"] = df["pe"].replace({"good": pd.NA, "poor": pd.NA})


def _validate_categorical_columns(df: pd.DataFrame, data_path: str) -> None:
    for column, allowed_values in RAW_CATEGORICAL_VALUES.items():
        present = set(df[column].dropna().astype(str))
        invalid = sorted(present - allowed_values)
        if invalid:
            raise ValueError(
                f"{data_path} column '{column}' contains unsupported values: "
                f"{invalid}. Expected one of {sorted(allowed_values)} or missing."
            )


def _validate_identifier_column(df: pd.DataFrame, data_path: str) -> None:
    if df[RAW_ID_COLUMN].isna().any():
        raise ValueError(f"{data_path} column 'id' must not contain missing values.")
    if df[RAW_ID_COLUMN].duplicated().any():
        raise ValueError(f"{data_path} column 'id' must contain unique row ids.")


def _validate_target_column(df: pd.DataFrame, data_path: str) -> None:
    if df[RAW_TARGET_COLUMN].isna().any():
        raise ValueError(f"{data_path} column 'class' must not contain missing values.")
    labels = set(df[RAW_TARGET_COLUMN].astype(str))
    if labels != {"ckd", "notckd"}:
        raise ValueError(
            f"{data_path} must contain both CKD labels: 'ckd' and 'notckd'. "
            f"Observed labels: {sorted(labels)}."
        )
