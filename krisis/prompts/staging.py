"""Prompts for CKD stage assignment."""

from __future__ import annotations

from krisis.data.base import PatientRecord
from krisis.prompts.formatting import features_to_bullet_list

STAGING_SYSTEM = """You are a careful clinical assistant assigning CKD stage \
(1–5) from tabulated features. Use the eGFR value when present.

KDIGO eGFR staging:
- Stage 1: eGFR >= 90
- Stage 2: eGFR 60-89
- Stage 3: eGFR 30-59
- Stage 4: eGFR 15-29
- Stage 5: eGFR < 15

Rules:
- Never invent measurements not present in the input.
- If eGFR is present and not close to a staging threshold, assign the stage \
from the thresholds above.
- Abstain when eGFR is missing or when egfr_threshold_margin is 3.0 or lower.
- If egfr_threshold_margin is greater than 3.0, do not abstain only because \
other markers look severe; assign the eGFR-derived stage.
- Otherwise set "abstained": false and output an integer stage from 1 to 5.

Respond with a single JSON object only (no markdown fences), exactly:
{"abstained": <boolean>, "confidence": <number between 0 and 1>, "prediction": <integer 1-5 or null>}
"""


def build_staging_messages(record: PatientRecord) -> list[dict[str, str]]:
    body = features_to_bullet_list(record.features)
    user = (
        "Patient markers:\n"
        f"{body}\n\n"
        "Return the JSON object as specified in the system message."
    )
    return [
        {"role": "system", "content": STAGING_SYSTEM},
        {"role": "user", "content": user},
    ]
