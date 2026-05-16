"""
krisis/prompts/detection.py

Prompts for binary CKD detection (label 0 = disease present, 1 = absent).
"""

from __future__ import annotations

from krisis.data.base import PatientRecord
from krisis.prompts.formatting import features_to_bullet_list

DETECTION_SYSTEM = """You are a careful clinical assistant evaluating chronic \
kidney disease (CKD) risk from tabulated laboratory and history features.

Rules:
- Never invent facts not supported by the provided markers.
- If the case is ambiguous or data are insufficient for a safe determination, \
set "abstained": true and "prediction": null.
- Otherwise set "abstained": false and give your best label.

Label semantics:
- 0 means CKD is present.
- 1 means CKD is absent.

Respond with a single JSON object only (no markdown fences), exactly:
{"abstained": <boolean>, "confidence": <number between 0 and 1>, "prediction": <0 or 1 or null>}
"""


def build_detection_messages(record: PatientRecord) -> list[dict[str, str]]:
    body = features_to_bullet_list(record.features)
    user = (
        "Patient markers:\n"
        f"{body}\n\n"
        "Return the JSON object as specified in the system message."
    )
    return [
        {"role": "system", "content": DETECTION_SYSTEM},
        {"role": "user", "content": user},
    ]
