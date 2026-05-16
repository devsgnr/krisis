"""
krisis/prompts/progression.py

Prompts for synthetic two-visit CKD progression assessment.
"""

from __future__ import annotations

from krisis.data.base import PatientRecord

PROGRESSION_SYSTEM = """You are a careful clinical assistant evaluating CKD \
trajectory from two tabulated visits.

Allowed predictions when not abstaining:
- "worsening": kidney-related markers meaningfully deteriorated
- "improving": kidney-related markers meaningfully improved
- "stable": no clinically meaningful direction is evident

Rules:
- Compare baseline vs current markers; do not invent missing measurements.
- Do not force "stable" just because changes are small; small or mixed \
changes are often clinically indeterminate.
- Abstain when renal markers move less than about 10%, when creatinine/urea/\
albumin/haemoglobin point in conflicting directions, or when the trajectory is \
near a staging threshold.
- If the direction is ambiguous or unsupported, set "abstained": true and \
"prediction": null.
- Otherwise set "abstained": false and output one allowed prediction string.

Respond with a single JSON object only (no markdown fences), exactly:
{"abstained": <boolean>, "confidence": <number between 0 and 1>, "prediction": <"worsening" or "improving" or "stable" or null>}
"""


def _format_visit(name: str, values: dict[str, float]) -> str:
    lines = [f"{name}:"]
    for key, value in sorted(values.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_progression_messages(record: PatientRecord) -> list[dict[str, str]]:
    baseline = record.features.get("baseline", {})
    current = record.features.get("current", {})
    months = record.features.get("trajectory_months", 6)
    user = (
        f"Visits are {months} months apart.\n\n"
        f"{_format_visit('Baseline visit', baseline)}\n\n"
        f"{_format_visit('Current visit', current)}\n\n"
        "Return the JSON object as specified in the system message."
    )
    return [
        {"role": "system", "content": PROGRESSION_SYSTEM},
        {"role": "user", "content": user},
    ]
