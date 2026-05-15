from __future__ import annotations

import re

CLASS_NAMES: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
STAGE_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_STAGE: dict[int, str] = {idx: name for name, idx in STAGE_TO_ID.items()}
IGNORE_LABEL = -1


def normalize_stage_label(description: str) -> int | None:
    """Map a Sleep-EDF annotation description to a 5-class label id."""

    text = re.sub(r"\s+", " ", description.strip().lower())
    if not text:
        return None

    if text.startswith("sleep stage "):
        text = text.removeprefix("sleep stage ").strip()
    elif text.startswith("stage "):
        text = text.removeprefix("stage ").strip()

    if text in {"w", "wake", "awake"}:
        return STAGE_TO_ID["W"]
    if text in {"1", "n1", "s1"}:
        return STAGE_TO_ID["N1"]
    if text in {"2", "n2", "s2"}:
        return STAGE_TO_ID["N2"]
    if text in {"3", "4", "n3", "n4", "s3", "s4"}:
        return STAGE_TO_ID["N3"]
    if text in {"r", "rem", "stage r"}:
        return STAGE_TO_ID["REM"]

    if (
        "movement" in text
        or "unknown" in text
        or "unscored" in text
        or text in {"?", "mt"}
    ):
        return None

    return None
