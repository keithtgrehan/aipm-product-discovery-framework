from __future__ import annotations

import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _missing_items(answer_norm: str, expected: List[str]) -> List[str]:
    missing: List[str] = []
    for item in expected:
        probe = normalize_text(item)
        if probe and probe not in answer_norm:
            missing.append(item)
    return missing


def score_omission(answer: str, checklist: Dict[str, List[str]]) -> Dict[str, object]:
    conditions = checklist.get("critical_conditions", [])
    qualifiers = checklist.get("critical_qualifiers", [])

    answer_norm = normalize_text(answer)
    missing_conditions = _missing_items(answer_norm, conditions)
    missing_qualifiers = _missing_items(answer_norm, qualifiers)

    total_expected = max(1, len(conditions) + len(qualifiers))
    missing_total = len(missing_conditions) + len(missing_qualifiers)
    omission_score = missing_total / total_expected

    return {
        "missing_conditions": missing_conditions,
        "missing_qualifiers": missing_qualifiers,
        "missing_total": missing_total,
        "total_expected": total_expected,
        "omission_score": omission_score,
        "binary_fail": missing_total > 0,
    }
