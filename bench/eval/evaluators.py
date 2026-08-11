"""LangSmith evaluators for comparing extracted fields against ground truth.

Values are normalized before comparison so formatting differences (date
format, phone punctuation, casing) don't count as extraction errors —
only genuinely wrong or missing values do.
"""

import re

from dateutil import parser as date_parser

from bench.form.field_schema import SCORED_FIELD_NAMES

DATE_FIELDS = {"date_of_birth", "start_date", "signature_date"}
NUMERIC_FIELDS = {"employee_id", "phone_number", "emergency_contact_phone"}


def normalize_value(field_name: str, value) -> str:
    value = "" if value is None else str(value).strip()
    if not value:
        return ""
    if field_name in DATE_FIELDS:
        try:
            return date_parser.parse(value).strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            return value.lower()
    if field_name in NUMERIC_FIELDS:
        return re.sub(r"\D", "", value)
    # Trailing punctuation ("Sr." vs "Sr") shouldn't fail a text field any
    # more than casing should.
    value = value.strip(".,").strip()
    return re.sub(r"\s+", " ", value.lower())


def _get_outputs(obj) -> dict:
    if hasattr(obj, "outputs"):
        return obj.outputs or {}
    if isinstance(obj, dict):
        return obj.get("outputs", {}) or {}
    return {}


def field_accuracy_evaluator(run, example):
    """Fraction of the scored fields that match ground truth after normalization.

    `signature_name` is extracted but excluded from scoring — see
    bench/form/field_schema.py:SCORED_FIELD_NAMES for why.
    """
    predicted = _get_outputs(run)
    expected = _get_outputs(example)

    mismatches = []
    correct = 0
    for field in SCORED_FIELD_NAMES:
        predicted_norm = normalize_value(field, predicted.get(field, ""))
        expected_norm = normalize_value(field, expected.get(field, ""))
        if predicted_norm == expected_norm:
            correct += 1
        else:
            mismatches.append(f"{field}: expected={expected_norm!r} got={predicted_norm!r}")

    score = correct / len(SCORED_FIELD_NAMES)
    comment = "all scored fields correct" if not mismatches else "; ".join(mismatches)
    return {"score": score, "comment": comment}


def exact_match_evaluator(run, example):
    """1.0 only if every field matches; 0.0 otherwise."""
    result = field_accuracy_evaluator(run, example)
    return {"score": 1.0 if result["score"] == 1.0 else 0.0, "comment": result["comment"]}
