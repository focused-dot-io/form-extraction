"""Shared helpers for the vision-model extractors (OpenAI, Gemini, Claude)."""

from bench.form.field_schema import FIELD_NAMES, FormFields

EXTRACTION_INSTRUCTIONS = (
    "This image is a filled-out 'Employee Onboarding Form'. It may be handwritten "
    "(neat or messy) or digitally typed. Read every field carefully, including the "
    "checked box under Employment Type. Extract exactly the fields in the provided "
    "schema. Normalize dates to YYYY-MM-DD. Strip all non-digit characters from phone "
    "numbers. If a field is illegible or blank, return an empty string for it rather "
    "than guessing a value that isn't visibly written on the page."
)


def empty_result() -> dict:
    return {name: "" for name in FIELD_NAMES}


def coerce_result(raw: dict) -> dict:
    """Fill in any missing keys so every extractor always returns the full field set."""
    result = empty_result()
    for name in FIELD_NAMES:
        value = raw.get(name, "")
        result[name] = "" if value is None else str(value)
    return result


def validate_against_schema(raw: dict) -> dict:
    try:
        parsed = FormFields(**{k: raw.get(k, "") for k in FIELD_NAMES})
        return parsed.model_dump()
    except Exception:
        return coerce_result(raw)
