"""Field extraction via plain Tesseract OCR + label-anchored regex parsing.

This is the weak baseline in the comparison: no vision-model understanding,
just raw text recognition plus knowledge of where each label sits on this
specific form. Two known, deliberate limitations (not bugs):

- It has no way to tell which checkbox is filled, so `employment_type`
  always comes back blank.
- The signature and its date sit side by side on one row. When Tesseract
  merges that row into a single OCR line, there's no reliable way for a
  plain regex to know where the name ends and the date begins — both
  fields can come back with the same combined text, or blank.
"""

import io
import re

import pytesseract
from PIL import Image

from bench.extraction.extractors.common import coerce_result
from bench.extraction.image_utils import load_as_png_bytes
from bench.form.field_schema import FIELD_LABELS, FIELD_NAMES

NUMERIC_FIELDS = {"employee_id", "phone_number", "emergency_contact_phone"}
CHECKBOX_FIELDS = {"employment_type"}

# Longest-first so a label that's a substring of another (e.g. "Date
# (MM/DD/YYYY)" inside "Start Date (MM/DD/YYYY)") never matches short.
_LABELS_BY_LENGTH_DESC = sorted(set(FIELD_LABELS.values()), key=len, reverse=True)
_LABEL_PATTERN = re.compile("|".join(re.escape(label) for label in _LABELS_BY_LENGTH_DESC), re.IGNORECASE)


def extract(image_path: str) -> dict:
    png_bytes = load_as_png_bytes(image_path)
    image = Image.open(io.BytesIO(png_bytes))
    raw_text = pytesseract.image_to_string(image)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    result = coerce_result({})
    for field_name in FIELD_NAMES:
        if field_name in CHECKBOX_FIELDS:
            continue
        label = FIELD_LABELS.get(field_name, "")
        value = _find_value_for_label(lines, label)
        if field_name in NUMERIC_FIELDS:
            value = re.sub(r"\D", "", value)
        result[field_name] = value.strip()
    return result


def _is_pure_label_text(line: str) -> bool:
    """True if, after stripping every known label out of the line, nothing is left."""
    stripped = _LABEL_PATTERN.sub("", line)
    return stripped.strip(" :._-") == ""


def _find_value_for_label(lines: list[str], label: str) -> str:
    if not label:
        return ""
    label_lower = label.lower()

    for i, line in enumerate(lines):
        # Match against every known label at once so a longer label "wins"
        # over one that happens to be its substring (e.g. "Start Date
        # (MM/DD/YYYY)" contains "Date (MM/DD/YYYY)" — without this, a plain
        # substring search for "Date (MM/DD/YYYY)" would wrongly match here).
        matches = list(_LABEL_PATTERN.finditer(line))
        for match_index, match in enumerate(matches):
            if match.group().lower() != label_lower:
                continue

            # Same-line trailing text, cut off at the next label on the line
            # (e.g. "Signature   Date (MM/DD/YYYY)" has two labels in one row).
            next_start = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(line)
            trailing = line[match.end():next_start].strip(" :._-")
            if trailing:
                return trailing

            # No value trailed the label on its own line — check the
            # neighboring lines, skipping any that are pure label text.
            for neighbor_index in (i + 1, i - 1):
                if 0 <= neighbor_index < len(lines):
                    candidate = lines[neighbor_index]
                    if not _is_pure_label_text(candidate):
                        return candidate
            return ""

    return ""
