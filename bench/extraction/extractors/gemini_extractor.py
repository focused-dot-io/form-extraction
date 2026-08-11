"""Field extraction via Google's Gemini vision models (google-genai SDK)."""

import json
import os
import time

from google import genai
from google.genai import errors, types
from langsmith.wrappers import wrap_gemini

from bench.extraction.extractors.common import EXTRACTION_INSTRUCTIONS, coerce_result, validate_against_schema
from bench.extraction.image_utils import load_as_png_bytes
from bench.form.field_schema import FormFields

DEFAULT_MODEL = "gemini-flash-latest"

# The free tier caps at 5 requests/minute — comfortably exceeded by an
# 11-image eval run, so back off and retry rather than losing samples.
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 20

_client: genai.Client | None = None


def _get_model() -> str:
    # Read lazily, not at import time — otherwise this reads the env before
    # a caller's load_dotenv() has run, silently falling back to the default.
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _get_client() -> genai.Client:
    # wrap_gemini traces each call as its own LLM run (model, token usage)
    # rather than folding it into the parent run — needed for LangSmith's
    # "Models" column.
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        _client = wrap_gemini(genai.Client(api_key=api_key))
    return _client


def extract(image_path: str) -> dict:
    png_bytes = load_as_png_bytes(image_path)

    response = _generate_with_retry(png_bytes)

    try:
        raw = json.loads(response.text)
    except (TypeError, ValueError):
        return coerce_result({})
    return validate_against_schema(raw)


def _generate_with_retry(png_bytes: bytes):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _get_client().models.generate_content(
                model=_get_model(),
                contents=[
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    EXTRACTION_INSTRUCTIONS,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FormFields,
                ),
            )
        except errors.ClientError as e:
            if e.code != 429 or attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
