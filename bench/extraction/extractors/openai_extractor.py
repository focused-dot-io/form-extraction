"""Field extraction via OpenAI's vision models."""

import json
import os

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from bench.extraction.extractors.common import EXTRACTION_INSTRUCTIONS, coerce_result, validate_against_schema
from bench.extraction.image_utils import load_as_png_bytes, png_bytes_to_b64
from bench.form.field_schema import JSON_SCHEMA

DEFAULT_MODEL = "gpt-4o"

_client: OpenAI | None = None


def get_model() -> str:
    # Read lazily, not at import time — otherwise this reads the env before
    # a caller's load_dotenv() has run, silently falling back to the default.
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def _get_client() -> OpenAI:
    # wrap_openai makes each call show up as its own traced LLM run (with
    # model name, token usage, etc.) instead of disappearing into the parent
    # run — that's what LangSmith needs to populate the "Models" column.
    global _client
    if _client is None:
        _client = wrap_openai(OpenAI())
    return _client


def extract(image_path: str) -> dict:
    png_bytes = load_as_png_bytes(image_path)
    b64 = png_bytes_to_b64(png_bytes)

    response = _get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_INSTRUCTIONS},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "form_fields", "schema": JSON_SCHEMA, "strict": True},
        },
    )

    content = response.choices[0].message.content
    try:
        raw = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return coerce_result({})
    return validate_against_schema(raw)
