"""Field extraction via Anthropic's Claude vision models."""

import json
import os

import anthropic
from langsmith.wrappers import wrap_anthropic

from bench.extraction.extractors.common import EXTRACTION_INSTRUCTIONS, coerce_result, validate_against_schema
from bench.extraction.image_utils import load_as_png_bytes, png_bytes_to_b64
from bench.form.field_schema import JSON_SCHEMA

DEFAULT_MODEL = "claude-opus-5"

_client: anthropic.Anthropic | None = None


def _get_model() -> str:
    # Read lazily, not at import time — otherwise this reads the env before
    # a caller's load_dotenv() has run, silently falling back to the default.
    return os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)


def _get_client() -> anthropic.Anthropic:
    # wrap_anthropic traces each call as its own LLM run (model, token usage)
    # rather than folding it into the parent run — needed for LangSmith's
    # "Models" column.
    global _client
    if _client is None:
        _client = wrap_anthropic(anthropic.Anthropic())
    return _client


def extract(image_path: str) -> dict:
    png_bytes = load_as_png_bytes(image_path)
    b64 = png_bytes_to_b64(png_bytes)

    response = _get_client().messages.create(
        model=_get_model(),
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": EXTRACTION_INSTRUCTIONS},
                ],
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": JSON_SCHEMA}},
    )

    if response.stop_reason == "refusal":
        return coerce_result({})

    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        return coerce_result({})

    try:
        raw = json.loads(text_block.text)
    except json.JSONDecodeError:
        return coerce_result({})
    return validate_against_schema(raw)
