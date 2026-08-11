#!/usr/bin/env python
"""Run all four extractors (OpenAI, Gemini, Claude, OCR) on one form image
and print a side-by-side comparison table.

Usage:
    python run.py path/to/filled_form.jpg
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()  # before importing bench.* — those modules read provider API keys at import time

from bench.extraction.graph import run_all
from bench.form.field_schema import FIELD_NAMES


def print_comparison(combined: dict) -> None:
    providers = ["openai", "gemini", "claude", "ocr"]
    col_width = 22

    header = f"{'field':<24}" + "".join(f"{p:<{col_width}}" for p in providers)
    print(header)
    print("-" * len(header))

    for field in FIELD_NAMES:
        row = f"{field:<24}"
        for provider in providers:
            value = combined.get(provider, {}).get(field, "")
            display = (value[: col_width - 3] + "...") if len(value) > col_width - 1 else value
            row += f"{display:<{col_width}}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Path to a filled-in form image or PDF")
    args = parser.parse_args()

    combined = run_all(args.image_path)
    print_comparison(combined)


if __name__ == "__main__":
    sys.exit(main())
