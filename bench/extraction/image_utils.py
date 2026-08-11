"""Normalizes filled-form inputs (photo or PDF) to PNG bytes.

Every extractor reads the same PNG bytes for a given sample, so
differences in results come from the extraction method, not from
feeding different file formats to different providers.
"""

import base64
import io
from pathlib import Path

from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".webp"}


def load_as_png_bytes(path: str | Path, max_long_edge: int = 2000) -> bytes:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        image = _first_pdf_page_as_image(path)
    elif path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        image = Image.open(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    image = image.convert("RGB")
    image = _downscale(image, max_long_edge)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _first_pdf_page_as_image(path: Path) -> Image.Image:
    from pdf2image import convert_from_path

    pages = convert_from_path(str(path), dpi=200, first_page=1, last_page=1)
    if not pages:
        raise ValueError(f"No pages found in {path}")
    return pages[0]


def _downscale(image: Image.Image, max_long_edge: int) -> Image.Image:
    long_edge = max(image.size)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)


def png_bytes_to_b64(png_bytes: bytes) -> str:
    return base64.standard_b64encode(png_bytes).decode("utf-8")
