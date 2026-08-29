"""Helpers for decoding OpenAI-compatible image inputs."""

from __future__ import annotations

import base64
from urllib.parse import unquote_to_bytes


def decode_data_url(value: str) -> bytes:
    header, separator, payload = value.partition(",")
    if not separator or not header.startswith("data:"):
        raise ValueError("Invalid data URL")
    if ";base64" in header.lower():
        return base64.b64decode(payload, validate=True)
    return unquote_to_bytes(payload)
