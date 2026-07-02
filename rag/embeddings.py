"""Compatibility stubs for old embedding imports.

ClinIQ v1.1 performs live online retrieval and does not require a local embedding
model for predefined files. These functions are kept to avoid breaking old imports.
"""

from __future__ import annotations


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [[0.0] for _ in texts]


def embed_query(text: str) -> list[float]:
    return [0.0]
