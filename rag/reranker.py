"""Compatibility reranker for old imports.

Live online RAG ranks documents inside rag.retriever using claim anchors.
"""

from __future__ import annotations

from typing import Any


def rerank(query: str, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:top_k]
