"""Compatibility module for older imports.

ClinIQ v1.1 uses live online retrieval and writes claim-specific evidence to
online_knowledge/. Chroma/static vector-store indexing is intentionally disabled
because the user requested no predefined local knowledge base.
"""

from __future__ import annotations

from typing import Any


class VectorStoreManager:
    """No-op compatibility class.

    Kept so old imports do not crash. Do not use this for retrieval in v1.1.
    """

    COLLECTIONS: dict[str, str] = {}

    def index_all(self) -> dict[str, int]:
        return {}

    def index_collection(self, folder_key: str) -> int:
        return 0

    def query_collection(self, folder_key: str, query: str, top_k: int) -> list[dict[str, Any]]:
        return []
