from __future__ import annotations

from typing import Any, Dict, List

from .index_faiss import FaissIndex


def retrieve_top_k(index: FaissIndex, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return index.search(question, top_k=top_k)
