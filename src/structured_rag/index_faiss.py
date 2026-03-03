from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.jsonl"
CONFIG_FILE = "config.json"


class FaissIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, passages: Sequence[str], metadata: Sequence[Dict[str, Any]]) -> None:
        if not passages:
            raise ValueError("No passages provided to build index.")
        if len(passages) != len(metadata):
            raise ValueError("Passages and metadata must have equal length.")

        model = self._get_model()
        vectors = model.encode(
            list(passages),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(vectors, dtype=np.float32)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.metadata = list(metadata)

    def save(self, out_dir: str | Path) -> None:
        if self.index is None:
            raise ValueError("No index available to save.")

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_path / INDEX_FILE))

        with (out_path / METADATA_FILE).open("w", encoding="utf-8") as f:
            for row in self.metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        (out_path / CONFIG_FILE).write_text(
            json.dumps({"embedding_model": self.model_name}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: str | Path, model_name: str | None = None) -> "FaissIndex":
        index_path = Path(index_dir)
        config_path = index_path / CONFIG_FILE

        resolved_model = model_name
        if resolved_model is None and config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            resolved_model = config.get("embedding_model")

        obj = cls(model_name=resolved_model or "sentence-transformers/all-MiniLM-L6-v2")
        obj.index = faiss.read_index(str(index_path / INDEX_FILE))

        metadata: List[Dict[str, Any]] = []
        with (index_path / METADATA_FILE).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))
        obj.metadata = metadata
        return obj

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not loaded.")
        if not query.strip():
            return []

        top_k = max(1, min(top_k, self.index.ntotal))
        model = self._get_model()

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            row = dict(self.metadata[idx])
            row["score"] = float(score)
            row["rank"] = rank
            results.append(row)
        return results
