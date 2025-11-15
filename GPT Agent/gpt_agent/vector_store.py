"""Simple vector store backed by NumPy arrays."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class RetrievedChunk:
    metadata: dict
    score: float

    @property
    def text(self) -> str:
        return self.metadata.get("text", "")


class VectorStore:
    def __init__(self, embeddings: np.ndarray, metadata: List[dict]):
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings and metadata lengths must match")
        self.embeddings = self._normalize(embeddings)
        self.metadata = metadata

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms

    def save(self, vector_path: Path, metadata_path: Path) -> None:
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(vector_path, embeddings=self.embeddings)
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in self.metadata:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, vector_path: Path, metadata_path: Path) -> "VectorStore":
        if not vector_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Vector store or metadata file missing. Run build_index.py first.")
        data = np.load(vector_path)
        embeddings = data["embeddings"]
        metadata: List[dict] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                metadata.append(json.loads(line))
        return cls(embeddings=embeddings, metadata=metadata)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5, filter_fn=None) -> List[RetrievedChunk]:
        if query_embedding.size == 0 or self.embeddings.size == 0:
            return []
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        normalized_query = query_embedding / query_norm
        scores = self.embeddings @ normalized_query
        sorted_indices = np.argsort(scores)[::-1]
        selected: List[RetrievedChunk] = []
        for idx in sorted_indices:
            if filter_fn is not None and not filter_fn(self.metadata[idx]):
                continue
            if scores[idx] <= 0:
                continue
            selected.append(RetrievedChunk(self.metadata[idx], float(scores[idx])))
            if len(selected) >= top_k:
                break
        return selected
