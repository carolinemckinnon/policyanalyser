"""Embedding helpers using the OpenAI API."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from openai import OpenAI


def embed_texts(texts: Sequence[str], model: str, batch_size: int = 64) -> np.ndarray:
    client = OpenAI()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        if not batch:
            continue
        response = client.embeddings.create(model=model, input=list(batch))
        # Response data is returned in the same order as inputs
        vectors.extend(item.embedding for item in response.data)
    if not vectors:
        return np.empty((0, 0))
    return np.array(vectors, dtype=float)


def embed_query(text: str, model: str) -> np.ndarray:
    embeddings = embed_texts([text], model=model, batch_size=1)
    return embeddings[0] if embeddings.size else np.empty((0,))
