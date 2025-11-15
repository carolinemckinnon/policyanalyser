"""CLI helper to build the agent vector store."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import AgentConfig
from .doc_loader import chunk_documents, load_source_documents
from .embeddings import embed_texts
from .vector_store import VectorStore


def build_vector_store(config: AgentConfig) -> Tuple[int, int]:
    documents = load_source_documents(
        config.vpm_dir,
        config.legislation_dir,
        include_legislation=config.include_legislation,
    )
    if not documents:
        raise FileNotFoundError("No VPM or legislation documents found.")
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise RuntimeError("Documents were loaded but chunking produced no data.")
    texts = [chunk.text for chunk in chunks]
    metadata = [chunk.to_metadata() for chunk in chunks]
    embeddings = embed_texts(texts, model=config.embedding_model)
    if embeddings.shape[0] != len(metadata):
        raise RuntimeError("Failed to compute embeddings for all chunks.")
    store = VectorStore(embeddings=embeddings, metadata=metadata)
    store.save(config.vector_path, config.metadata_path)
    return len(documents), len(chunks)
