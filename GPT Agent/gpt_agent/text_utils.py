"""Utility helpers for working with text chunks."""
from __future__ import annotations

import re
from typing import Iterable, List


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = value.replace("\r", "\n")
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    words: List[str] = text.split()
    if not words:
        return []
    effective_overlap = min(chunk_overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - effective_overlap, 1)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
    return chunks


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "untitled"
