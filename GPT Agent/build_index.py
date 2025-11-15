"""Build the VPM / legislation vector store."""
from __future__ import annotations

import argparse

from gpt_agent.config import AgentConfig
from gpt_agent.index_builder import build_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or rebuild the GPT Agent vector store.")
    parser.add_argument("--chunk-size", type=int, help="Words per chunk (default from config).")
    parser.add_argument("--chunk-overlap", type=int, help="Word overlap between chunks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AgentConfig()
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap
    documents, chunks = build_vector_store(config)
    print(f"Built vector store with {documents} documents and {chunks} chunks.")
    print(f"Embeddings stored at: {config.vector_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
