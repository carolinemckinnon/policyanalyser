"""CLI entry point for asking the VicPol GPT agent a question."""
from __future__ import annotations

import argparse
import json
import sys

from gpt_agent.agent import PolicyAgent
from gpt_agent.config import AgentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the VicPol GPT Agent a question.")
    parser.add_argument("question", nargs="?", help="Question to ask. If omitted, read from stdin.")
    parser.add_argument("--legislation-url", dest="legislation_url", help="Specific legislation URL to fetch.")
    parser.add_argument("--top-k", type=int, default=None, help="Override number of context chunks to use.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    question = args.question or sys.stdin.read().strip()
    if not question:
        print("No question provided.")
        return 1
    config = AgentConfig()
    if args.top_k is not None:
        config.top_k = args.top_k
    agent = PolicyAgent(config=config)
    answer = agent.answer(question, legislation_url=args.legislation_url)
    print("\n=== ANSWER ===\n")
    print(answer.text)
    print("\n=== SOURCES ===")
    for citation in answer.citations:
        print(f"- {citation.get('chunk_id')}: {citation.get('title')} -> {citation.get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
