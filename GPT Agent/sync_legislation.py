"""Download legislation pages so the agent can reference them offline."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from gpt_agent.config import AgentConfig
from gpt_agent.legislation_fetcher import LegislationFetcher
from gpt_agent.text_utils import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download legislation text into the local corpus.")
    parser.add_argument("url", help="Legislation URL or path from legislation.vic.gov.au")
    parser.add_argument("--output", help="Optional output filename.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AgentConfig()
    config.legislation_dir.mkdir(parents=True, exist_ok=True)
    fetcher = LegislationFetcher()
    text = fetcher.fetch_url(args.url)
    if not text:
        print("Failed to download legislation content.")
        return 1
    if args.output:
        destination = Path(args.output)
        if not destination.is_absolute():
            destination = config.legislation_dir / destination
    else:
        parsed = urlparse(args.url)
        slug_source = parsed.path or parsed.netloc or "legislation"
        destination = config.legislation_dir / f"{slugify(slug_source)}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    print(f"Saved legislation content to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
