"""Utilities for fetching legislation text from vic.gov.au."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class LegislationFetcher:
    base_url: str = "https://www.legislation.vic.gov.au"
    timeout: int = 20
    max_chars: int = 1600

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "VicPol-GPT-Agent/1.0 (+https://www.police.vic.gov.au)",
            "Accept-Language": "en-AU,en;q=0.9",
        })

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
        text = main.get_text(" ", strip=True)
        text = " ".join(text.split())
        return text[: self.max_chars]

    def search_portal(self, query: str) -> Optional[str]:
        params = {"q": query}
        response = self.session.get(f"{self.base_url}/search", params=params, timeout=self.timeout)
        if not response.ok:
            return None
        return self._extract_text(response.text)

    def fetch_url(self, url: str) -> Optional[str]:
        if not url.startswith("http"):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        response = self.session.get(url, timeout=self.timeout)
        if not response.ok:
            return None
        return self._extract_text(response.text)
