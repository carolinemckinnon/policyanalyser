"""Configuration helpers for the VicPol GPT Agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def _default_repo_root() -> Path:
    # config.py -> gpt_agent -> GPT Agent -> repo root
    return Path(__file__).resolve().parents[2]


def _default_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class AgentConfig:
    """Holds filesystem paths and runtime settings for the agent."""

    repo_root: Path = field(default_factory=_default_repo_root)
    agent_root: Path = field(default_factory=_default_agent_root)
    storage_dir: Path = field(default_factory=lambda: _default_agent_root() / "storage")
    vpm_dir: Path = field(default_factory=lambda: _default_repo_root() / "VPMs")
    legislation_dir: Path = field(default_factory=lambda: _default_repo_root() / "legislation_texts")
    vector_store_name: str = "vector_store"
    include_legislation: bool = field(
        default_factory=lambda: os.getenv("AGENT_INCLUDE_LEGISLATION", "0").lower() in {"1", "true", "yes"}
    )

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    completion_model: str = field(default_factory=lambda: os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0"))
    top_k: int = int(os.getenv("AGENT_TOP_K", "6"))
    chunk_size: int = int(os.getenv("AGENT_CHUNK_SIZE", "450"))
    chunk_overlap: int = int(os.getenv("AGENT_CHUNK_OVERLAP", "75"))
    legislation_search_chars: int = int(os.getenv("LEGISLATION_SEARCH_CHARS", "1600"))

    def __post_init__(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._vector_path = self.storage_dir / f"{self.vector_store_name}.npz"
        self._metadata_path = self.storage_dir / f"{self.vector_store_name}.jsonl"

    @property
    def vector_path(self) -> Path:
        return self._vector_path

    @property
    def metadata_path(self) -> Path:
        return self._metadata_path

    @classmethod
    def from_env(cls, overrides: Optional[Dict[str, str]] = None) -> "AgentConfig":
        data: Dict[str, str] = dict(os.environ)
        if overrides:
            data.update(overrides)
        # Using overrides only for env-style fields is overkill; just instantiate normally.
        return cls()

    def ensure_data_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vpm_dir.mkdir(parents=True, exist_ok=True)
        self.legislation_dir.mkdir(parents=True, exist_ok=True)

    def describe_paths(self) -> Dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "agent_root": str(self.agent_root),
            "storage_dir": str(self.storage_dir),
            "vpm_dir": str(self.vpm_dir),
            "legislation_dir": str(self.legislation_dir),
            "vector_path": str(self.vector_path),
            "metadata_path": str(self.metadata_path),
            "include_legislation": str(self.include_legislation),
        }
