"""Central configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = Path(__file__).resolve().parent / "rag" / "corpus"
INDEX_DIR = DATA_DIR / "index"


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    project: str | None = os.getenv("GOOGLE_CLOUD_PROJECT")
    location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model: str = os.getenv("ADVISOR_MODEL", "gemini-2.5-flash")
    judge_model: str = os.getenv("JUDGE_MODEL", "gemini-2.5-pro")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
    offline: bool = _flag("ADVISOR_OFFLINE")
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    max_tool_rounds: int = int(os.getenv("MAX_TOOL_ROUNDS", "4"))
    max_revisions: int = int(os.getenv("MAX_REVISIONS", "1"))
    groundedness_threshold: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.8"))


def settings() -> Settings:
    return Settings()
