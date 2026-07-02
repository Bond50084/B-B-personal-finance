"""Central configuration for the FoerderMatch MVP.

All runtime options are read from environment variables (optionally via a
local ``.env`` file, see ``.env.example``) so the same code base runs in
several modes without code changes:

    EMBEDDING_PROVIDER:  mock | openai | ollama | gemini
    LLM_PROVIDER:        mock | openai | ollama | anthropic | gemini

``mock`` keeps the whole pipeline runnable fully offline (no API keys, no
model downloads) and is the default, which makes end-to-end tests and CI
trivial. For production matching quality, switch to a real embedding model
(e.g. OpenAI ``text-embedding-3-small``) and a real LLM.

Swap-out points for later iterations are deliberately isolated:
    * SQLite  -> PostgreSQL:   only ``database.py`` touches SQL.
    * Chroma  -> pgvector/...: only ``ingest.py`` / ``matching_engine.py``
                               talk to the vector store.
    * Scraper -> crawler farm: only the source functions in ``ingest.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env once at import time; real environment variables take precedence.
load_dotenv(BASE_DIR / ".env", override=False)


def _env(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings (instantiate once, pass around)."""

    # --- Paths ------------------------------------------------------------
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    storage_dir: Path = BASE_DIR / "storage"
    sqlite_path: Path = BASE_DIR / "storage" / "foerdermatch.sqlite3"
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"
    reports_dir: Path = BASE_DIR / "reports"
    mock_data_path: Path = BASE_DIR / "data" / "mock_foerderdatenbank.json"

    # --- Vector store -------------------------------------------------------
    collection_name: str = _env("CHROMA_COLLECTION", "subsidy_programs")

    # --- Provider selection -------------------------------------------------
    embedding_provider: str = _env("EMBEDDING_PROVIDER", "mock").lower()
    llm_provider: str = _env("LLM_PROVIDER", "mock").lower()

    # --- Model names ----------------------------------------------------------
    openai_llm_model: str = _env("OPENAI_LLM_MODEL", "gpt-4o")
    openai_embedding_model: str = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    anthropic_llm_model: str = _env("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-6")
    gemini_llm_model: str = _env("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
    gemini_embedding_model: str = _env("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    ollama_llm_model: str = _env("OLLAMA_LLM_MODEL", "llama3.1")
    ollama_embedding_model: str = _env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_temperature: float = 0.2
    # Cap on the report length. The concise prompt produces ~one sentence per
    # programme, so a small cap keeps cost and latency down.
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 900)
    mock_embedding_dim: int = _env_int("MOCK_EMBEDDING_DIM", 512)

    # --- Matching -------------------------------------------------------------
    top_k: int = _env_int("TOP_K", 5)
    # Retrieve a larger candidate pool before Python-side post-filtering.
    retrieval_pool_factor: int = _env_int("RETRIEVAL_POOL_FACTOR", 3)
    # Hard metadata filters (Component C). Each can be disabled via env for
    # debugging, e.g. FILTER_REGION=false.
    filter_region: bool = _env_bool("FILTER_REGION", True)
    filter_deadline: bool = _env_bool("FILTER_DEADLINE", True)
    filter_applicant: bool = _env_bool("FILTER_APPLICANT", True)

    # --- Scraper (live mode) ----------------------------------------------------
    scraper_user_agent: str = _env(
        "SCRAPER_USER_AGENT",
        "FoerderMatch-MVP/0.1 (Foerdermittel-Recherche; Kontakt siehe Betreiber)",
    )
    scraper_delay_seconds: float = float(_env("SCRAPER_DELAY_SECONDS", "2.0"))
    scraper_timeout_seconds: float = float(_env("SCRAPER_TIMEOUT_SECONDS", "20"))

    def ensure_directories(self) -> None:
        """Create all writable directories the app needs."""
        for path in (self.data_dir, self.storage_dir, self.chroma_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
