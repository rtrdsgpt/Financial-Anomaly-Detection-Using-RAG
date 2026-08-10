"""
Centralized settings via pydantic-settings, reading env vars / `.env`.

Replaces the `api_keys.txt` plaintext-key pattern used by
`main_oop.py`/`app_oop.py` as the primary source of truth, while staying
backward-compatible: if an env var isn't set, we fall back to
`api_keys.txt` if that file happens to exist, so nothing that already
relies on it breaks.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_legacy_api_keys(path: str = "api_keys.txt") -> dict:
    """Parse the old `KEY="value"` per-line format, if the file exists"""
    keys = {}
    file = Path(path)
    if not file.exists():
        return keys

    for line in file.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            keys[key.strip()] = value.strip().strip('"')
    return keys


class Settings(BaseSettings):
    """Application configuration. Values are resolved in priority order:
    explicit constructor kwargs > environment variables > `.env` file >
    field defaults below > (for the two API keys only) `api_keys.txt`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    groq_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Default pipeline configuration
    default_ticker: str = "TSLA"
    default_benchmark: str = "SPY"
    default_start_date: str = "2024-10-01"
    default_end_date: str = "2024-12-31"

    # RAG configuration
    vector_store_backend: str = "chroma"
    chroma_persist_directory: str = "chroma_db"
    chunking_strategy: str = "headline"
    use_reranker: bool = True
    use_grounded_citations: bool = True
    rag_min_relevance_score: float = 0.0
    rag_max_retries: int = 2

    # MLOps
    mlflow_tracking_uri: str = "mlruns"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.groq_api_key or not self.finnhub_api_key:
            legacy = _load_legacy_api_keys()
            self.groq_api_key = self.groq_api_key or legacy.get("GROQ_API")
            self.finnhub_api_key = self.finnhub_api_key or legacy.get("FIN_HUB")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton -- suitable as a FastAPI dependency"""
    return Settings()
