"""
ENGRAM Protocol — Centralized Configuration


Single source of truth for all runtime configuration.
Uses pydantic-settings for validation and type coercion.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from kvcos.core.types import CompressionMethod, IndexBackend, StorageBackend


class EngramConfig(BaseSettings):
    """ENGRAM runtime configuration.

    Loaded from environment variables with ENGRAM_ prefix,
    or from a .env file in the project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────
    port: int = 8080
    host: str = "0.0.0.0"

    # ── Storage ───────────────────────────────────────────────
    data_dir: Path = Path.home() / ".engram" / "data"
    backend: StorageBackend = StorageBackend.LOCAL
    default_compression: CompressionMethod = CompressionMethod.Q8_0

    # ── FAISS Index (D2) ──────────────────────────────────────
    index_backend: IndexBackend = IndexBackend.FAISS_FLAT_IP
    index_dir: Path = Path.home() / ".engram" / "index"
    # State vector dimension — must match extraction output
    # 128 for mean_pool (head_dim), 160 for svd_project (rank-160)
    state_vec_dim: int = 160

    # ── LLM Runtime (D1) ──────────────────────────────────────
    model_path: str = ""  # Path to GGUF model file
    n_gpu_layers: int = 0  # D1: CPU-only Phase 1 (avoids Issue #743)
    n_ctx: int = 16384  # D6: 16K context for Phase 1 demo target

    # ── Phase 2: Remote backends ──────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_max_memory_gb: float = 2.0
    s3_bucket: str = "engram-cache"
    s3_region: str = "eu-central-1"
    cloudflare_r2_endpoint: str = ""

    # ── Phase 2: Semantic index ───────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "engram_states"
    cohere_api_key: str = ""

    # ── Phase 4: Cross-model transfer ─────────────────────────
    adapter_enabled: bool = False
    adapter_checkpoint_dir: Path = Path.home() / ".engram" / "adapters"

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> EngramConfig:
    """Get the singleton config instance. Cached after first call."""
    config = EngramConfig()
    config.ensure_dirs()
    return config
