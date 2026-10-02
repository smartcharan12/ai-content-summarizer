"""Runtime settings for the inference service (API / Docker / AWS).

Training is configured by YAML files; the *running service* is configured by
environment variables. That is the twelve-factor-app convention: the same
Docker image runs on your laptop and on EC2, and only the environment changes.

Values are read from (highest priority first):
1. real environment variables (e.g. ``-e DEVICE=cpu`` in ``docker run``)
2. a ``.env`` file in the project root (copy ``.env.example``)
3. the defaults below
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from summarizer.constants import PROJECT_ROOT


class AppSettings(BaseSettings):
    """Environment-driven settings, validated by Pydantic at start-up."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated variables in .env
        case_sensitive=False,
        protected_namespaces=(),  # allow field names starting with "model_"
    )

    # --- Application -----------------------------------------------------
    app_name: str = "AI Content Summarizer"
    app_env: Literal["development", "staging", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "*"  # comma-separated list, e.g. "https://me.dev,http://localhost:3000"

    # --- Model -----------------------------------------------------------
    # "registry": load the production model recorded in artifacts/models/registry.json
    #             (falls back to model_name_or_path if no model has been trained yet)
    # "path":     always load model_name_or_path (a local folder or a Hub model id)
    model_source: Literal["registry", "path"] = "registry"
    model_name_or_path: str = "google/pegasus-xsum"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    generation_profile: str = "optimized"
    load_model_on_startup: bool = True

    # --- Input limits (protect the service from abuse / OOM) -------------
    min_input_words: int = Field(default=5, ge=1)
    max_input_chars: int = Field(default=20_000, ge=100)
    max_batch_size: int = Field(default=8, ge=1, le=64)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached settings instance (read the environment only once)."""
    return AppSettings()
