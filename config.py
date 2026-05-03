"""BallotBox AI — Centralized Application Configuration.

Loads all settings from environment variables with sensible defaults.
Uses Google Cloud-aware defaults for seamless Cloud Run deployment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

from constants import (
    DEFAULT_CACHE_TTL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_RATE_LIMIT,
    DEFAULT_REGION,
)

__all__ = ["Settings", "get_settings"]

logger = logging.getLogger(__name__)

_VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables.

    Attributes:
        project_id: Google Cloud project ID for Vertex AI.
        location: Google Cloud region (default: us-central1).
        gemini_model: Gemini model identifier for AI generation.
        allowed_origins: Comma-separated CORS origins.
        port: HTTP server port (default: 8080 for Cloud Run).
        environment: Deployment environment (development/production).
        log_level: Python logging level.
        ga_measurement_id: Google Analytics 4 measurement ID.
        enable_cloud_logging: Enable Google Cloud Logging integration.
        cache_ttl_seconds: TTL for in-memory response caching.
        rate_limit_per_minute: Max API requests per IP per minute.
    """

    project_id: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    location: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", DEFAULT_REGION)
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    )
    allowed_origins: str = field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", str(DEFAULT_PORT)))
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "production")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    )
    ga_measurement_id: str = field(
        default_factory=lambda: os.getenv("GA_MEASUREMENT_ID", "")
    )
    enable_cloud_logging: bool = field(
        default_factory=lambda: os.getenv(
            "ENABLE_CLOUD_LOGGING", "true"
        ).lower() == "true"
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL", str(DEFAULT_CACHE_TTL)))
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT", str(DEFAULT_RATE_LIMIT)))
    )

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.log_level not in _VALID_LOG_LEVELS:
            object.__setattr__(self, "log_level", DEFAULT_LOG_LEVEL)
        if not 1 <= self.port <= 65535:
            object.__setattr__(self, "port", DEFAULT_PORT)
        if self.cache_ttl_seconds < 0:
            object.__setattr__(self, "cache_ttl_seconds", DEFAULT_CACHE_TTL)
        if self.rate_limit_per_minute < 1:
            object.__setattr__(self, "rate_limit_per_minute", DEFAULT_RATE_LIMIT)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton.

    Returns:
        Settings: Frozen dataclass with all application configuration.
    """
    settings = Settings()
    logger.info(
        "Settings loaded: project=%s, region=%s, model=%s, env=%s",
        settings.project_id,
        settings.location,
        settings.gemini_model,
        settings.environment,
    )
    return settings
