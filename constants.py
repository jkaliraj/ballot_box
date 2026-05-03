"""BallotBox AI — Application-wide constants.

Centralises all magic numbers, default values, and configuration
boundaries to eliminate scattered literals and improve maintainability.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "APP_DESCRIPTION",
    "APP_TITLE",
    "APP_VERSION",
    "CACHE_KEY_LENGTH",
    "DEFAULT_CACHE_TTL",
    "DEFAULT_COUNTRY",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_MODEL",
    "DEFAULT_PORT",
    "DEFAULT_RATE_LIMIT",
    "DEFAULT_REGION",
    "GEMINI_CHAT_TEMPERATURE",
    "GEMINI_JSON_TEMPERATURE",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_TOPIC_TEMPERATURE",
    "GZIP_MINIMUM_SIZE",
    "HSTS_MAX_AGE",
    "MAX_CACHE_SIZE_TIMELINE",
    "MAX_CACHE_SIZE_TOPIC",
    "MAX_COUNTRY_LENGTH",
    "MAX_MESSAGE_LENGTH",
    "MAX_TOPIC_LENGTH",
    "MIN_INPUT_LENGTH",
    "RATE_LIMIT_WINDOW_SECONDS",
    "SERVICE_NAME",
]

# ── Application metadata ──────────────────────────────────────
APP_TITLE: Final[str] = "BallotBox AI"
APP_VERSION: Final[str] = "1.0.0"
APP_DESCRIPTION: Final[str] = (
    "Election Literacy & Voter Empowerment Platform powered by "
    "Google Gemini 2.5 Flash & Vertex AI"
)
SERVICE_NAME: Final[str] = "ballotbox-ai"

# ── Input validation boundaries ───────────────────────────────
MAX_MESSAGE_LENGTH: Final[int] = 2000
MAX_TOPIC_LENGTH: Final[int] = 500
MAX_COUNTRY_LENGTH: Final[int] = 100
MIN_INPUT_LENGTH: Final[int] = 1

# ── Default configuration values ──────────────────────────────
DEFAULT_COUNTRY: Final[str] = "India"
DEFAULT_PORT: Final[int] = 8080
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_CACHE_TTL: Final[int] = 3600
DEFAULT_RATE_LIMIT: Final[int] = 60
DEFAULT_REGION: Final[str] = "us-central1"
DEFAULT_MODEL: Final[str] = "gemini-2.5-flash"

# ── Middleware configuration ──────────────────────────────────
GZIP_MINIMUM_SIZE: Final[int] = 500
RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
HSTS_MAX_AGE: Final[int] = 31536000

# ── Cache configuration ───────────────────────────────────────
CACHE_KEY_LENGTH: Final[int] = 16
MAX_CACHE_SIZE_TIMELINE: Final[int] = 50
MAX_CACHE_SIZE_TOPIC: Final[int] = 100

# ── Gemini AI parameters ─────────────────────────────────────
GEMINI_CHAT_TEMPERATURE: Final[float] = 0.7
GEMINI_JSON_TEMPERATURE: Final[float] = 0.3
GEMINI_TOPIC_TEMPERATURE: Final[float] = 0.5
GEMINI_MAX_OUTPUT_TOKENS: Final[int] = 2048
