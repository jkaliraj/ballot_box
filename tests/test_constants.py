"""Unit tests for BallotBox AI constants module."""

from constants import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    CACHE_KEY_LENGTH,
    DEFAULT_CACHE_TTL,
    DEFAULT_COUNTRY,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_RATE_LIMIT,
    DEFAULT_REGION,
    GEMINI_CHAT_TEMPERATURE,
    GEMINI_JSON_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TOPIC_TEMPERATURE,
    GZIP_MINIMUM_SIZE,
    HSTS_MAX_AGE,
    MAX_CACHE_SIZE_TIMELINE,
    MAX_CACHE_SIZE_TOPIC,
    MAX_COUNTRY_LENGTH,
    MAX_MESSAGE_LENGTH,
    MAX_TOPIC_LENGTH,
    MIN_INPUT_LENGTH,
    RATE_LIMIT_WINDOW_SECONDS,
    SERVICE_NAME,
)


def test_app_metadata():
    """App metadata constants should be non-empty strings."""
    assert isinstance(APP_TITLE, str) and len(APP_TITLE) > 0
    assert isinstance(APP_VERSION, str) and len(APP_VERSION) > 0
    assert isinstance(APP_DESCRIPTION, str) and len(APP_DESCRIPTION) > 0
    assert isinstance(SERVICE_NAME, str) and len(SERVICE_NAME) > 0


def test_input_limits():
    """Input validation constants should be positive integers."""
    assert MAX_MESSAGE_LENGTH == 2000
    assert MAX_TOPIC_LENGTH == 500
    assert MAX_COUNTRY_LENGTH == 100
    assert MIN_INPUT_LENGTH == 1


def test_default_values():
    """Default configuration values should be set correctly."""
    assert DEFAULT_COUNTRY == "India"
    assert DEFAULT_PORT == 8080
    assert DEFAULT_LOG_LEVEL == "INFO"
    assert DEFAULT_CACHE_TTL == 3600
    assert DEFAULT_RATE_LIMIT == 60
    assert DEFAULT_REGION == "us-central1"
    assert DEFAULT_MODEL == "gemini-2.5-flash"


def test_security_constants():
    """Security-related constants should have correct values."""
    assert GZIP_MINIMUM_SIZE == 500
    assert RATE_LIMIT_WINDOW_SECONDS == 60
    assert HSTS_MAX_AGE == 31536000
    assert CACHE_KEY_LENGTH == 16


def test_cache_constants():
    """Cache size limits should be positive integers."""
    assert MAX_CACHE_SIZE_TIMELINE == 50
    assert MAX_CACHE_SIZE_TOPIC == 100


def test_gemini_temperatures():
    """Gemini temperature constants should be valid floats between 0 and 1."""
    assert 0 <= GEMINI_CHAT_TEMPERATURE <= 1
    assert 0 <= GEMINI_JSON_TEMPERATURE <= 1
    assert 0 <= GEMINI_TOPIC_TEMPERATURE <= 1
    assert GEMINI_MAX_OUTPUT_TOKENS > 0
