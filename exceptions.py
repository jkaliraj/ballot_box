"""BallotBox AI — Custom exception hierarchy.

Provides domain-specific exceptions for clear error categorisation,
structured error responses, and improved observability through
consistent error handling patterns.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AIParseError",
    "AIServiceError",
    "BallotBoxError",
    "ConfigurationError",
    "DataLoadError",
    "InputValidationError",
    "RateLimitExceededError",
]


class BallotBoxError(Exception):
    """Base exception for all BallotBox AI application errors.

    Attributes:
        message: Human-readable error description.
        detail: Optional machine-readable context for logging.
    """

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class AIServiceError(BallotBoxError):
    """Raised when the Vertex AI / Gemini service call fails.

    Covers network errors, authentication failures, model unavailability,
    and quota exhaustion.
    """


class AIParseError(BallotBoxError):
    """Raised when Gemini returns non-parseable or malformed JSON.

    Typically occurs when the model ignores the structured output
    instruction and returns free-form text.
    """


class ConfigurationError(BallotBoxError):
    """Raised when required configuration or environment variables are missing."""


class RateLimitExceededError(BallotBoxError):
    """Raised when a client exceeds the per-IP request rate limit."""


class DataLoadError(BallotBoxError):
    """Raised when static data files (glossary, process) cannot be loaded."""


class InputValidationError(BallotBoxError):
    """Raised when user input fails server-side validation checks."""
