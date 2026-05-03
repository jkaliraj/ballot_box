"""BallotBox AI — Server-side input sanitization utilities.

Provides defense-in-depth text sanitization for user inputs beyond
Pydantic validation, preventing injection attacks and ensuring
safe processing through the AI pipeline.
"""

from __future__ import annotations

import html
import re
from typing import Final

__all__ = ["sanitize_text", "strip_control_chars"]

# Control characters excluding tab, newline, carriage return
_CONTROL_CHAR_RE: Final[re.Pattern[str]] = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def sanitize_text(text: str) -> str:
    """Sanitize user-provided text for safe processing.

    Applies the following transformations:
    1. Strips leading/trailing whitespace.
    2. Removes ASCII control characters (preserves tabs and newlines).
    3. HTML-escapes angle brackets to neutralise injection attempts.

    Args:
        text: Raw user input string.

    Returns:
        Sanitized string safe for logging, AI prompt injection, and storage.
    """
    text = text.strip()
    text = strip_control_chars(text)
    text = html.escape(text, quote=False)
    return text


def strip_control_chars(text: str) -> str:
    """Remove invisible control characters from text.

    Preserves printable whitespace (spaces, tabs, newlines) while
    removing characters that could cause rendering issues or bypass
    input validation.

    Args:
        text: Input string possibly containing control characters.

    Returns:
        Cleaned string with control characters removed.
    """
    return _CONTROL_CHAR_RE.sub("", text)
