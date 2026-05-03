"""Unit tests for BallotBox AI sanitize module."""

from sanitize import sanitize_text, strip_control_chars


def test_sanitize_text_strips_whitespace():
    """sanitize_text should strip leading and trailing whitespace."""
    assert sanitize_text("  hello  ") == "hello"


def test_sanitize_text_removes_control_chars():
    """sanitize_text should remove ASCII control characters."""
    assert sanitize_text("hello\x00world") == "helloworld"
    assert sanitize_text("test\x01\x02data") == "testdata"


def test_sanitize_text_preserves_allowed_whitespace():
    """sanitize_text should preserve tabs and newlines within text."""
    result = sanitize_text("hello\tworld")
    assert "\t" in result


def test_sanitize_text_html_escapes():
    """sanitize_text should HTML-escape dangerous characters."""
    result = sanitize_text('<script>alert("xss")</script>')
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_sanitize_text_empty_string():
    """sanitize_text should return empty string for empty input."""
    assert sanitize_text("") == ""
    assert sanitize_text("   ") == ""


def test_strip_control_chars_removes_null_bytes():
    """strip_control_chars should remove null bytes and other control chars."""
    assert strip_control_chars("hello\x00world") == "helloworld"


def test_strip_control_chars_preserves_normal_text():
    """strip_control_chars should not modify normal text."""
    assert strip_control_chars("Hello World 123!") == "Hello World 123!"


def test_strip_control_chars_preserves_tabs_newlines():
    """strip_control_chars should preserve tab, newline, and carriage return."""
    text = "line1\nline2\tdata\r\n"
    result = strip_control_chars(text)
    assert "\n" in result
    assert "\t" in result
