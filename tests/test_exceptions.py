"""Unit tests for BallotBox AI custom exceptions module."""

from exceptions import (
    AIParseError,
    AIServiceError,
    BallotBoxError,
    ConfigurationError,
    DataLoadError,
    InputValidationError,
    RateLimitExceededError,
)


def test_base_exception_hierarchy():
    """All custom exceptions should inherit from BallotBoxError."""
    assert issubclass(AIServiceError, BallotBoxError)
    assert issubclass(AIParseError, BallotBoxError)
    assert issubclass(ConfigurationError, BallotBoxError)
    assert issubclass(RateLimitExceededError, BallotBoxError)
    assert issubclass(DataLoadError, BallotBoxError)
    assert issubclass(InputValidationError, BallotBoxError)


def test_ballotbox_error_message():
    """BallotBoxError should store message and detail."""
    exc = BallotBoxError("test message", detail={"key": "value"})
    assert exc.message == "test message"
    assert exc.detail == {"key": "value"}
    assert str(exc) == "test message"


def test_ai_service_error():
    """AIServiceError should be raisable with message and detail."""
    exc = AIServiceError("service failed", detail={"status": 500})
    assert exc.message == "service failed"
    assert exc.detail["status"] == 500


def test_ai_parse_error():
    """AIParseError should be raisable with message and detail."""
    exc = AIParseError("parse failed")
    assert exc.message == "parse failed"
    assert exc.detail is not None


def test_configuration_error():
    """ConfigurationError should be raisable with message."""
    exc = ConfigurationError("missing config")
    assert exc.message == "missing config"


def test_rate_limit_error():
    """RateLimitExceededError should be raisable."""
    exc = RateLimitExceededError("too many requests")
    assert isinstance(exc, BallotBoxError)


def test_data_load_error():
    """DataLoadError should be raisable."""
    exc = DataLoadError("file not found", detail={"path": "/data/foo.json"})
    assert exc.detail["path"] == "/data/foo.json"


def test_input_validation_error():
    """InputValidationError should be raisable."""
    exc = InputValidationError("invalid input")
    assert isinstance(exc, BallotBoxError)


def test_exceptions_are_catchable_as_base():
    """All custom exceptions should be catchable as BallotBoxError."""
    exceptions = [
        AIServiceError("test"),
        AIParseError("test"),
        ConfigurationError("test"),
        RateLimitExceededError("test"),
        DataLoadError("test"),
        InputValidationError("test"),
    ]
    for exc in exceptions:
        try:
            raise exc
        except BallotBoxError as caught:
            assert caught.message == "test"
