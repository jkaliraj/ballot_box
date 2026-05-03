"""Unit tests for BallotBox Google Cloud service integrations."""

import logging
import os
from unittest.mock import patch

import pytest

from services.google_cloud import (
    get_cloud_run_metadata,
    log_structured,
    setup_cloud_logging,
)


def test_get_cloud_run_metadata_defaults():
    """Metadata should return 'local' defaults when env vars are not set."""
    with patch.dict(os.environ, {}, clear=True):
        meta = get_cloud_run_metadata()
    assert meta["service"] == "local"
    assert meta["revision"] == "local"
    assert meta["configuration"] == "local"


def test_get_cloud_run_metadata_from_env():
    """Metadata should read from Cloud Run environment variables."""
    env = {
        "K_SERVICE": "ballotbox-ai",
        "K_REVISION": "ballotbox-ai-00001",
        "K_CONFIGURATION": "ballotbox-ai",
        "GOOGLE_CLOUD_PROJECT": "nextmove-agent",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
    }
    with patch.dict(os.environ, env, clear=True):
        meta = get_cloud_run_metadata()
    assert meta["service"] == "ballotbox-ai"
    assert meta["revision"] == "ballotbox-ai-00001"
    assert meta["project_id"] == "nextmove-agent"
    assert meta["region"] == "us-central1"


def test_setup_cloud_logging_fallback():
    """setup_cloud_logging should fall back to standard logging gracefully."""
    # When google.cloud.logging is not importable, it should not raise
    with patch.dict("sys.modules", {"google.cloud.logging": None}):
        setup_cloud_logging("INFO")
    # No assertion needed — just verify no exception raised


def test_log_structured_emits_message():
    """log_structured should log a message with structured extras."""
    test_logger = logging.getLogger("test.structured")
    with patch.object(test_logger, "log") as mock_log:
        log_structured(test_logger, logging.INFO, "test message", user="bob")
        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        assert args[0] == logging.INFO
        assert args[1] == "test message"
        assert "json_fields" in kwargs.get("extra", {})
        assert kwargs["extra"]["json_fields"]["user"] == "bob"


def test_log_structured_no_kwargs():
    """log_structured without extra kwargs should have empty extra."""
    test_logger = logging.getLogger("test.structured.plain")
    with patch.object(test_logger, "log") as mock_log:
        log_structured(test_logger, logging.WARNING, "plain message")
        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        assert args[1] == "plain message"
