"""Unit tests for the error reporting service."""

import importlib
from unittest.mock import MagicMock, patch

from services.error_reporting import report_error


def test_report_error_logs_fallback():
    """report_error should not raise even without the Cloud client."""
    report_error(
        ValueError("test error"),
        context={"path": "/api/test", "method": "POST"},
    )


def test_report_error_with_none_context():
    """report_error should handle None context."""
    report_error(RuntimeError("another test"), context=None)


def test_report_error_with_empty_context():
    """report_error should handle empty context dict."""
    report_error(RuntimeError("empty ctx"), context={})


def test_report_error_cloud_client_success():
    """report_error should use Cloud Error Reporting when available."""
    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.Client.return_value = mock_client

    with patch.dict("sys.modules", {"google.cloud.error_reporting": mock_module}):
        import services.error_reporting as mod

        importlib.reload(mod)
        mod.report_error(ValueError("cloud test"), context={"a": "b"})
        mock_client.report_exception.assert_called_once()

    importlib.reload(mod)


def test_report_error_cloud_client_raises():
    """report_error should handle Cloud client exceptions gracefully."""
    mock_module = MagicMock()
    mock_module.Client.side_effect = RuntimeError("cloud error")

    with patch.dict("sys.modules", {"google.cloud.error_reporting": mock_module}):
        import services.error_reporting as mod

        importlib.reload(mod)
        # Should not raise
        mod.report_error(ValueError("test"), context=None)

    importlib.reload(mod)
