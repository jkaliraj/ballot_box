"""Unit tests for BallotBox Gemini AI module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.gemini import _clean_json, SYSTEM_INSTRUCTION
from exceptions import AIParseError, AIServiceError, ConfigurationError


class TestCleanJson:
    """Test suite for the _clean_json helper."""

    def test_plain_json(self):
        """Plain JSON string should be parsed correctly."""
        raw = '{"key": "value"}'
        assert _clean_json(raw) == {"key": "value"}

    def test_with_fences(self):
        """JSON wrapped in ```json fences should be extracted."""
        raw = '```json\n{"key": "value"}\n```'
        assert _clean_json(raw) == {"key": "value"}

    def test_with_fences_no_lang(self):
        """JSON wrapped in plain ``` fences should be extracted."""
        raw = '```\n[1, 2, 3]\n```'
        assert _clean_json(raw) == [1, 2, 3]

    def test_invalid_json_raises(self):
        """Non-JSON text should raise JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _clean_json("not json at all")

    def test_nested_objects(self):
        """Nested JSON structures should parse correctly."""
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = _clean_json(raw)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_array_of_objects(self):
        """Array of objects should parse correctly."""
        raw = '[{"a": 1}, {"b": 2}]'
        result = _clean_json(raw)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped before parsing."""
        raw = '  \n  {"key": "value"}  \n  '
        assert _clean_json(raw) == {"key": "value"}

    def test_empty_object(self):
        """Empty JSON object should parse correctly."""
        assert _clean_json("{}") == {}

    def test_empty_array(self):
        """Empty JSON array should parse correctly."""
        assert _clean_json("[]") == []


class TestSystemInstruction:
    """Test suite for the Gemini system instruction prompt."""

    def test_is_non_partisan(self):
        """System instruction should enforce non-partisan behavior."""
        assert "non-partisan" in SYSTEM_INSTRUCTION
        assert "neutral" in SYSTEM_INSTRUCTION

    def test_contains_guidelines(self):
        """System instruction should contain educational guidelines."""
        assert "factual" in SYSTEM_INSTRUCTION
        assert "civic participation" in SYSTEM_INSTRUCTION

    def test_prevents_party_endorsement(self):
        """System instruction should redirect party-specific questions."""
        assert "party" in SYSTEM_INSTRUCTION.lower()
        assert "redirect" in SYSTEM_INSTRUCTION.lower()

    def test_uses_structured_formatting(self):
        """System instruction should encourage structured responses."""
        assert "bullet points" in SYSTEM_INSTRUCTION
        assert "numbered steps" in SYSTEM_INSTRUCTION

    def test_covers_global_elections(self):
        """System instruction should support global scope."""
        assert "global" in SYSTEM_INSTRUCTION.lower()

    def test_accessible_language(self):
        """System instruction should request accessible language."""
        assert "accessible language" in SYSTEM_INSTRUCTION


# ── AI Function Tests (mocked Vertex AI) ──────────────────


@pytest.fixture(autouse=True)
def _reset_gemini_client():
    """Reset the module-level Gemini client before each test."""
    import ai.gemini as mod
    mod._client = None
    yield
    mod._client = None


def _mock_genai_response(text: str) -> MagicMock:
    """Build a mock Gemini generate_content response."""
    response = MagicMock()
    response.text = text
    return response


@pytest.mark.anyio
async def test_chat_returns_text():
    """chat() should return the Gemini response text."""
    from ai.gemini import chat

    mock_response = _mock_genai_response("Hello from BallotBox AI!")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai.gemini._get_client", return_value=mock_client):
        result = await chat("What is voting?")
    assert result == "Hello from BallotBox AI!"


@pytest.mark.anyio
async def test_chat_raises_on_error():
    """chat() should raise AIServiceError when Gemini fails."""
    from ai.gemini import chat

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("API down")

    with patch("ai.gemini._get_client", return_value=mock_client):
        with pytest.raises(AIServiceError):
            await chat("test")


@pytest.mark.anyio
async def test_generate_timeline_success():
    """generate_timeline() should return parsed JSON timeline."""
    from ai.gemini import generate_timeline

    timeline_data = [{"phase": "Registration", "timeframe": "Jan", "description": "Reg", "key_actions": []}]
    mock_response = _mock_genai_response(json.dumps(timeline_data))
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai.gemini._get_client", return_value=mock_client), \
         patch("ai.gemini.timeline_cache") as mock_cache:
        mock_cache.get.return_value = None
        result = await generate_timeline("India")
    assert len(result) == 1
    assert result[0]["phase"] == "Registration"


@pytest.mark.anyio
async def test_generate_timeline_cache_hit():
    """generate_timeline() should return cached data on cache hit."""
    from ai.gemini import generate_timeline

    cached_data = [{"phase": "Cached", "timeframe": "N/A", "description": "From cache", "key_actions": []}]
    with patch("ai.gemini.timeline_cache") as mock_cache:
        mock_cache.get.return_value = cached_data
        result = await generate_timeline("India")
    assert result[0]["phase"] == "Cached"


@pytest.mark.anyio
async def test_generate_timeline_raises_parse_error():
    """generate_timeline() should raise AIParseError on invalid JSON."""
    from ai.gemini import generate_timeline

    mock_response = _mock_genai_response("not valid json")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai.gemini._get_client", return_value=mock_client), \
         patch("ai.gemini.timeline_cache") as mock_cache:
        mock_cache.get.return_value = None
        with pytest.raises(AIParseError):
            await generate_timeline("India")


@pytest.mark.anyio
async def test_generate_timeline_raises_service_error():
    """generate_timeline() should raise AIServiceError on API failure."""
    from ai.gemini import generate_timeline

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("timeout")

    with patch("ai.gemini._get_client", return_value=mock_client), \
         patch("ai.gemini.timeline_cache") as mock_cache:
        mock_cache.get.return_value = None
        with pytest.raises(AIServiceError):
            await generate_timeline("India")


@pytest.mark.anyio
async def test_voter_readiness_check_success():
    """voter_readiness_check() should return parsed readiness data."""
    from ai.gemini import voter_readiness_check

    readiness = {"score": 80, "status": "ready", "summary": "Good", "action_items": [], "tips": []}
    mock_response = _mock_genai_response(json.dumps(readiness))
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai.gemini._get_client", return_value=mock_client):
        result = await voter_readiness_check({"registered": True})
    assert result["score"] == 80


@pytest.mark.anyio
async def test_voter_readiness_raises_on_bad_json():
    """voter_readiness_check() should raise AIParseError on bad JSON."""
    from ai.gemini import voter_readiness_check

    mock_response = _mock_genai_response("invalid")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai.gemini._get_client", return_value=mock_client):
        with pytest.raises(AIParseError):
            await voter_readiness_check({"registered": True})


@pytest.mark.anyio
async def test_explain_topic_success():
    """explain_topic() should return parsed topic data."""
    from ai.gemini import explain_topic

    topic_data = {"title": "EVM", "summary": "Electronic Voting Machine", "key_points": [], "related_topics": [], "did_you_know": ""}
    mock_response = _mock_genai_response(json.dumps(topic_data))
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai.gemini._get_client", return_value=mock_client), \
         patch("ai.gemini.topic_cache") as mock_cache:
        mock_cache.get.return_value = None
        result = await explain_topic("EVM")
    assert result["title"] == "EVM"


@pytest.mark.anyio
async def test_explain_topic_cache_hit():
    """explain_topic() should return cached data on cache hit."""
    from ai.gemini import explain_topic

    cached = {"title": "Cached", "summary": "From cache", "key_points": [], "related_topics": [], "did_you_know": ""}
    with patch("ai.gemini.topic_cache") as mock_cache:
        mock_cache.get.return_value = cached
        result = await explain_topic("test")
    assert result["title"] == "Cached"


@pytest.mark.anyio
async def test_get_client_raises_without_project():
    """_get_client() should raise ConfigurationError without project_id."""
    from ai.gemini import _get_client

    mock_settings = MagicMock()
    mock_settings.project_id = ""
    with patch("ai.gemini.get_settings", return_value=mock_settings):
        with pytest.raises(ConfigurationError):
            _get_client()
