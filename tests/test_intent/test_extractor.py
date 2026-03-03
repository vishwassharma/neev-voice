"""Tests for intent extraction module."""

import json

import pytest

from neev_voice.config import NeevSettings
from neev_voice.intent.extractor import (
    ExtractedIntent,
    IntentCategory,
    IntentExtractor,
)
from neev_voice.llm.agent import EnrichmentAgent


@pytest.fixture
def settings():
    """Create test settings."""
    return NeevSettings(sarvam_api_key="test", anthropic_api_key="test-key")


@pytest.fixture
def mock_agent(settings, mocker):
    """Create a mocked EnrichmentAgent."""
    agent = EnrichmentAgent(settings)
    agent.enrich = mocker.AsyncMock()
    return agent


@pytest.fixture
def extractor(mock_agent):
    """Create an IntentExtractor with mocked agent."""
    return IntentExtractor(mock_agent)


class TestIntentCategory:
    """Tests for IntentCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories are defined."""
        expected = [
            "problem_statement",
            "solution",
            "clue",
            "mixed",
            "agreement",
            "disagreement",
            "question",
        ]
        for cat in expected:
            assert IntentCategory(cat)

    def test_string_value(self):
        """Test enum values are strings."""
        assert IntentCategory.PROBLEM_STATEMENT == "problem_statement"
        assert isinstance(IntentCategory.AGREEMENT, str)


class TestExtractedIntent:
    """Tests for ExtractedIntent dataclass."""

    def test_creation(self):
        """Test ExtractedIntent creation."""
        intent = ExtractedIntent(
            category=IntentCategory.PROBLEM_STATEMENT,
            summary="There is a bug in the login flow",
            key_points=["login fails", "error 500"],
            raw_text="login mein bug hai, error 500 aa raha hai",
        )
        assert intent.category == IntentCategory.PROBLEM_STATEMENT
        assert len(intent.key_points) == 2
        assert intent.raw_text


class TestParseIntentResponse:
    """Tests for _parse_intent_response static method."""

    def test_parse_clean_json(self):
        """Test parsing clean JSON response."""
        response = json.dumps(
            {
                "category": "problem_statement",
                "summary": "Login is broken",
                "key_points": ["error 500", "login page"],
            }
        )
        result = IntentExtractor._parse_intent_response(response, "raw text")
        assert result.category == IntentCategory.PROBLEM_STATEMENT
        assert result.summary == "Login is broken"
        assert len(result.key_points) == 2
        assert result.raw_text == "raw text"

    def test_parse_markdown_fenced_json(self):
        """Test parsing JSON wrapped in markdown code fences."""
        inner = json.dumps(
            {
                "category": "solution",
                "summary": "Use caching",
                "key_points": ["add redis"],
            }
        )
        response = f"```json\n{inner}\n```"
        result = IntentExtractor._parse_intent_response(response, "raw")
        assert result.category == IntentCategory.SOLUTION

    def test_parse_unknown_category_defaults_to_mixed(self):
        """Test unknown category defaults to MIXED."""
        response = json.dumps(
            {
                "category": "unknown_type",
                "summary": "Something",
                "key_points": [],
            }
        )
        result = IntentExtractor._parse_intent_response(response, "raw")
        assert result.category == IntentCategory.MIXED

    def test_parse_missing_fields(self):
        """Test parsing response with missing optional fields."""
        response = json.dumps({"category": "agreement"})
        result = IntentExtractor._parse_intent_response(response, "raw")
        assert result.category == IntentCategory.AGREEMENT
        assert result.summary == ""
        assert result.key_points == []

    def test_parse_invalid_json_raises(self):
        """Test parsing invalid JSON raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to parse"):
            IntentExtractor._parse_intent_response("not json at all", "raw")

    def test_parse_agreement(self):
        """Test parsing agreement intent."""
        response = json.dumps(
            {
                "category": "agreement",
                "summary": "User agrees with the approach",
                "key_points": ["haan theek hai"],
            }
        )
        result = IntentExtractor._parse_intent_response(response, "haan theek hai")
        assert result.category == IntentCategory.AGREEMENT

    def test_parse_disagreement(self):
        """Test parsing disagreement intent."""
        response = json.dumps(
            {
                "category": "disagreement",
                "summary": "User disagrees with the design",
                "key_points": ["nahi yaar", "galat hai ye"],
            }
        )
        result = IntentExtractor._parse_intent_response(response, "nahi yaar galat hai ye")
        assert result.category == IntentCategory.DISAGREEMENT


class TestExtract:
    """Tests for the extract method."""

    async def test_extract_problem_statement(self, extractor, mock_agent):
        """Test extracting a problem statement intent."""
        mock_agent.enrich.return_value = json.dumps(
            {
                "category": "problem_statement",
                "summary": "Login page is broken",
                "key_points": ["500 error", "login form"],
            }
        )

        result = await extractor.extract("login page pe 500 error aa raha hai")
        assert result.category == IntentCategory.PROBLEM_STATEMENT
        assert result.raw_text == "login page pe 500 error aa raha hai"
        mock_agent.enrich.assert_called_once()

    async def test_extract_solution(self, extractor, mock_agent):
        """Test extracting a solution intent."""
        mock_agent.enrich.return_value = json.dumps(
            {
                "category": "solution",
                "summary": "Add caching to reduce latency",
                "key_points": ["redis cache", "5 min TTL"],
            }
        )

        result = await extractor.extract("redis cache laga do with 5 minute TTL")
        assert result.category == IntentCategory.SOLUTION

    async def test_extract_question(self, extractor, mock_agent):
        """Test extracting a question intent."""
        mock_agent.enrich.return_value = json.dumps(
            {
                "category": "question",
                "summary": "Asking about authentication method",
                "key_points": ["OAuth vs JWT"],
            }
        )

        result = await extractor.extract("authentication ke liye OAuth use karein ya JWT?")
        assert result.category == IntentCategory.QUESTION


class TestExtractDiscussionIntent:
    """Tests for the extract_discussion_intent method."""

    async def test_agreement_detection(self, extractor, mock_agent):
        """Test detecting agreement in discussion context."""
        mock_agent.enrich.return_value = json.dumps(
            {
                "category": "agreement",
                "summary": "User agrees with the architecture",
                "key_points": ["agrees with microservices approach"],
            }
        )

        result = await extractor.extract_discussion_intent(
            "haan theek hai, microservices approach sahi hai",
            "We propose a microservices architecture",
        )
        assert result.category == IntentCategory.AGREEMENT

    async def test_disagreement_detection(self, extractor, mock_agent):
        """Test detecting disagreement in discussion context."""
        mock_agent.enrich.return_value = json.dumps(
            {
                "category": "disagreement",
                "summary": "User prefers monolith",
                "key_points": ["too complex", "prefers monolith"],
            }
        )

        result = await extractor.extract_discussion_intent(
            "nahi yaar, ye bahut complex hoga, monolith better hai",
            "We propose a microservices architecture",
        )
        assert result.category == IntentCategory.DISAGREEMENT

    async def test_discussion_prompt_includes_section(self, extractor, mock_agent):
        """Test that the discussion prompt includes the section text."""
        mock_agent.enrich.return_value = json.dumps(
            {
                "category": "agreement",
                "summary": "ok",
                "key_points": [],
            }
        )

        await extractor.extract_discussion_intent("ok", "some section text")

        call_args = mock_agent.enrich.call_args[0][0]
        assert "some section text" in call_args
