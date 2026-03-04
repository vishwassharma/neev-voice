"""Tests for IntentClassifier lightweight intent classification.

Tests the IntentClassifier class that uses the claude CLI subprocess
for single-turn intent classification without codebase exploration.
"""

import json
from unittest.mock import AsyncMock

import pytest

from neev_voice.config import NeevSettings
from neev_voice.exceptions import NeevLLMError
from neev_voice.intent.classifier import IntentClassifier
from neev_voice.intent.extractor import IntentCategory


def _make_mock_process(stdout_text: str, returncode: int = 0) -> AsyncMock:
    """Create a mock asyncio subprocess with given stdout output.

    Args:
        stdout_text: Text to return as stdout.
        returncode: Process return code.

    Returns:
        AsyncMock configured as a subprocess process.
    """
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        stdout_text.encode("utf-8"),
        b"",
    )
    mock_process.returncode = returncode
    return mock_process


class TestIntentClassifierInit:
    """Tests for IntentClassifier initialization."""

    def test_stores_settings(self):
        """Test classifier stores settings on init."""
        settings = NeevSettings(sarvam_api_key="test", anthropic_api_key="key")
        classifier = IntentClassifier(settings)
        assert classifier.settings is settings

    def test_default_model_from_settings(self):
        """Test classifier uses model from settings."""
        settings = NeevSettings(sarvam_api_key="test", anthropic_api_key="key")
        classifier = IntentClassifier(settings)
        assert classifier.settings.claude_model == settings.claude_model


class TestIntentClassifierClassify:
    """Tests for IntentClassifier.classify() method."""

    @pytest.fixture
    def settings(self):
        """Create test settings with API key."""
        return NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="test-key",
        )

    @pytest.fixture
    def settings_no_key(self, monkeypatch):
        """Create test settings without API key."""
        monkeypatch.delenv("NEEV_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("NEEV_OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        return NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="",
            _env_file=None,
        )

    async def test_classify_works_without_api_key(self, settings_no_key, mocker):
        """Test classify works without API key (claude CLI manages its own auth)."""
        classifier = IntentClassifier(settings_no_key)
        response = json.dumps(
            {
                "category": "question",
                "summary": "A question",
                "key_points": [],
            }
        )

        mock_subprocess = mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response),
        )

        result = await classifier.classify("some text")
        assert result.category == IntentCategory.QUESTION

        # API key should NOT be in env when not configured
        call_kwargs = mock_subprocess.call_args.kwargs
        assert "ANTHROPIC_API_KEY" not in call_kwargs["env"]

    async def test_classify_returns_extracted_intent(self, settings, mocker):
        """Test classify returns a valid ExtractedIntent."""
        classifier = IntentClassifier(settings)
        response = json.dumps(
            {
                "category": "problem_statement",
                "summary": "User describes a bug",
                "key_points": ["crash on startup", "after update"],
            }
        )

        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response),
        )

        result = await classifier.classify("there is a bug after the update")
        assert result.category == IntentCategory.PROBLEM_STATEMENT
        assert "bug" in result.summary
        assert len(result.key_points) == 2
        assert result.raw_text == "there is a bug after the update"

    async def test_classify_handles_code_fenced_json(self, settings, mocker):
        """Test classify strips markdown code fences from response."""
        classifier = IntentClassifier(settings)
        response = (
            '```json\n{"category": "question", "summary": "Asking about auth",'
            ' "key_points": ["auth flow"]}\n```'
        )

        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response),
        )

        result = await classifier.classify("how does auth work?")
        assert result.category == IntentCategory.QUESTION

    async def test_classify_unknown_category_defaults_to_mixed(self, settings, mocker):
        """Test unknown category string defaults to MIXED."""
        classifier = IntentClassifier(settings)
        response = json.dumps(
            {
                "category": "unknown_type",
                "summary": "Some summary",
                "key_points": [],
            }
        )

        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response),
        )

        result = await classifier.classify("something")
        assert result.category == IntentCategory.MIXED

    async def test_classify_invalid_json_raises(self, settings, mocker):
        """Test classify raises NeevLLMError on invalid JSON response."""
        classifier = IntentClassifier(settings)

        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process("not valid json at all"),
        )

        with pytest.raises(NeevLLMError, match="Failed to parse"):
            await classifier.classify("test")

    async def test_classify_sets_api_key_in_env_when_available(self, settings, mocker):
        """Test classify passes ANTHROPIC_API_KEY when configured."""
        classifier = IntentClassifier(settings)
        response = json.dumps(
            {
                "category": "question",
                "summary": "A question",
                "key_points": [],
            }
        )

        mock_subprocess = mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response),
        )

        await classifier.classify("test")

        call_kwargs = mock_subprocess.call_args.kwargs
        assert call_kwargs["env"]["ANTHROPIC_API_KEY"] == "test-key"

    async def test_classify_uses_claude_cli(self, settings, mocker):
        """Test classify invokes the claude CLI command."""
        classifier = IntentClassifier(settings)
        response = json.dumps(
            {
                "category": "solution",
                "summary": "A solution",
                "key_points": [],
            }
        )

        mock_subprocess = mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response),
        )

        await classifier.classify("test")

        cmd_args = mock_subprocess.call_args.args
        assert cmd_args[0] == "claude"
        assert "--dangerously-skip-permissions" in cmd_args
        assert "--output-format" in cmd_args

    async def test_classify_pipes_prompt_via_stdin(self, settings, mocker):
        """Test classify pipes the intent prompt via stdin."""
        classifier = IntentClassifier(settings)
        response = json.dumps(
            {
                "category": "clue",
                "summary": "A clue",
                "key_points": [],
            }
        )

        mock_process = _make_mock_process(response)
        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )

        await classifier.classify("the config file is wrong")

        piped_input = mock_process.communicate.call_args.kwargs["input"].decode("utf-8")
        assert "the config file is wrong" in piped_input

    async def test_classify_handles_cli_error(self, settings, mocker):
        """Test classify raises NeevLLMError on non-zero exit with no output."""
        classifier = IntentClassifier(settings)

        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process("", returncode=1),
        )

        with pytest.raises(NeevLLMError, match="failed"):
            await classifier.classify("test")

    async def test_classify_discussion_intent(self, settings, mocker):
        """Test classify_discussion returns intent for discussion context."""
        classifier = IntentClassifier(settings)
        response = json.dumps(
            {
                "category": "agreement",
                "summary": "User agrees",
                "key_points": ["sounds good"],
            }
        )

        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response),
        )

        result = await classifier.classify_discussion("haan theek hai", "section content")
        assert result.category == IntentCategory.AGREEMENT

    async def test_classify_discussion_includes_section_in_prompt(self, settings, mocker):
        """Test discussion classification includes section text in prompt."""
        classifier = IntentClassifier(settings)
        response = json.dumps(
            {
                "category": "disagreement",
                "summary": "User disagrees",
                "key_points": [],
            }
        )

        mock_process = _make_mock_process(response)
        mocker.patch(
            "neev_voice.intent.classifier.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )

        await classifier.classify_discussion("nahi", "the proposed design")

        piped_input = mock_process.communicate.call_args.kwargs["input"].decode("utf-8")
        assert "nahi" in piped_input
        assert "the proposed design" in piped_input
