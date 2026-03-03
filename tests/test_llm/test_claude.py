"""Tests for Claude Code CLI integration."""

import json

import pytest

from neev_voice.config import NeevSettings
from neev_voice.llm.claude import ClaudeCodeClient


@pytest.fixture
def settings():
    """Create test settings."""
    return NeevSettings(sarvam_api_key="test", claude_timeout=10)


@pytest.fixture
def client(settings):
    """Create a ClaudeCodeClient."""
    return ClaudeCodeClient(settings=settings)


class TestClaudeCodeClientInit:
    """Tests for ClaudeCodeClient initialization."""

    def test_init(self, settings):
        """Test client initializes with settings."""
        client = ClaudeCodeClient(settings)
        assert client.settings.claude_timeout == 10


class TestParseResponse:
    """Tests for response parsing."""

    def test_parse_json_result_field(self):
        """Test parsing JSON with result field."""
        data = json.dumps({"result": "Hello world"})
        assert ClaudeCodeClient._parse_response(data) == "Hello world"

    def test_parse_json_text_field(self):
        """Test parsing JSON with text field."""
        data = json.dumps({"text": "Hello world"})
        assert ClaudeCodeClient._parse_response(data) == "Hello world"

    def test_parse_json_content_blocks(self):
        """Test parsing JSON array of content blocks."""
        data = json.dumps(
            [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        )
        assert ClaudeCodeClient._parse_response(data) == "Hello\nWorld"

    def test_parse_plain_text(self):
        """Test parsing plain text fallback."""
        assert ClaudeCodeClient._parse_response("just plain text") == "just plain text"

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        assert ClaudeCodeClient._parse_response("") == ""

    def test_parse_dict_fallback(self):
        """Test parsing dict without standard fields."""
        data = json.dumps({"foo": "bar"})
        result = ClaudeCodeClient._parse_response(data)
        assert "foo" in result or "bar" in result

    def test_parse_empty_content_blocks(self):
        """Test parsing empty content block array."""
        data = json.dumps([{"type": "image", "url": "test.png"}])
        result = ClaudeCodeClient._parse_response(data)
        assert result  # Should return string representation


class TestQuery:
    """Tests for the query method."""

    async def test_query_success(self, client, mocker):
        """Test successful query returns parsed response."""
        mock_process = mocker.AsyncMock()
        mock_process.communicate = mocker.AsyncMock(
            return_value=(
                json.dumps({"result": "The answer is 42"}).encode(),
                b"",
            )
        )
        mock_process.returncode = 0

        mocker.patch(
            "neev_voice.llm.claude.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )

        result = await client.query("What is the meaning of life?")
        assert result == "The answer is 42"

    async def test_query_with_context(self, client, mocker):
        """Test query prepends context to prompt."""
        mock_process = mocker.AsyncMock()
        mock_process.communicate = mocker.AsyncMock(
            return_value=(json.dumps({"result": "ok"}).encode(), b"")
        )
        mock_process.returncode = 0

        mock_exec = mocker.patch(
            "neev_voice.llm.claude.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )

        await client.query("question", context="some context")

        # Verify the prompt includes context
        call_args = mock_exec.call_args[0]
        prompt_arg = call_args[2]  # Third positional arg is the prompt
        assert "some context" in prompt_arg
        assert "question" in prompt_arg

    async def test_query_timeout(self, client, mocker):
        """Test query raises RuntimeError on timeout."""
        mock_process = mocker.AsyncMock()
        mock_process.communicate = mocker.AsyncMock(side_effect=TimeoutError())
        mock_process.kill = mocker.MagicMock()

        mocker.patch(
            "neev_voice.llm.claude.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )
        mocker.patch(
            "neev_voice.llm.claude.asyncio.wait_for",
            side_effect=TimeoutError(),
        )

        with pytest.raises(RuntimeError, match="timed out"):
            await client.query("slow question")

    async def test_query_cli_not_found(self, client, mocker):
        """Test query raises RuntimeError when CLI not installed."""
        mocker.patch(
            "neev_voice.llm.claude.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError(),
        )

        with pytest.raises(RuntimeError, match="Claude CLI not found"):
            await client.query("test")

    async def test_query_nonzero_exit(self, client, mocker):
        """Test query raises RuntimeError on non-zero exit code."""
        mock_process = mocker.AsyncMock()
        mock_process.communicate = mocker.AsyncMock(return_value=(b"", b"Some error occurred"))
        mock_process.returncode = 1

        mocker.patch(
            "neev_voice.llm.claude.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )

        with pytest.raises(RuntimeError, match="Claude CLI error"):
            await client.query("test")


class TestAnalyzeCodebase:
    """Tests for the analyze_codebase method."""

    async def test_analyze_missing_path(self, client, tmp_path):
        """Test analyze raises FileNotFoundError for missing path."""
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Codebase path not found"):
            await client.analyze_codebase(missing, "What is this?")

    async def test_analyze_calls_query(self, client, tmp_path, mocker):
        """Test analyze_codebase delegates to query with context."""
        codebase = tmp_path / "myproject"
        codebase.mkdir()

        mock_query = mocker.patch.object(
            client, "query", new=mocker.AsyncMock(return_value="analysis result")
        )

        result = await client.analyze_codebase(codebase, "Explain the architecture")
        assert result == "analysis result"
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args
        assert "Explain the architecture" in call_kwargs[0][0]
