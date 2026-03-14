"""Tests for the prepare-enquiry engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neev_voice.discuss.prepare_enquiry import PrepareEnquiryEngine
from neev_voice.discuss.session import SessionInfo


@pytest.fixture
def session(tmp_path: Path) -> SessionInfo:
    """Create a test session."""
    return SessionInfo(
        name="test-session",
        research_path=str(tmp_path / "research"),
        source_path=str(tmp_path / "source"),
        output_path=str(tmp_path / "output"),
    )


@pytest.fixture
def settings() -> MagicMock:
    """Create mock settings."""
    s = MagicMock()
    s.claude_model = "sonnet"
    s.resolved_llm_api_key = "test-key"
    s.resolved_llm_api_base = ""
    s.resolved_mcp_config = "/mock/mcp.json"
    s.resolved_discuss_model = "sonnet"
    s.discuss_base_dir = ".scratch/neev/discuss"
    return s


class TestPrepareEnquiryEngine:
    """Tests for PrepareEnquiryEngine."""

    def test_init(self, session: SessionInfo, settings: MagicMock, tmp_path: Path) -> None:
        """Engine initializes correctly."""
        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        assert engine.session is session
        assert engine.session_dir == tmp_path

    def test_parse_response_with_sections(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_parse_response extracts answer and transcript sections."""
        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        response = """## Answer
The answer is 42.

## Transcript
The answer to your question is forty-two."""

        answer, transcript = engine._parse_response(response)
        assert "42" in answer
        assert "forty-two" in transcript

    def test_parse_response_no_sections(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_parse_response falls back to full response."""
        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        response = "Just a plain answer without sections."
        answer, transcript = engine._parse_response(response)
        assert answer == response.strip()
        assert transcript == answer

    def test_parse_response_answer_only(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_parse_response uses answer as transcript fallback."""
        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        response = """## Answer
Only an answer section, no transcript."""

        answer, transcript = engine._parse_response(response)
        assert "Only an answer" in answer
        assert transcript == answer

    def test_save_enquiry(self, session: SessionInfo, settings: MagicMock, tmp_path: Path) -> None:
        """_save_enquiry creates files in enquiry directory."""
        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        engine._save_enquiry("What is X?", "X is a pattern.", "X is a design pattern.")

        enquiry_dirs = list((tmp_path / "enquiries").iterdir())
        assert len(enquiry_dirs) == 1

        enquiry_dir = enquiry_dirs[0]
        assert (enquiry_dir / "query.txt").read_text() == "What is X?"
        assert (enquiry_dir / "answer.md").read_text() == "X is a pattern."
        assert (enquiry_dir / "transcript.md").read_text() == "X is a design pattern."

        metadata = json.loads((enquiry_dir / "metadata.json").read_text())
        assert "timestamp" in metadata
        assert metadata["query_length"] == len("What is X?")


class TestPrepareEnquiryEngineRun:
    """Tests for run() method."""

    @patch.object(PrepareEnquiryEngine, "_run_claude")
    async def test_run_new_enquiry(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() generates answer for new enquiry."""
        mock_claude.return_value = """## Answer
DI is a pattern where dependencies are injected.

## Transcript
Dependency injection is a design pattern where dependencies are provided externally."""

        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        transcript = await engine.run("What is dependency injection?")

        assert "dependency injection" in transcript.lower()
        mock_claude.assert_called_once()

        # Verify the prompt used research path
        prompt_arg = mock_claude.call_args[0][0]
        assert session.research_path in prompt_arg

    @patch.object(PrepareEnquiryEngine, "_run_claude")
    async def test_run_followup_enquiry(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() updates previous answer for follow-up enquiry."""
        mock_claude.return_value = """## Answer
Updated: DI uses constructor injection mainly.

## Transcript
The primary method of dependency injection is constructor injection."""

        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        transcript = await engine.run(
            "How exactly is DI implemented?",
            from_presentation_enquiry=True,
            previous_answer="DI is a pattern.",
        )

        assert "constructor injection" in transcript.lower()

        # Verify the update prompt was used
        prompt_arg = mock_claude.call_args[0][0]
        assert "Previous answer" in prompt_arg
        assert "DI is a pattern." in prompt_arg

    @patch.object(PrepareEnquiryEngine, "_run_claude")
    async def test_run_saves_artifacts(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() saves all enquiry artifacts."""
        mock_claude.return_value = "## Answer\nThe answer.\n\n## Transcript\nSpoken answer."

        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        await engine.run("Test query")

        enquiry_dirs = list((tmp_path / "enquiries").iterdir())
        assert len(enquiry_dirs) == 1

        d = enquiry_dirs[0]
        assert (d / "query.txt").exists()
        assert (d / "answer.md").exists()
        assert (d / "transcript.md").exists()
        assert (d / "metadata.json").exists()

    @patch.object(PrepareEnquiryEngine, "_run_claude")
    async def test_run_empty_response(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() handles empty Claude response."""
        mock_claude.return_value = ""

        engine = PrepareEnquiryEngine(session, settings, session_dir=tmp_path)
        transcript = await engine.run("Question?")
        assert transcript == ""  # Empty response parsed as empty
