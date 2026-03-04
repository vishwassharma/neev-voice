"""Tests for the transcript review gate module."""

from unittest.mock import MagicMock, patch

import pytest

from neev_voice.exceptions import TranscriptRejectedError
from neev_voice.review import TranscriptReviewAction, TranscriptReviewer


class TestTranscriptReviewAction:
    """Tests for the TranscriptReviewAction enum."""

    def test_values(self):
        """Test enum has accept, edit, reject values."""
        assert TranscriptReviewAction.ACCEPT == "accept"
        assert TranscriptReviewAction.EDIT == "edit"
        assert TranscriptReviewAction.REJECT == "reject"

    def test_is_str_enum(self):
        """Test enum members are strings."""
        assert isinstance(TranscriptReviewAction.ACCEPT, str)


class TestTranscriptReviewer:
    """Tests for the TranscriptReviewer class."""

    @pytest.fixture
    def reviewer(self):
        """Create a TranscriptReviewer with a mocked Console."""
        console = MagicMock()
        return TranscriptReviewer(console=console)

    @pytest.fixture
    def transcript_file(self, tmp_path):
        """Create a temporary transcript file."""
        path = tmp_path / "transcription.txt"
        path.write_text("Original transcript text")
        return path

    def test_default_console(self):
        """Test TranscriptReviewer creates a Console if none provided."""
        reviewer = TranscriptReviewer()
        assert reviewer._console is not None

    @pytest.mark.asyncio
    async def test_accept_returns_original_text(self, reviewer, transcript_file):
        """Test accept action returns the original transcript unchanged."""
        with patch.object(reviewer, "_prompt_action", return_value=TranscriptReviewAction.ACCEPT):
            action, text = await reviewer.review("Original text", transcript_file)

        assert action == TranscriptReviewAction.ACCEPT
        assert text == "Original text"

    @pytest.mark.asyncio
    async def test_edit_opens_editor_and_returns_modified_text(self, reviewer, transcript_file):
        """Test edit action opens $EDITOR and returns edited content."""
        transcript_file.write_text("Edited transcript text")

        with (
            patch.object(reviewer, "_prompt_action", return_value=TranscriptReviewAction.EDIT),
            patch("neev_voice.review.subprocess.run") as mock_run,
        ):
            action, text = await reviewer.review("Original text", transcript_file)

        assert action == TranscriptReviewAction.EDIT
        assert text == "Edited transcript text"
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_raises_error(self, reviewer, transcript_file):
        """Test reject action raises TranscriptRejectedError."""
        with (
            patch.object(reviewer, "_prompt_action", return_value=TranscriptReviewAction.REJECT),
            pytest.raises(TranscriptRejectedError, match="Transcript rejected"),
        ):
            await reviewer.review("Original text", transcript_file)

    def test_display_transcript_renders_panel(self, reviewer):
        """Test _display_transcript calls console.print with a Panel."""
        reviewer._display_transcript("Some text")
        assert reviewer._console.print.call_count == 2  # blank line + panel

    def test_prompt_action_accept(self, reviewer):
        """Test _prompt_action returns ACCEPT for 'a' input."""
        with patch("builtins.input", return_value="a"):
            result = reviewer._prompt_action()
        assert result == TranscriptReviewAction.ACCEPT

    def test_prompt_action_edit(self, reviewer):
        """Test _prompt_action returns EDIT for 'e' input."""
        with patch("builtins.input", return_value="e"):
            result = reviewer._prompt_action()
        assert result == TranscriptReviewAction.EDIT

    def test_prompt_action_reject(self, reviewer):
        """Test _prompt_action returns REJECT for 'r' input."""
        with patch("builtins.input", return_value="r"):
            result = reviewer._prompt_action()
        assert result == TranscriptReviewAction.REJECT

    def test_prompt_action_case_insensitive(self, reviewer):
        """Test _prompt_action accepts uppercase input."""
        with patch("builtins.input", return_value="A"):
            result = reviewer._prompt_action()
        assert result == TranscriptReviewAction.ACCEPT

    def test_prompt_action_retries_on_invalid(self, reviewer):
        """Test _prompt_action loops on invalid input then accepts valid."""
        with patch("builtins.input", side_effect=["x", "z", "a"]):
            result = reviewer._prompt_action()
        assert result == TranscriptReviewAction.ACCEPT

    def test_prompt_action_strips_whitespace(self, reviewer):
        """Test _prompt_action strips whitespace from input."""
        with patch("builtins.input", return_value="  e  "):
            result = reviewer._prompt_action()
        assert result == TranscriptReviewAction.EDIT

    @pytest.mark.asyncio
    async def test_open_editor_calls_subprocess(self, reviewer, transcript_file):
        """Test _open_editor runs the resolved editor on the file."""
        with (
            patch.object(reviewer, "_resolve_editor", return_value="nano"),
            patch("neev_voice.review.subprocess.run") as mock_run,
        ):
            result = await reviewer._open_editor(transcript_file)

        mock_run.assert_called_once_with(["nano", str(transcript_file)], check=True)
        assert result == "Original transcript text"

    @pytest.mark.asyncio
    async def test_open_editor_strips_trailing_whitespace(self, reviewer, tmp_path):
        """Test _open_editor strips trailing whitespace from file content."""
        path = tmp_path / "transcript.txt"
        path.write_text("  some text  \n\n")

        with (
            patch.object(reviewer, "_resolve_editor", return_value="vi"),
            patch("neev_voice.review.subprocess.run"),
        ):
            result = await reviewer._open_editor(path)

        assert result == "some text"


class TestResolveEditor:
    """Tests for editor resolution logic."""

    def test_visual_takes_priority(self):
        """Test $VISUAL is preferred over $EDITOR."""
        with patch.dict("os.environ", {"VISUAL": "code", "EDITOR": "vim"}):
            assert TranscriptReviewer._resolve_editor() == "code"

    def test_editor_used_when_no_visual(self):
        """Test $EDITOR is used when $VISUAL is not set."""
        with patch.dict("os.environ", {"EDITOR": "nano"}, clear=True):
            assert TranscriptReviewer._resolve_editor() == "nano"

    def test_fallback_to_vi(self):
        """Test falls back to vi when no env vars set."""
        with patch.dict("os.environ", {}, clear=True):
            assert TranscriptReviewer._resolve_editor() == "vi"

    def test_visual_empty_string_uses_editor(self):
        """Test empty $VISUAL falls through to $EDITOR."""
        with patch.dict("os.environ", {"VISUAL": "", "EDITOR": "emacs"}, clear=True):
            assert TranscriptReviewer._resolve_editor() == "emacs"
