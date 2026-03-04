"""Integration tests for CLI commands using Typer's CliRunner."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from neev_voice.cli import _build_discussion_result_md, _display_intent, _get_settings, app
from neev_voice.intent.extractor import ExtractedIntent, IntentCategory

runner = CliRunner()


class TestConfigCommand:
    """Tests for the config CLI command (sub-app)."""

    def test_config_displays_settings(self):
        """Test config command shows configuration table."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "STT Provider" in result.output
        assert "TTS Provider" in result.output
        assert "Sample Rate" in result.output

    def test_config_shows_api_key_status(self):
        """Test config shows whether API key is configured."""
        result = runner.invoke(app, ["config"])
        assert "Sarvam API Key" in result.output

    def test_config_shows_anthropic_key_status(self):
        """Test config shows Anthropic API key status."""
        result = runner.invoke(app, ["config"])
        assert "Anthropic API Key" in result.output

    def test_config_shows_llm_provider(self):
        """Test config shows LLM Provider row."""
        result = runner.invoke(app, ["config"])
        assert "LLM Provider" in result.output

    def test_config_shows_llm_api_base(self):
        """Test config shows LLM API Base row."""
        result = runner.invoke(app, ["config"])
        assert "LLM API Base" in result.output

    def test_config_shows_openrouter_key_status(self):
        """Test config shows OpenRouter API Key status."""
        result = runner.invoke(app, ["config"])
        assert "OpenRouter API Key" in result.output

    def test_config_shows_stt_max_audio_duration(self):
        """Test config shows STT max audio duration."""
        result = runner.invoke(app, ["config"])
        assert "STT Max Audio Duration" in result.output


class TestConfigSetCommand:
    """Tests for the config set CLI sub-command."""

    def test_config_set_valid_key(self, mocker):
        """Test config set prints success message for valid key."""
        mocker.patch("neev_voice.cli.update_config_value")
        result = runner.invoke(app, ["config", "set", "claude_model", "opus"])
        assert result.exit_code == 0
        assert "Set" in result.output

    def test_config_set_invalid_key(self):
        """Test config set rejects unknown setting key."""
        result = runner.invoke(app, ["config", "set", "bogus_key", "value"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_config_set_invalid_enum_value(self):
        """Test config set rejects invalid enum value."""
        result = runner.invoke(app, ["config", "set", "stt_mode", "badmode"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestConfigInitCommand:
    """Tests for the config init CLI sub-command."""

    def test_config_init_creates_file(self, mocker):
        """Test config init creates a default config file."""
        mocker.patch(
            "neev_voice.cli.create_default_config",
            return_value=Path("/fake/path/voice.json"),
        )
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        assert "Created config file" in result.output

    def test_config_init_existing_file_errors(self, mocker):
        """Test config init errors when file exists without --force."""
        mocker.patch(
            "neev_voice.cli.create_default_config",
            side_effect=FileExistsError(
                "Config file already exists: /fake. Use --force to overwrite."
            ),
        )
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_config_init_force_overwrites(self, mocker):
        """Test config init --force overwrites existing file."""
        mock_create = mocker.patch(
            "neev_voice.cli.create_default_config",
            return_value=Path("/fake/path/voice.json"),
        )
        result = runner.invoke(app, ["config", "init", "--force"])
        assert result.exit_code == 0
        mock_create.assert_called_once_with(force=True)

    def test_config_init_help(self):
        """Test config init --help shows description."""
        result = runner.invoke(app, ["config", "init", "--help"])
        assert result.exit_code == 0
        assert "default config file" in result.output.lower() or "force" in result.output.lower()


class TestConfigSetBlocksAPIKeys:
    """Tests for config set rejecting API key fields."""

    def test_config_set_rejects_sarvam_api_key(self):
        """Test config set rejects sarvam_api_key."""
        result = runner.invoke(app, ["config", "set", "sarvam_api_key", "sk-test"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_config_set_rejects_anthropic_api_key(self):
        """Test config set rejects anthropic_api_key."""
        result = runner.invoke(app, ["config", "set", "anthropic_api_key", "sk-ant"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_config_set_rejects_openrouter_api_key(self):
        """Test config set rejects openrouter_api_key."""
        result = runner.invoke(app, ["config", "set", "openrouter_api_key", "sk-or"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestConfigPathCommand:
    """Tests for the config path CLI sub-command."""

    def test_config_path_shows_path(self):
        """Test config path prints the config file path."""
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert "voice.json" in result.output


class TestProvidersCommand:
    """Tests for the providers CLI command."""

    def test_providers_lists_all(self):
        """Test providers command lists all providers."""
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "sarvam" in result.output
        assert "edge" in result.output

    def test_providers_shows_types(self):
        """Test providers command shows STT and TTS types."""
        result = runner.invoke(app, ["providers"])
        assert "STT" in result.output
        assert "TTS" in result.output

    def test_providers_shows_notes(self):
        """Test providers command shows API key requirements."""
        result = runner.invoke(app, ["providers"])
        assert "API_KEY" in result.output or "API key" in result.output


class TestHelpCommand:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Test main help shows all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "listen" in result.output
        assert "discuss" in result.output
        assert "config" in result.output
        assert "providers" in result.output

    def test_listen_help(self):
        """Test listen command help."""
        result = runner.invoke(app, ["listen", "--help"])
        assert result.exit_code == 0
        assert "Record audio" in result.output or "push-to-talk" in result.output.lower()

    def test_discuss_help(self):
        """Test discuss command help."""
        result = runner.invoke(app, ["discuss", "--help"])
        assert result.exit_code == 0
        assert "document" in result.output.lower()


class TestDisplayIntent:
    """Tests for _display_intent helper function."""

    def test_display_problem_statement(self, capsys):
        """Test displaying a problem statement intent."""
        intent = ExtractedIntent(
            category=IntentCategory.PROBLEM_STATEMENT,
            summary="Login page is broken",
            key_points=["error 500", "login form crashes"],
            raw_text="raw text",
        )
        _display_intent(intent)
        # No exception means it rendered successfully

    def test_display_agreement(self, capsys):
        """Test displaying an agreement intent."""
        intent = ExtractedIntent(
            category=IntentCategory.AGREEMENT,
            summary="User agrees",
            key_points=[],
            raw_text="haan",
        )
        _display_intent(intent)

    def test_display_all_categories(self):
        """Test display works for all categories."""
        for category in IntentCategory:
            intent = ExtractedIntent(
                category=category,
                summary=f"Test {category.value}",
                key_points=["point1"],
                raw_text="test",
            )
            _display_intent(intent)


class TestGetSettings:
    """Tests for _get_settings helper."""

    def test_returns_settings(self):
        """Test _get_settings returns a NeevSettings instance."""
        from neev_voice.config import NeevSettings

        settings = _get_settings()
        assert isinstance(settings, NeevSettings)


class TestListenCommand:
    """Tests for listen command async flow."""

    async def test_listen_async_success(self, mocker, tmp_path):
        """Test the async listen flow with mocked dependencies."""
        from neev_voice.audio.recorder import AudioSegment
        from neev_voice.cli import _listen_async
        from neev_voice.stt.base import TranscriptionResult

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        settings.stt_mode.value = "translate"
        settings.sarvam_api_key = "test"
        settings.anthropic_api_key = "test-key"
        settings.sample_rate = 16000
        settings.silence_threshold = 0.03
        settings.silence_duration = 1.5
        settings.key_release_timeout = 0.15
        settings.claude_timeout = 10

        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        mock_stt = AsyncMock()
        mock_stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hello world", language="en", confidence=0.9, provider="mock"
            )
        )
        mocker.patch("neev_voice.stt.sarvam.get_stt_provider", return_value=mock_stt)

        mock_segment = AudioSegment(
            data=np.zeros((16000, 1), dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
        )
        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(return_value=mock_segment)
        mocker.patch("neev_voice.audio.recorder.AudioRecorder", return_value=mock_recorder)
        mocker.patch(
            "neev_voice.audio.recorder.AudioRecorder.save_wav",
            return_value=tmp_path / "test.wav",
        )

        # Create a dummy wav file for shutil.copy2
        (tmp_path / "test.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        mock_scratch = MagicMock()
        mock_scratch.flow_dir = tmp_path / "scratch"
        mock_scratch.audio_path = tmp_path / "scratch" / "audio.wav"
        (tmp_path / "scratch").mkdir(parents=True, exist_ok=True)
        mocker.patch("neev_voice.scratch.ScratchPad", return_value=mock_scratch)

        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractedIntent(
                category=IntentCategory.QUESTION,
                summary="A question",
                key_points=["test"],
                raw_text="hello world",
            )
        )
        mocker.patch("neev_voice.intent.extractor.IntentExtractor", return_value=mock_extractor)
        mocker.patch("neev_voice.llm.agent.EnrichmentAgent")

        await _listen_async(None, None, mode=None, verbose=True, no_review=True)

        mock_scratch.save_transcription.assert_called_once_with("hello world")
        mock_scratch.save_metadata.assert_called_once()

    async def test_listen_async_invalid_provider(self, mocker):
        """Test listen with invalid STT provider."""
        from click.exceptions import Exit

        from neev_voice.cli import _listen_async

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        mocker.patch("neev_voice.cli._get_settings", return_value=settings)
        mocker.patch(
            "neev_voice.stt.sarvam.get_stt_provider",
            side_effect=ValueError("Unknown provider"),
        )

        with pytest.raises(Exit):
            await _listen_async("bad_provider", None, mode=None, verbose=False)


class TestListenCancellation:
    """Tests for listen command cancellation via ESC."""

    async def test_listen_async_cancelled(self, mocker, tmp_path):
        """Test listen gracefully handles RecordingCancelledError."""
        from neev_voice.audio.recorder import RecordingCancelledError
        from neev_voice.cli import _listen_async

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        settings.stt_mode.value = "translate"
        settings.sarvam_api_key = "test"
        settings.sample_rate = 16000
        settings.key_release_timeout = 0.15
        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        mocker.patch("neev_voice.stt.sarvam.get_stt_provider")

        mock_scratch = MagicMock()
        mock_scratch.flow_dir = tmp_path / "scratch"
        mocker.patch("neev_voice.scratch.ScratchPad", return_value=mock_scratch)

        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(
            side_effect=RecordingCancelledError("Recording cancelled by user")
        )
        mocker.patch("neev_voice.audio.recorder.AudioRecorder", return_value=mock_recorder)
        mocker.patch("neev_voice.llm.agent.EnrichmentAgent")
        mocker.patch("neev_voice.intent.extractor.IntentExtractor")

        from click.exceptions import Exit

        with pytest.raises(Exit) as exc_info:
            await _listen_async(None, None, mode=None, verbose=False)
        assert exc_info.value.exit_code == 0


class TestDiscussCommand:
    """Tests for discuss command async flow."""

    async def test_discuss_async_missing_doc(self, mocker, tmp_path):
        """Test discuss with missing document."""
        from click.exceptions import Exit

        from neev_voice.cli import _discuss_async

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        settings.tts_provider.value = "edge"
        settings.sarvam_api_key = "test"
        settings.anthropic_api_key = "test-key"
        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        mocker.patch("neev_voice.stt.sarvam.get_stt_provider")
        mocker.patch("neev_voice.tts.edge.get_tts_provider")
        mocker.patch("neev_voice.audio.recorder.AudioRecorder")
        mocker.patch("neev_voice.llm.agent.EnrichmentAgent")
        mocker.patch("neev_voice.intent.extractor.IntentExtractor")

        mock_scratch = MagicMock()
        mock_scratch.flow_dir = tmp_path / "scratch"
        mocker.patch("neev_voice.scratch.ScratchPad", return_value=mock_scratch)
        mocker.patch("neev_voice.scratch.ScratchPad.get_latest_folder", return_value=None)

        mock_manager = MagicMock()
        mock_manager.run_discussion = AsyncMock(side_effect=FileNotFoundError("not found"))
        mocker.patch(
            "neev_voice.discussion.manager.DiscussionManager",
            return_value=mock_manager,
        )

        with pytest.raises(Exit):
            await _discuss_async(str(tmp_path / "missing.md"), None, None, None, False)

    async def test_discuss_async_success(self, mocker, tmp_path):
        """Test successful discuss flow with scratch pad integration."""
        from neev_voice.cli import _discuss_async
        from neev_voice.discussion.manager import DiscussionResult

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        settings.tts_provider.value = "edge"
        settings.sarvam_api_key = "test"
        settings.anthropic_api_key = "test-key"
        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        mocker.patch("neev_voice.stt.sarvam.get_stt_provider")
        mocker.patch("neev_voice.tts.edge.get_tts_provider")
        mocker.patch("neev_voice.audio.recorder.AudioRecorder")
        mocker.patch("neev_voice.llm.agent.EnrichmentAgent")
        mocker.patch("neev_voice.intent.extractor.IntentExtractor")

        mock_scratch = MagicMock()
        mock_scratch.flow_dir = tmp_path / "scratch"
        mocker.patch("neev_voice.scratch.ScratchPad", return_value=mock_scratch)
        mocker.patch("neev_voice.scratch.ScratchPad.get_latest_folder", return_value=None)

        mock_manager = MagicMock()
        mock_manager.run_discussion = AsyncMock(
            return_value=[
                DiscussionResult(
                    section="## Intro",
                    user_response="yes",
                    intent=IntentCategory.AGREEMENT,
                    summary="agrees",
                ),
                DiscussionResult(
                    section="## Design",
                    user_response="no",
                    intent=IntentCategory.DISAGREEMENT,
                    summary="disagrees",
                ),
            ]
        )
        mocker.patch(
            "neev_voice.discussion.manager.DiscussionManager",
            return_value=mock_manager,
        )

        doc = tmp_path / "test.md"
        doc.write_text("test")
        await _discuss_async(str(doc), None, None, None, verbose=True)

        # Verify scratch pad methods were called
        assert mock_scratch.save_section.call_count == 2
        mock_scratch.save_summary.assert_called_once()
        mock_scratch.save_metadata.assert_called_once()
        mock_scratch.save_discussion_result.assert_called_once()

    async def test_discuss_reads_latest_listen_context(self, mocker, tmp_path):
        """Test discuss reads transcription from latest listen folder."""
        from neev_voice.cli import _discuss_async
        from neev_voice.discussion.manager import DiscussionResult

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        settings.tts_provider.value = "edge"
        settings.sarvam_api_key = "test"
        settings.anthropic_api_key = "test-key"
        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        mocker.patch("neev_voice.stt.sarvam.get_stt_provider")
        mocker.patch("neev_voice.tts.edge.get_tts_provider")
        mocker.patch("neev_voice.audio.recorder.AudioRecorder")
        mocker.patch("neev_voice.llm.agent.EnrichmentAgent")
        mocker.patch("neev_voice.intent.extractor.IntentExtractor")

        mock_scratch = MagicMock()
        mock_scratch.flow_dir = tmp_path / "scratch"
        mocker.patch("neev_voice.scratch.ScratchPad", return_value=mock_scratch)

        # Create a latest listen folder with transcription
        listen_dir = tmp_path / "listen"
        listen_dir.mkdir()
        transcription_file = listen_dir / "transcription.txt"
        transcription_file.write_text("previous transcription")
        mocker.patch(
            "neev_voice.scratch.ScratchPad.get_latest_folder",
            return_value=listen_dir,
        )

        mock_manager = MagicMock()
        mock_manager.run_discussion = AsyncMock(
            return_value=[
                DiscussionResult(
                    section="## Intro",
                    user_response="yes",
                    intent=IntentCategory.AGREEMENT,
                    summary="agrees",
                ),
            ]
        )
        mocker.patch(
            "neev_voice.discussion.manager.DiscussionManager",
            return_value=mock_manager,
        )

        doc = tmp_path / "test.md"
        doc.write_text("test")
        await _discuss_async(str(doc), None, None, None, False)


class TestListenWithMode:
    """Tests for listen command with --mode option."""

    async def test_listen_async_with_codemix_mode(self, mocker, tmp_path):
        """Test listen with explicit codemix mode."""
        from neev_voice.audio.recorder import AudioSegment
        from neev_voice.cli import _listen_async
        from neev_voice.stt.base import TranscriptionResult

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        settings.stt_mode.value = "translate"
        settings.sarvam_api_key = "test"
        settings.anthropic_api_key = "test-key"
        settings.sample_rate = 16000
        settings.key_release_timeout = 0.15
        settings.claude_timeout = 10

        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        mock_stt = AsyncMock()
        mock_stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hello", language="en", confidence=0.9, provider="mock"
            )
        )
        mocker.patch("neev_voice.stt.sarvam.get_stt_provider", return_value=mock_stt)

        mock_segment = AudioSegment(
            data=np.zeros((16000, 1), dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
        )
        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(return_value=mock_segment)
        mocker.patch("neev_voice.audio.recorder.AudioRecorder", return_value=mock_recorder)
        mocker.patch(
            "neev_voice.audio.recorder.AudioRecorder.save_wav", return_value=tmp_path / "t.wav"
        )

        # Create a dummy wav file
        (tmp_path / "t.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        mock_scratch = MagicMock()
        mock_scratch.flow_dir = tmp_path / "scratch"
        mock_scratch.audio_path = tmp_path / "scratch" / "audio.wav"
        (tmp_path / "scratch").mkdir(parents=True, exist_ok=True)
        mocker.patch("neev_voice.scratch.ScratchPad", return_value=mock_scratch)

        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractedIntent(
                category=IntentCategory.QUESTION,
                summary="Q",
                key_points=[],
                raw_text="hello",
            )
        )
        mocker.patch("neev_voice.intent.extractor.IntentExtractor", return_value=mock_extractor)
        mocker.patch("neev_voice.llm.agent.EnrichmentAgent")

        await _listen_async(None, None, mode="codemix", verbose=False, no_review=True)

    async def test_listen_async_invalid_mode(self, mocker):
        """Test listen with invalid STT mode exits with error."""
        from click.exceptions import Exit

        from neev_voice.cli import _listen_async

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        with pytest.raises(Exit):
            await _listen_async(None, None, mode="badmode", verbose=False)

    def test_listen_help_shows_mode_option(self):
        """Test listen --help mentions the --mode option."""
        result = runner.invoke(app, ["listen", "--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes for CI where Rich renders styled output
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--mode" in plain

    def test_discuss_help_shows_mode_option(self):
        """Test discuss --help mentions the --mode option."""
        result = runner.invoke(app, ["discuss", "--help"])
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--mode" in plain


class TestBuildDiscussionResultMd:
    """Tests for _build_discussion_result_md helper."""

    def test_contains_document_name(self):
        """Test result md contains the document filename."""
        from neev_voice.discussion.manager import DiscussionResult

        results = [
            DiscussionResult(
                section="## Intro\nContent",
                user_response="yes",
                intent=IntentCategory.AGREEMENT,
                summary="User agrees",
            ),
        ]
        md = _build_discussion_result_md(Path("plan.md"), results, 1, 0)
        assert "plan.md" in md

    def test_contains_agreements_section(self):
        """Test result md contains Agreements section."""
        from neev_voice.discussion.manager import DiscussionResult

        results = [
            DiscussionResult(
                section="## Intro",
                user_response="yes",
                intent=IntentCategory.AGREEMENT,
                summary="agrees",
            ),
        ]
        md = _build_discussion_result_md(Path("doc.md"), results, 1, 0)
        assert "## Agreements" in md

    def test_contains_disagreements_section(self):
        """Test result md contains Disagreements section."""
        from neev_voice.discussion.manager import DiscussionResult

        results = [
            DiscussionResult(
                section="## Design",
                user_response="no",
                intent=IntentCategory.DISAGREEMENT,
                summary="disagrees",
            ),
        ]
        md = _build_discussion_result_md(Path("doc.md"), results, 0, 1)
        assert "## Disagreements" in md

    def test_contains_section_details(self):
        """Test result md contains Section Details with all sections."""
        from neev_voice.discussion.manager import DiscussionResult

        results = [
            DiscussionResult(
                section="## Part 1",
                user_response="ok",
                intent=IntentCategory.AGREEMENT,
                summary="agrees",
            ),
            DiscussionResult(
                section="## Part 2",
                user_response="hmm",
                intent=IntentCategory.QUESTION,
                summary="has question",
            ),
        ]
        md = _build_discussion_result_md(Path("doc.md"), results, 1, 0)
        assert "## Section Details" in md
        assert "## Part 1" in md
        assert "## Part 2" in md

    def test_explorations_section_for_non_agree_disagree(self):
        """Test result md has Explorations section for other intents."""
        from neev_voice.discussion.manager import DiscussionResult

        results = [
            DiscussionResult(
                section="## Idea",
                user_response="interesting",
                intent=IntentCategory.QUESTION,
                summary="exploring",
            ),
        ]
        md = _build_discussion_result_md(Path("doc.md"), results, 0, 0)
        assert "## Explorations & Insights" in md


class TestVersionCommand:
    """Tests for the neev version CLI command."""

    def test_version_shows_output(self):
        """Test version command exits successfully and shows version string."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "neev-voice" in result.output

    def test_version_matches_package(self):
        """Test version command output matches the package __version__."""
        from neev_voice import __version__

        result = runner.invoke(app, ["version"])
        assert __version__ in result.output

    def test_version_in_help(self):
        """Test version command appears in main help output."""
        result = runner.invoke(app, ["--help"])
        assert "version" in result.output

    def test_version_help(self):
        """Test version --help shows description."""
        result = runner.invoke(app, ["version", "--help"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()


class TestVersionModule:
    """Tests for __version__ in neev_voice package."""

    def test_version_is_string(self):
        """Test __version__ is a string."""
        from neev_voice import __version__

        assert isinstance(__version__, str)

    def test_version_not_empty(self):
        """Test __version__ is not empty."""
        from neev_voice import __version__

        assert len(__version__) > 0

    def test_version_format(self):
        """Test __version__ follows semantic versioning pattern."""
        import re

        from neev_voice import __version__

        # Accept either semver (X.Y.Z) or dev fallback (0.0.0-dev)
        assert re.match(r"^\d+\.\d+\.\d+", __version__)


class TestListenReviewGate:
    """Tests for the transcript review gate in the listen command."""

    @pytest.fixture
    def _mock_listen_pipeline(self, mocker, tmp_path):
        """Mock all listen pipeline dependencies up to the review gate.

        Sets up mocked settings, STT, recorder, scratch pad, extractor,
        and enrichment agent so tests can focus on the review gate behavior.
        """
        from neev_voice.audio.recorder import AudioSegment

        settings = MagicMock()
        settings.stt_provider.value = "sarvam"
        settings.stt_mode.value = "translate"
        settings.sarvam_api_key = "test"
        settings.anthropic_api_key = "test-key"
        settings.sample_rate = 16000
        settings.silence_threshold = 0.03
        settings.silence_duration = 1.5
        settings.key_release_timeout = 0.15
        settings.claude_timeout = 10
        mocker.patch("neev_voice.cli._get_settings", return_value=settings)

        from neev_voice.stt.base import TranscriptionResult

        mock_stt = AsyncMock()
        mock_stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hello world", language="en", confidence=0.9, provider="mock"
            )
        )
        mocker.patch("neev_voice.stt.sarvam.get_stt_provider", return_value=mock_stt)

        mock_segment = AudioSegment(
            data=np.zeros((16000, 1), dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
        )
        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(return_value=mock_segment)
        mocker.patch("neev_voice.audio.recorder.AudioRecorder", return_value=mock_recorder)
        mocker.patch(
            "neev_voice.audio.recorder.AudioRecorder.save_wav",
            return_value=tmp_path / "test.wav",
        )
        (tmp_path / "test.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        mock_scratch = MagicMock()
        mock_scratch.flow_dir = tmp_path / "scratch"
        mock_scratch.audio_path = tmp_path / "scratch" / "audio.wav"
        mock_scratch.transcription_path = tmp_path / "scratch" / "transcription.txt"
        (tmp_path / "scratch").mkdir(parents=True, exist_ok=True)
        mocker.patch("neev_voice.scratch.ScratchPad", return_value=mock_scratch)
        self._mock_scratch = mock_scratch

        mock_extractor = MagicMock()
        mock_extractor.extract = AsyncMock(
            return_value=ExtractedIntent(
                category=IntentCategory.QUESTION,
                summary="A question",
                key_points=["test"],
                raw_text="hello world",
            )
        )
        mocker.patch("neev_voice.intent.extractor.IntentExtractor", return_value=mock_extractor)
        self._mock_extractor = mock_extractor
        mocker.patch("neev_voice.llm.agent.EnrichmentAgent")

    @pytest.mark.usefixtures("_mock_listen_pipeline")
    async def test_no_review_flag_skips_review(self, mocker):
        """Test --no-review skips the transcript review gate entirely."""
        from neev_voice.cli import _listen_async

        mock_reviewer_cls = mocker.patch("neev_voice.review.TranscriptReviewer")

        await _listen_async(None, None, mode=None, verbose=False, no_review=True)

        mock_reviewer_cls.assert_not_called()
        self._mock_extractor.extract.assert_called_once_with("hello world")

    @pytest.mark.usefixtures("_mock_listen_pipeline")
    async def test_review_accept_proceeds_with_original(self, mocker):
        """Test accept action proceeds with original transcript text."""
        from neev_voice.cli import _listen_async
        from neev_voice.review import TranscriptReviewAction

        mock_reviewer = MagicMock()
        mock_reviewer.review = AsyncMock(
            return_value=(TranscriptReviewAction.ACCEPT, "hello world")
        )
        mocker.patch("neev_voice.review.TranscriptReviewer", return_value=mock_reviewer)

        await _listen_async(None, None, mode=None, verbose=False, no_review=False)

        mock_reviewer.review.assert_called_once()
        self._mock_extractor.extract.assert_called_once_with("hello world")

    @pytest.mark.usefixtures("_mock_listen_pipeline")
    async def test_review_edit_uses_edited_text(self, mocker):
        """Test edit action uses edited text for intent extraction."""
        from neev_voice.cli import _listen_async
        from neev_voice.review import TranscriptReviewAction

        mock_reviewer = MagicMock()
        mock_reviewer.review = AsyncMock(
            return_value=(TranscriptReviewAction.EDIT, "edited transcript")
        )
        mocker.patch("neev_voice.review.TranscriptReviewer", return_value=mock_reviewer)

        await _listen_async(None, None, mode=None, verbose=False, no_review=False)

        self._mock_extractor.extract.assert_called_once_with("edited transcript")
        self._mock_scratch.save_transcription.assert_any_call("edited transcript")

    @pytest.mark.usefixtures("_mock_listen_pipeline")
    async def test_review_edit_saves_to_scratch(self, mocker):
        """Test edit action saves edited text to scratch pad."""
        from neev_voice.cli import _listen_async
        from neev_voice.review import TranscriptReviewAction

        mock_reviewer = MagicMock()
        mock_reviewer.review = AsyncMock(return_value=(TranscriptReviewAction.EDIT, "updated text"))
        mocker.patch("neev_voice.review.TranscriptReviewer", return_value=mock_reviewer)

        await _listen_async(None, None, mode=None, verbose=False, no_review=False)

        # save_transcription called twice: once for initial, once for edited
        calls = self._mock_scratch.save_transcription.call_args_list
        assert any(c.args == ("updated text",) for c in calls)

    @pytest.mark.usefixtures("_mock_listen_pipeline")
    async def test_review_reject_exits_gracefully(self, mocker):
        """Test reject action exits with code 0."""
        from click.exceptions import Exit

        from neev_voice.cli import _listen_async
        from neev_voice.exceptions import TranscriptRejectedError

        mock_reviewer = MagicMock()
        mock_reviewer.review = AsyncMock(
            side_effect=TranscriptRejectedError("Transcript rejected by user.")
        )
        mocker.patch("neev_voice.review.TranscriptReviewer", return_value=mock_reviewer)

        with pytest.raises(Exit) as exc_info:
            await _listen_async(None, None, mode=None, verbose=False, no_review=False)

        assert exc_info.value.exit_code == 0
        self._mock_extractor.extract.assert_not_called()

    @pytest.mark.usefixtures("_mock_listen_pipeline")
    async def test_review_metadata_uses_final_text(self, mocker):
        """Test metadata is saved with the final (possibly edited) transcript."""
        from neev_voice.cli import _listen_async
        from neev_voice.review import TranscriptReviewAction

        mock_reviewer = MagicMock()
        mock_reviewer.review = AsyncMock(return_value=(TranscriptReviewAction.EDIT, "final text"))
        mocker.patch("neev_voice.review.TranscriptReviewer", return_value=mock_reviewer)

        await _listen_async(None, None, mode=None, verbose=False, no_review=False)

        metadata_call = self._mock_scratch.save_metadata.call_args
        assert metadata_call.kwargs["transcription"] == "final text"

    def test_listen_help_shows_no_review_option(self):
        """Test listen --help mentions the --no-review option."""
        result = runner.invoke(app, ["listen", "--help"])
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--no-review" in plain


class TestConfigModule:
    """Tests for the config module directly."""

    def test_default_settings(self):
        """Test NeevSettings with default values."""
        from neev_voice.config import NeevSettings

        settings = NeevSettings(sarvam_api_key="test")
        assert settings.sample_rate == 16000
        assert settings.silence_threshold == 0.03
        assert settings.silence_duration == 1.5
        assert settings.claude_timeout == 30

    def test_default_stt_mode_is_translate(self):
        """Test default STT mode is translate."""
        from neev_voice.config import NeevSettings, SarvamSTTMode

        settings = NeevSettings(sarvam_api_key="test")
        assert settings.stt_mode == SarvamSTTMode.TRANSLATE

    def test_stt_mode_enum_values(self):
        """Test SarvamSTTMode enum has all expected values."""
        from neev_voice.config import SarvamSTTMode

        assert SarvamSTTMode.TRANSLATE.value == "translate"
        assert SarvamSTTMode.CODEMIX.value == "codemix"
        assert SarvamSTTMode.FORMAL.value == "formal"

    def test_custom_stt_mode(self):
        """Test setting STT mode to codemix."""
        from neev_voice.config import NeevSettings, SarvamSTTMode

        settings = NeevSettings(sarvam_api_key="test", stt_mode=SarvamSTTMode.CODEMIX)
        assert settings.stt_mode == SarvamSTTMode.CODEMIX

    def test_stt_provider_enum(self):
        """Test STT provider enum values."""
        from neev_voice.config import STTProviderType

        assert STTProviderType.SARVAM.value == "sarvam"

    def test_tts_provider_enum(self):
        """Test TTS provider enum values."""
        from neev_voice.config import TTSProviderType

        assert TTSProviderType.SARVAM.value == "sarvam"
        assert TTSProviderType.EDGE.value == "edge"

    def test_default_stt_max_audio_duration(self):
        """Test default stt_max_audio_duration is 30.0."""
        from neev_voice.config import NeevSettings

        settings = NeevSettings(sarvam_api_key="test")
        assert settings.stt_max_audio_duration == 30.0

    def test_anthropic_api_key_field_exists(self):
        """Test anthropic_api_key field exists on NeevSettings."""
        from neev_voice.config import NeevSettings

        settings = NeevSettings(sarvam_api_key="test")
        assert hasattr(settings, "anthropic_api_key")
        assert isinstance(settings.anthropic_api_key, str)


class TestGetEnrichmentAgent:
    """Tests for _get_enrichment_agent factory function."""

    def test_v1_returns_enrichment_agent(self):
        """Test v1 setting returns original EnrichmentAgent."""
        from neev_voice.cli import _get_enrichment_agent
        from neev_voice.config import EnrichmentVersion, NeevSettings
        from neev_voice.llm.agent import EnrichmentAgent

        settings = NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="test-key",
            enrichment_version=EnrichmentVersion.V1,
        )
        agent = _get_enrichment_agent(settings, "/tmp/scratch")
        assert isinstance(agent, EnrichmentAgent)

    def test_v2_returns_enrichment_loop_agent(self):
        """Test v2 setting returns EnrichmentLoopAgent."""
        from neev_voice.cli import _get_enrichment_agent
        from neev_voice.config import EnrichmentVersion, NeevSettings
        from neev_voice.llm.enrichment_loop import EnrichmentLoopAgent

        settings = NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="test-key",
            enrichment_version=EnrichmentVersion.V2,
        )
        agent = _get_enrichment_agent(settings, "/tmp/scratch")
        assert isinstance(agent, EnrichmentLoopAgent)

    def test_v2_uses_max_iterations_from_settings(self):
        """Test v2 agent receives max_iterations from settings."""
        from neev_voice.cli import _get_enrichment_agent
        from neev_voice.config import EnrichmentVersion, NeevSettings
        from neev_voice.llm.enrichment_loop import EnrichmentLoopAgent

        settings = NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="test-key",
            enrichment_version=EnrichmentVersion.V2,
            enrichment_max_iterations=5,
        )
        agent = _get_enrichment_agent(settings, "/tmp/scratch")
        assert isinstance(agent, EnrichmentLoopAgent)
        assert agent.max_iterations == 5

    def test_default_version_is_v2(self):
        """Test default enrichment_version returns v2 agent."""
        from neev_voice.cli import _get_enrichment_agent
        from neev_voice.config import NeevSettings
        from neev_voice.llm.enrichment_loop import EnrichmentLoopAgent

        settings = NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="test-key",
        )
        agent = _get_enrichment_agent(settings, "/tmp/scratch")
        assert isinstance(agent, EnrichmentLoopAgent)


class TestConfigEnrichmentDisplay:
    """Tests for enrichment config display in config command."""

    def test_config_shows_enrichment_version(self):
        """Test config shows Enrichment Version row."""
        result = runner.invoke(app, ["config"])
        assert "Enrichment Version" in result.output

    def test_config_shows_enrichment_max_iterations(self):
        """Test config shows Enrichment Max Iterations row."""
        result = runner.invoke(app, ["config"])
        assert "Enrichment Max Iterations" in result.output
