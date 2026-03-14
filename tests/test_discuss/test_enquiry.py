"""Tests for the enquiry engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neev_voice.discuss.enquiry import EnquiryEngine, EnquiryResult
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
    s.sample_rate = 16000
    s.silence_threshold = 0.03
    s.silence_duration = 1.5
    s.key_release_timeout = 0.15
    s.discuss_base_dir = ".scratch/neev/discuss"
    return s


class TestEnquiryResult:
    """Tests for EnquiryResult dataclass."""

    def test_escaped_result(self) -> None:
        """Escaped result has no query."""
        result = EnquiryResult(escaped=True)
        assert result.escaped
        assert result.query is None
        assert result.source == "escaped"

    def test_voice_result(self) -> None:
        """Voice result has query and audio_path."""
        result = EnquiryResult(
            escaped=False,
            query="What is this?",
            audio_path=Path("/tmp/audio.wav"),
            source="voice",
        )
        assert not result.escaped
        assert result.query == "What is this?"
        assert result.audio_path == Path("/tmp/audio.wav")
        assert result.source == "voice"

    def test_manual_result(self) -> None:
        """Manual result has query but no audio_path."""
        result = EnquiryResult(
            escaped=False,
            query="Explain X",
            source="manual",
        )
        assert not result.escaped
        assert result.query == "Explain X"
        assert result.audio_path is None
        assert result.source == "manual"


class TestEnquiryEngine:
    """Tests for EnquiryEngine."""

    def test_init_defaults(self, session: SessionInfo, settings: MagicMock) -> None:
        """Engine initializes with default session dir."""
        engine = EnquiryEngine(session, settings)
        assert engine.session is session
        assert engine.settings is settings
        assert engine.stt_provider is None

    def test_init_custom_session_dir(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """Engine accepts custom session directory."""
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)
        assert engine.session_dir == tmp_path

    def test_make_enquiry_dir(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_make_enquiry_dir creates timestamped directory."""
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)
        enquiry_dir = engine._make_enquiry_dir()
        assert enquiry_dir.exists()
        assert enquiry_dir.parent.name == "enquiries"

    def test_cleanup_enquiry_dir(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_cleanup_enquiry_dir removes directory."""
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)
        enquiry_dir = engine._make_enquiry_dir()
        (enquiry_dir / "test.txt").write_text("test")
        assert enquiry_dir.exists()

        engine._cleanup_enquiry_dir(enquiry_dir)
        assert not enquiry_dir.exists()

    def test_cleanup_nonexistent_dir(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_cleanup_enquiry_dir handles nonexistent directory."""
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)
        # Should not raise
        engine._cleanup_enquiry_dir(tmp_path / "nonexistent")

    @patch("neev_voice.discuss.enquiry.click.edit")
    def test_handle_manual_with_text(
        self,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_manual returns query when editor saved."""
        mock_edit.return_value = "  What is dependency injection?  "
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)

        result = engine._handle_manual()
        assert not result.escaped
        assert result.query == "What is dependency injection?"
        assert result.source == "manual"

    @patch("neev_voice.discuss.enquiry.click.edit")
    def test_handle_manual_editor_cancelled(
        self,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_manual returns no query when editor cancelled."""
        mock_edit.return_value = None
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)

        result = engine._handle_manual()
        assert not result.escaped
        assert result.query is None
        assert result.source == "manual"

    @patch("neev_voice.discuss.enquiry.click.edit")
    def test_handle_manual_empty_text(
        self,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_manual returns no query for empty text."""
        mock_edit.return_value = "   \n  "
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)

        result = engine._handle_manual()
        assert result.query is None

    @patch("neev_voice.discuss.enquiry.click.edit")
    def test_handle_manual_saves_query(
        self,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_manual saves query.txt in enquiry directory."""
        mock_edit.return_value = "My question"
        engine = EnquiryEngine(session, settings, session_dir=tmp_path)

        result = engine._handle_manual()
        assert result.query == "My question"

        # Find the saved query file
        enquiry_dirs = list((tmp_path / "enquiries").iterdir())
        assert len(enquiry_dirs) == 1
        query_file = enquiry_dirs[0] / "query.txt"
        assert query_file.read_text() == "My question"


class TestEnquiryEngineVoice:
    """Tests for voice recording path."""

    @patch("neev_voice.discuss.enquiry.click.edit")
    @patch("neev_voice.discuss.enquiry.AudioRecorder")
    async def test_handle_voice_recording_cancelled(
        self,
        mock_recorder_cls: MagicMock,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_voice returns on recording cancellation."""
        from neev_voice.audio.recorder import RecordingCancelledError

        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(side_effect=RecordingCancelledError())
        mock_recorder_cls.return_value = mock_recorder

        stt = AsyncMock()
        engine = EnquiryEngine(session, settings, stt_provider=stt, session_dir=tmp_path)
        result = await engine._handle_voice()

        assert not result.escaped
        assert result.query is None
        assert result.source == "voice"

    @patch("neev_voice.discuss.enquiry.click.edit")
    @patch("neev_voice.discuss.enquiry.AudioRecorder")
    async def test_handle_voice_no_stt_falls_to_manual(
        self,
        mock_recorder_cls: MagicMock,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_voice falls back to manual when no STT provider."""
        mock_edit.return_value = "Manual fallback"
        engine = EnquiryEngine(session, settings, stt_provider=None, session_dir=tmp_path)
        result = await engine._handle_voice()

        assert result.query == "Manual fallback"
        assert result.source == "manual"

    @patch("neev_voice.discuss.enquiry.click.edit")
    @patch("neev_voice.discuss.enquiry.AudioRecorder")
    async def test_handle_voice_editor_cancelled(
        self,
        mock_recorder_cls: MagicMock,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_voice cleans up when editor is cancelled."""
        mock_segment = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(return_value=mock_segment)
        mock_recorder_cls.return_value = mock_recorder
        mock_recorder_cls.save_wav.return_value = tmp_path / "temp.wav"
        (tmp_path / "temp.wav").write_bytes(b"fake wav")

        stt = AsyncMock()
        stt.transcribe = AsyncMock(return_value=MagicMock(text="transcribed text"))

        mock_edit.return_value = None  # Editor cancelled

        engine = EnquiryEngine(session, settings, stt_provider=stt, session_dir=tmp_path)
        result = await engine._handle_voice()

        assert result.query is None
        assert result.source == "voice"

    @patch("neev_voice.discuss.enquiry.click.edit")
    @patch("neev_voice.discuss.enquiry.AudioRecorder")
    async def test_handle_voice_success(
        self,
        mock_recorder_cls: MagicMock,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_voice returns query on success."""
        mock_segment = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(return_value=mock_segment)
        mock_recorder_cls.return_value = mock_recorder

        wav_file = tmp_path / "temp.wav"
        wav_file.write_bytes(b"fake wav")
        mock_recorder_cls.save_wav.return_value = wav_file

        stt = AsyncMock()
        stt.transcribe = AsyncMock(return_value=MagicMock(text="what is this concept"))

        mock_edit.return_value = "What is this concept?"

        engine = EnquiryEngine(session, settings, stt_provider=stt, session_dir=tmp_path)
        result = await engine._handle_voice()

        assert not result.escaped
        assert result.query == "What is this concept?"
        assert result.source == "voice"
        assert result.audio_path is not None

    @patch("neev_voice.discuss.enquiry.click.edit")
    @patch("neev_voice.discuss.enquiry.AudioRecorder")
    async def test_handle_voice_stt_error(
        self,
        mock_recorder_cls: MagicMock,
        mock_edit: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_handle_voice handles STT errors gracefully."""
        mock_segment = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.record_push_to_talk = AsyncMock(return_value=mock_segment)
        mock_recorder_cls.return_value = mock_recorder

        wav_file = tmp_path / "temp.wav"
        wav_file.write_bytes(b"fake wav")
        mock_recorder_cls.save_wav.return_value = wav_file

        stt = AsyncMock()
        stt.transcribe = AsyncMock(side_effect=RuntimeError("STT failed"))

        engine = EnquiryEngine(session, settings, stt_provider=stt, session_dir=tmp_path)
        result = await engine._handle_voice()

        assert result.query is None
        assert result.source == "voice"
