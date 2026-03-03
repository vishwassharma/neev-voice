"""Tests for discussion manager orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from neev_voice.audio.recorder import AudioRecorder, AudioSegment
from neev_voice.config import NeevSettings
from neev_voice.discussion.manager import DiscussionManager, DiscussionResult
from neev_voice.intent.extractor import ExtractedIntent, IntentCategory, IntentExtractor
from neev_voice.stt.base import STTProvider, TranscriptionResult
from neev_voice.tts.base import TTSProvider


@pytest.fixture
def settings():
    """Create test settings."""
    return NeevSettings(sarvam_api_key="test")


@pytest.fixture
def mock_recorder(settings, mocker):
    """Create a mocked AudioRecorder."""
    recorder = AudioRecorder(settings=settings)
    segment = AudioSegment(
        data=np.zeros((16000, 1), dtype=np.float32),
        sample_rate=16000,
        duration=1.0,
    )
    recorder.record_push_to_talk = mocker.AsyncMock(return_value=segment)
    mocker.patch.object(AudioRecorder, "save_wav", return_value=Path("/tmp/test.wav"))
    return recorder


@pytest.fixture
def mock_stt(mocker):
    """Create a mocked STT provider."""
    stt = mocker.AsyncMock(spec=STTProvider)
    stt.transcribe = mocker.AsyncMock(
        return_value=TranscriptionResult(
            text="haan theek hai",
            language="hi-IN",
            confidence=0.9,
            provider="mock",
        )
    )
    return stt


@pytest.fixture
def mock_tts(mocker):
    """Create a mocked TTS provider."""
    tts = mocker.AsyncMock(spec=TTSProvider)
    tts.synthesize = mocker.AsyncMock(return_value=Path("/tmp/test_tts.wav"))
    mocker.patch.object(TTSProvider, "play_audio")
    return tts


@pytest.fixture
def mock_intent_extractor(settings, mocker):
    """Create a mocked IntentExtractor."""
    mock_agent = MagicMock()
    extractor = IntentExtractor(mock_agent)
    extractor.extract_discussion_intent = mocker.AsyncMock(
        return_value=ExtractedIntent(
            category=IntentCategory.AGREEMENT,
            summary="User agrees",
            key_points=["agrees"],
            raw_text="haan theek hai",
        )
    )
    return extractor


@pytest.fixture
def manager(settings, mock_recorder, mock_stt, mock_tts, mock_intent_extractor):
    """Create a DiscussionManager with all mocked components."""
    return DiscussionManager(
        settings=settings,
        recorder=mock_recorder,
        stt=mock_stt,
        tts=mock_tts,
        intent_extractor=mock_intent_extractor,
    )


class TestChunkDocument:
    """Tests for document chunking."""

    def test_splits_on_headers(self):
        """Test splitting document on markdown headers."""
        text = "## Intro\nHello\n\n## Design\nArchitecture"
        sections = DiscussionManager._chunk_document(text)
        assert len(sections) == 2
        assert "Intro" in sections[0]
        assert "Design" in sections[1]

    def test_splits_on_double_newlines(self):
        """Test splitting on double newlines when no headers."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        sections = DiscussionManager._chunk_document(text)
        assert len(sections) == 3

    def test_empty_sections_filtered(self):
        """Test empty sections are filtered out."""
        text = "## One\nContent\n\n\n\n## Two\nMore"
        sections = DiscussionManager._chunk_document(text)
        for s in sections:
            assert s.strip()

    def test_single_section(self):
        """Test document with single section."""
        text = "Just one block of text."
        sections = DiscussionManager._chunk_document(text)
        assert len(sections) == 1


class TestLoadDocument:
    """Tests for document loading."""

    def test_loads_file(self, manager, tmp_path):
        """Test loading a document file."""
        doc = tmp_path / "test.md"
        doc.write_text("## Section 1\nContent\n\n## Section 2\nMore")
        sections = manager.load_document(doc)
        assert len(sections) == 2

    def test_missing_file_raises(self, manager, tmp_path):
        """Test loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Document not found"):
            manager.load_document(tmp_path / "missing.md")

    def test_stores_document_path(self, manager, tmp_path):
        """Test document path is stored internally."""
        doc = tmp_path / "test.md"
        doc.write_text("content")
        manager.load_document(doc)
        assert manager._document_path == str(doc)


class TestDiscussSection:
    """Tests for discussing a single section."""

    async def test_discuss_returns_result(self, manager):
        """Test discuss_section returns a DiscussionResult."""
        result = await manager.discuss_section("## Test Section\nContent here")
        assert isinstance(result, DiscussionResult)
        assert result.intent == IntentCategory.AGREEMENT

    async def test_discuss_calls_tts(self, manager, mock_tts):
        """Test discuss_section calls TTS to read the section."""
        await manager.discuss_section("Read this section")
        mock_tts.synthesize.assert_called_once_with("Read this section")

    async def test_discuss_records_audio(self, manager, mock_recorder):
        """Test discuss_section records user response via push-to-talk."""
        await manager.discuss_section("Section")
        mock_recorder.record_push_to_talk.assert_called_once()

    async def test_discuss_transcribes(self, manager, mock_stt):
        """Test discuss_section transcribes recorded audio."""
        await manager.discuss_section("Section")
        mock_stt.transcribe.assert_called_once()

    async def test_discuss_extracts_intent(self, manager, mock_intent_extractor):
        """Test discuss_section extracts intent from transcription."""
        await manager.discuss_section("Section content")
        mock_intent_extractor.extract_discussion_intent.assert_called_once()

    async def test_discuss_disagreement_result(self, manager, mock_intent_extractor):
        """Test discuss_section returns correct result for disagreement."""
        mock_intent_extractor.extract_discussion_intent.return_value = ExtractedIntent(
            category=IntentCategory.DISAGREEMENT,
            summary="User disagrees",
            key_points=["nahi"],
            raw_text="nahi yaar",
        )

        result = await manager.discuss_section("Some section")
        assert result.intent == IntentCategory.DISAGREEMENT
        assert result.summary == "User disagrees"

    async def test_discuss_agreement_result(self, manager):
        """Test discuss_section returns correct result for agreement."""
        result = await manager.discuss_section("Some section")
        assert result.intent == IntentCategory.AGREEMENT
        assert result.summary == "User agrees"


class TestRunDiscussion:
    """Tests for running a full discussion."""

    async def test_full_discussion(self, manager, tmp_path):
        """Test running a complete document discussion."""
        doc = tmp_path / "plan.md"
        doc.write_text("## Part 1\nFirst section\n\n## Part 2\nSecond section")

        results = await manager.run_discussion(doc)
        assert len(results) == 2
        assert all(isinstance(r, DiscussionResult) for r in results)

    async def test_discussion_missing_doc(self, manager, tmp_path):
        """Test run_discussion raises for missing document."""
        with pytest.raises(FileNotFoundError):
            await manager.run_discussion(tmp_path / "missing.md")
