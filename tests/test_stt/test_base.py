"""Tests for STT base classes."""

import pytest

from neev_voice.stt.base import STTProvider, TranscriptionResult


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_creation(self):
        """Test TranscriptionResult creation with all fields."""
        result = TranscriptionResult(
            text="hello world",
            language="en-US",
            confidence=0.95,
            provider="test",
        )
        assert result.text == "hello world"
        assert result.language == "en-US"
        assert result.confidence == 0.95
        assert result.provider == "test"

    def test_empty_text(self):
        """Test TranscriptionResult with empty text."""
        result = TranscriptionResult(
            text="",
            language="hi-IN",
            confidence=0.0,
            provider="test",
        )
        assert result.text == ""


class TestSTTProviderABC:
    """Tests for STTProvider abstract base class."""

    def test_cannot_instantiate(self):
        """Test that STTProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            STTProvider()

    def test_subclass_must_implement_transcribe(self):
        """Test that subclass without transcribe raises TypeError."""

        class BadProvider(STTProvider):
            pass

        with pytest.raises(TypeError):
            BadProvider()

    def test_valid_subclass(self):
        """Test that a proper subclass can be instantiated."""

        class GoodProvider(STTProvider):
            async def transcribe(self, audio_path):
                return TranscriptionResult(
                    text="test", language="en", confidence=1.0, provider="good"
                )

        provider = GoodProvider()
        assert isinstance(provider, STTProvider)
