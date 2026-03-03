"""Tests for Edge TTS provider."""

import pytest

from neev_voice.config import NeevSettings
from neev_voice.tts.base import TTSProvider
from neev_voice.tts.edge import DEFAULT_VOICE, EdgeTTS, get_tts_provider


class TestEdgeTTSInit:
    """Tests for EdgeTTS initialization."""

    def test_default_voice(self):
        """Test EdgeTTS uses default Hindi voice."""
        provider = EdgeTTS()
        assert provider.voice == DEFAULT_VOICE

    def test_custom_voice(self):
        """Test EdgeTTS accepts custom voice."""
        provider = EdgeTTS(voice="en-US-JennyNeural")
        assert provider.voice == "en-US-JennyNeural"

    def test_is_tts_provider(self):
        """Test EdgeTTS is a TTSProvider instance."""
        provider = EdgeTTS()
        assert isinstance(provider, TTSProvider)


class TestEdgeTTSSynthesize:
    """Tests for EdgeTTS synthesize method."""

    async def test_synthesize_calls_edge_tts(self, mocker):
        """Test synthesize creates Communicate and calls save."""
        mock_communicate = mocker.MagicMock()
        mock_communicate.save = mocker.AsyncMock()

        mocker.patch(
            "neev_voice.tts.edge.edge_tts.Communicate",
            return_value=mock_communicate,
        )

        provider = EdgeTTS()
        result = await provider.synthesize("hello world")

        assert result.suffix == ".mp3"
        mock_communicate.save.assert_called_once()

    async def test_synthesize_uses_correct_voice(self, mocker):
        """Test synthesize passes the configured voice."""
        mock_communicate_cls = mocker.patch("neev_voice.tts.edge.edge_tts.Communicate")
        mock_communicate_cls.return_value.save = mocker.AsyncMock()

        provider = EdgeTTS(voice="test-voice")
        await provider.synthesize("hello")

        mock_communicate_cls.assert_called_once_with("hello", "test-voice")


class TestGetTTSProvider:
    """Tests for the TTS provider factory function."""

    def test_get_edge_provider(self):
        """Test factory returns EdgeTTS for 'edge'."""
        provider = get_tts_provider("edge")
        assert isinstance(provider, EdgeTTS)

    def test_get_sarvam_provider(self):
        """Test factory returns SarvamTTS for 'sarvam' with settings."""
        settings = NeevSettings(sarvam_api_key="test-key")
        from neev_voice.tts.sarvam import SarvamTTS

        provider = get_tts_provider("sarvam", settings)
        assert isinstance(provider, SarvamTTS)

    def test_unknown_provider_raises(self):
        """Test factory raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts_provider("unknown")

    def test_sarvam_without_settings_raises(self):
        """Test factory raises when Sarvam requested without settings."""
        with pytest.raises(ValueError):
            get_tts_provider("sarvam", None)
