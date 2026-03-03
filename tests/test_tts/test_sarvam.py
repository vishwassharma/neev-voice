"""Tests for Sarvam AI TTS provider."""

import base64

import pytest

from neev_voice.config import NeevSettings
from neev_voice.tts.base import TTSProvider
from neev_voice.tts.sarvam import SarvamTTS


@pytest.fixture
def settings():
    """Create test settings with Sarvam API key."""
    return NeevSettings(sarvam_api_key="test-api-key-123")


@pytest.fixture
def settings_no_key():
    """Create test settings without API key."""
    return NeevSettings(sarvam_api_key="")


class TestSarvamTTSInit:
    """Tests for SarvamTTS initialization."""

    def test_init_with_key(self, settings):
        """Test SarvamTTS initializes with valid API key."""
        provider = SarvamTTS(settings)
        assert provider.settings.sarvam_api_key == "test-api-key-123"

    def test_init_without_key_raises(self, settings_no_key):
        """Test SarvamTTS raises ValueError without API key."""
        with pytest.raises(ValueError, match="Sarvam API key is required"):
            SarvamTTS(settings_no_key)

    def test_is_tts_provider(self, settings):
        """Test SarvamTTS is a TTSProvider instance."""
        provider = SarvamTTS(settings)
        assert isinstance(provider, TTSProvider)


class TestSarvamTTSSynthesize:
    """Tests for SarvamTTS synthesize method."""

    async def test_synthesize_success(self, settings, mocker):
        """Test successful synthesis via Sarvam API."""
        fake_audio = b"fake wav audio content"
        encoded_audio = base64.b64encode(fake_audio).decode()

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "audios": [encoded_audio],
        }

        mock_client = mocker.AsyncMock()
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        result = await provider.synthesize("namaste")

        assert result.exists()
        assert result.suffix == ".wav"
        assert result.read_bytes() == fake_audio
        # Clean up
        result.unlink()

    async def test_synthesize_api_error(self, settings, mocker):
        """Test synthesize raises RuntimeError on API error."""
        mock_response = mocker.MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = mocker.AsyncMock()
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        with pytest.raises(RuntimeError, match="Sarvam TTS API error"):
            await provider.synthesize("namaste")

    async def test_synthesize_no_audio_data(self, settings, mocker):
        """Test synthesize raises RuntimeError when no audio returned."""
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"audios": []}

        mock_client = mocker.AsyncMock()
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        with pytest.raises(RuntimeError, match="no audio data"):
            await provider.synthesize("namaste")
