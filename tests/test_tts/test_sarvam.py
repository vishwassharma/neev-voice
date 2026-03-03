"""Tests for Sarvam AI TTS provider."""

import base64

import pytest

from neev_voice.config import NeevSettings
from neev_voice.exceptions import NeevConfigError, NeevTTSError
from neev_voice.tts.base import TTSProvider
from neev_voice.tts.sarvam import SarvamTTS, _is_transient_error


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
        """Test SarvamTTS raises NeevConfigError without API key."""
        with pytest.raises(NeevConfigError, match="Sarvam API key is required"):
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
        mock_client.is_closed = False
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        result = await provider.synthesize("namaste")

        assert result.exists()
        assert result.suffix == ".wav"
        assert result.read_bytes() == fake_audio
        # Clean up
        result.unlink()

    async def test_synthesize_api_error_non_transient(self, settings, mocker):
        """Test synthesize raises NeevTTSError on non-transient API error (400)."""
        mock_response = mocker.MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        with pytest.raises(NeevTTSError, match="Sarvam TTS API error"):
            await provider.synthesize("namaste")
        # Non-transient: should NOT retry (single call)
        assert mock_client.post.call_count == 1

    async def test_synthesize_no_audio_data(self, settings, mocker):
        """Test synthesize raises NeevTTSError when no audio returned."""
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"audios": []}

        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        with pytest.raises(NeevTTSError, match="no audio data"):
            await provider.synthesize("namaste")


class TestSarvamTTSRetry:
    """Tests for retry/backoff on transient HTTP errors."""

    def test_transient_error_detection_timeout(self):
        """Test that httpx.TimeoutException is considered transient."""
        import httpx

        assert _is_transient_error(httpx.TimeoutException("timeout")) is True

    def test_transient_error_detection_500(self):
        """Test that NeevTTSError with HTTP 500 is considered transient."""
        assert _is_transient_error(NeevTTSError("Sarvam TTS API error (HTTP 500): err")) is True

    def test_non_transient_error_400(self):
        """Test that NeevTTSError with HTTP 400 is NOT considered transient."""
        assert _is_transient_error(NeevTTSError("Sarvam TTS API error (HTTP 400): bad")) is False

    async def test_retries_on_transient_failure(self, settings, mocker):
        """Test that synthesize retries on transient 500 then succeeds."""
        fake_audio = base64.b64encode(b"audio").decode()
        call_count = 0

        async def mock_post(*args, **kwargs):
            """First call returns 500, second returns 200."""
            nonlocal call_count
            call_count += 1
            resp = mocker.MagicMock()
            if call_count == 1:
                resp.status_code = 500
                resp.text = "Internal Server Error"
            else:
                resp.status_code = 200
                resp.json.return_value = {"audios": [fake_audio]}
            return resp

        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mocker.AsyncMock(side_effect=mock_post)
        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        result = await provider.synthesize("test")

        assert call_count == 2
        assert result.exists()
        result.unlink()


class TestSarvamTTSClientLifecycle:
    """Tests for httpx.AsyncClient reuse and cleanup."""

    def test_client_initially_none(self, settings):
        """Test that _client is None before first use."""
        provider = SarvamTTS(settings)
        assert provider._client is None

    def test_get_client_creates_client(self, settings, mocker):
        """Test that _get_client creates an AsyncClient on first call."""
        mock_client = mocker.MagicMock()
        mock_client.is_closed = False
        mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        client = provider._get_client()

        assert client is mock_client
        assert provider._client is mock_client

    def test_get_client_reuses_client(self, settings, mocker):
        """Test that _get_client returns the same client on subsequent calls."""
        mock_client = mocker.MagicMock()
        mock_client.is_closed = False
        mock_cls = mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        client1 = provider._get_client()
        client2 = provider._get_client()

        assert client1 is client2
        assert mock_cls.call_count == 1

    def test_get_client_recreates_if_closed(self, settings, mocker):
        """Test that _get_client creates a new client if the old one is closed."""
        mock_client = mocker.MagicMock()
        mock_client.is_closed = False
        mock_cls = mocker.patch("neev_voice.tts.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamTTS(settings)
        provider._get_client()
        assert mock_cls.call_count == 1

        mock_client.is_closed = True
        provider._get_client()
        assert mock_cls.call_count == 2

    async def test_close_closes_client(self, settings, mocker):
        """Test that close() calls aclose() on the client."""
        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False

        provider = SarvamTTS(settings)
        provider._client = mock_client

        await provider.close()

        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    async def test_close_noop_when_no_client(self, settings):
        """Test that close() is safe when no client exists."""
        provider = SarvamTTS(settings)
        await provider.close()  # Should not raise
