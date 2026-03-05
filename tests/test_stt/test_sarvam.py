"""Tests for Sarvam AI STT provider."""

import json

import numpy as np
import pytest
from scipy.io import wavfile

from neev_voice.config import NeevSettings, SarvamSTTMode
from neev_voice.exceptions import NeevConfigError, NeevSTTError
from neev_voice.stt.base import STTProvider, TranscriptionResult
from neev_voice.stt.sarvam import SarvamSTT, _is_transient_error, get_stt_provider


@pytest.fixture
def settings():
    """Create test settings with Sarvam API key."""
    return NeevSettings(sarvam_api_key="test-api-key-123")


@pytest.fixture
def settings_no_key(monkeypatch):
    """Create test settings without API key."""
    monkeypatch.delenv("NEEV_SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    return NeevSettings(sarvam_api_key="", _env_file=None)


def _make_wav(path, duration_s=1.0, sample_rate=16000):
    """Create a valid WAV file with silence of given duration.

    Args:
        path: Path to write the WAV file.
        duration_s: Duration of silence in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the created WAV file.
    """
    samples = int(duration_s * sample_rate)
    data = np.zeros(samples, dtype=np.int16)
    wavfile.write(str(path), sample_rate, data)
    return path


def _mock_sarvam_client(mocker, response_json, status_code=200):
    """Create a mocked httpx.AsyncClient that returns the given response.

    The mock sets ``is_closed = False`` so the shared client is reused
    across calls via ``_get_client()``.

    Args:
        mocker: pytest-mock mocker fixture.
        response_json: Dict to return from response.json().
        status_code: HTTP status code.

    Returns:
        The mock client object.
    """
    mock_response = mocker.MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_response.text = (
        json.dumps(response_json) if isinstance(response_json, dict) else str(response_json)
    )

    mock_client = mocker.AsyncMock()
    mock_client.is_closed = False
    mock_client.post = mocker.AsyncMock(return_value=mock_response)

    mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)
    return mock_client


class TestSarvamSTTInit:
    """Tests for SarvamSTT initialization."""

    def test_init_with_key(self, settings):
        """Test SarvamSTT initializes with valid API key."""
        provider = SarvamSTT(settings)
        assert provider.settings.sarvam_api_key == "test-api-key-123"

    def test_init_without_key_raises(self, settings_no_key):
        """Test SarvamSTT raises NeevConfigError without API key."""
        with pytest.raises(NeevConfigError, match="Sarvam API key is required"):
            SarvamSTT(settings_no_key)

    def test_is_stt_provider(self, settings):
        """Test SarvamSTT is an STTProvider instance."""
        provider = SarvamSTT(settings)
        assert isinstance(provider, STTProvider)


class TestSarvamSTTTranscribe:
    """Tests for SarvamSTT transcribe method."""

    async def test_transcribe_success(self, settings, tmp_path, mocker):
        """Test successful transcription via Sarvam API."""
        audio_file = _make_wav(tmp_path / "test.wav", duration_s=1.0)

        _mock_sarvam_client(
            mocker,
            {
                "transcript": "namaste duniya",
                "language_code": "hi-IN",
                "confidence": 0.92,
            },
        )

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "namaste duniya"
        assert result.language == "hi-IN"
        assert result.confidence == 0.92
        assert result.provider == "sarvam"

    async def test_transcribe_file_not_found(self, settings, tmp_path):
        """Test transcribe raises FileNotFoundError for missing file."""
        provider = SarvamSTT(settings)
        missing_file = tmp_path / "nonexistent.wav"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await provider.transcribe(missing_file)

    async def test_transcribe_api_error(self, settings, tmp_path, mocker):
        """Test transcribe raises NeevSTTError on API error."""
        audio_file = _make_wav(tmp_path / "test.wav")

        mock_response = mocker.MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        with pytest.raises(NeevSTTError, match="Sarvam STT API error"):
            await provider.transcribe(audio_file)

    async def test_transcribe_empty_transcript(self, settings, tmp_path, mocker):
        """Test transcribe handles empty transcript gracefully."""
        audio_file = _make_wav(tmp_path / "test.wav")

        _mock_sarvam_client(mocker, {})

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)
        assert result.text == ""

    async def test_transcribe_uses_default_translate_mode(self, settings, tmp_path, mocker):
        """Test transcribe sends translate mode by default."""
        audio_file = _make_wav(tmp_path / "test.wav")

        mock_client = _mock_sarvam_client(mocker, {"transcript": "hello"})

        provider = SarvamSTT(settings)
        await provider.transcribe(audio_file)

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["data"]["mode"] == "translate"

    async def test_transcribe_uses_codemix_mode(self, tmp_path, mocker):
        """Test transcribe sends codemix mode when configured."""
        codemix_settings = NeevSettings(sarvam_api_key="test-key", stt_mode=SarvamSTTMode.CODEMIX)
        audio_file = _make_wav(tmp_path / "test.wav")

        mock_client = _mock_sarvam_client(mocker, {"transcript": "namaste"})

        provider = SarvamSTT(codemix_settings)
        await provider.transcribe(audio_file)

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["data"]["mode"] == "codemix"

    async def test_transcribe_uses_formal_mode(self, tmp_path, mocker):
        """Test transcribe sends formal mode when configured."""
        formal_settings = NeevSettings(sarvam_api_key="test-key", stt_mode=SarvamSTTMode.FORMAL)
        audio_file = _make_wav(tmp_path / "test.wav")

        mock_client = _mock_sarvam_client(mocker, {"transcript": "namaste"})

        provider = SarvamSTT(formal_settings)
        await provider.transcribe(audio_file)

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["data"]["mode"] == "formal"


class TestSarvamSTTTranscribeRouting:
    """Tests for transcribe() routing: REST for short audio, streaming for long."""

    async def test_short_audio_uses_rest(self, settings, tmp_path, mocker):
        """Test audio ≤30s uses REST API via _transcribe_single."""
        audio_file = _make_wav(tmp_path / "short.wav", duration_s=10.0)

        mock_client = _mock_sarvam_client(
            mocker,
            {
                "transcript": "hello short",
                "language_code": "hi-IN",
                "confidence": 0.95,
            },
        )

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert result.text == "hello short"
        assert result.provider == "sarvam"
        assert mock_client.post.call_count == 1

    async def test_long_audio_uses_streaming(self, settings, tmp_path, mocker):
        """Test audio >30s routes to _transcribe_streaming."""
        audio_file = _make_wav(tmp_path / "long.wav", duration_s=60.0)

        mock_streaming = mocker.AsyncMock(
            return_value=TranscriptionResult(
                text="streamed result",
                language="hi-IN",
                confidence=0.9,
                provider="sarvam-streaming",
            )
        )
        mocker.patch.object(SarvamSTT, "_transcribe_streaming", mock_streaming)

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert result.text == "streamed result"
        assert result.provider == "sarvam-streaming"
        mock_streaming.assert_awaited_once_with(audio_file)

    async def test_exactly_30s_uses_rest(self, settings, tmp_path, mocker):
        """Test audio exactly at max duration uses REST API."""
        audio_file = _make_wav(tmp_path / "exact.wav", duration_s=30.0)

        _mock_sarvam_client(
            mocker,
            {"transcript": "exact 30s", "language_code": "hi-IN", "confidence": 0.9},
        )

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert result.text == "exact 30s"
        assert result.provider == "sarvam"

    async def test_just_over_30s_uses_streaming(self, settings, tmp_path, mocker):
        """Test audio just over max duration routes to streaming."""
        audio_file = _make_wav(tmp_path / "over.wav", duration_s=30.1)

        mock_streaming = mocker.AsyncMock(
            return_value=TranscriptionResult(
                text="streamed over 30s",
                language="hi-IN",
                confidence=0.88,
                provider="sarvam-streaming",
            )
        )
        mocker.patch.object(SarvamSTT, "_transcribe_streaming", mock_streaming)

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert result.provider == "sarvam-streaming"
        mock_streaming.assert_awaited_once()


class TestMergeResults:
    """Tests for the _merge_results static method."""

    def test_merge_empty_list(self):
        """Test merging empty list returns defaults."""
        result = SarvamSTT._merge_results([])
        assert result.text == ""
        assert result.confidence == 0.0

    def test_merge_single_result(self):
        """Test merging single result returns it unchanged."""
        r = TranscriptionResult(text="hello", language="hi-IN", confidence=0.9, provider="sarvam")
        result = SarvamSTT._merge_results([r])
        assert result.text == "hello"
        assert result.confidence == 0.9

    def test_merge_multiple_results(self):
        """Test merging multiple results concatenates and averages."""
        results = [
            TranscriptionResult(text="hello", language="hi-IN", confidence=0.9, provider="sarvam"),
            TranscriptionResult(text="world", language="hi-IN", confidence=0.7, provider="sarvam"),
        ]
        result = SarvamSTT._merge_results(results)
        assert result.text == "hello world"
        assert abs(result.confidence - 0.8) < 1e-6
        assert result.language == "hi-IN"
        assert result.provider == "sarvam"


class TestRetryBehavior:
    """Tests for retry/backoff on transient HTTP errors."""

    def test_transient_error_detection_timeout(self):
        """Test that httpx.TimeoutException is considered transient."""
        import httpx

        assert _is_transient_error(httpx.TimeoutException("timeout")) is True

    def test_transient_error_detection_connect(self):
        """Test that httpx.ConnectError is considered transient."""
        import httpx

        assert _is_transient_error(httpx.ConnectError("connection refused")) is True

    def test_transient_error_detection_server_error(self):
        """Test that NeevSTTError with HTTP 500 is considered transient."""
        assert _is_transient_error(NeevSTTError("Sarvam STT API error (HTTP 500): error")) is True

    def test_transient_error_detection_429(self):
        """Test that NeevSTTError with HTTP 429 is considered transient."""
        assert _is_transient_error(NeevSTTError("Sarvam STT API error (HTTP 429): rate")) is True

    def test_non_transient_error_401(self):
        """Test that NeevSTTError with HTTP 401 is NOT considered transient."""
        assert _is_transient_error(NeevSTTError("Sarvam STT API error (HTTP 401): unauth")) is False

    def test_non_transient_error_generic(self):
        """Test that a generic ValueError is NOT considered transient."""
        assert _is_transient_error(ValueError("bad value")) is False

    async def test_retries_on_transient_failure(self, settings, tmp_path, mocker):
        """Test that _transcribe_single retries on transient 500 error."""
        audio_file = _make_wav(tmp_path / "test.wav")

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
                resp.json.return_value = {
                    "transcript": "success",
                    "language_code": "hi-IN",
                    "confidence": 0.9,
                }
            return resp

        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mocker.AsyncMock(side_effect=mock_post)
        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert call_count == 2
        assert result.text == "success"

    async def test_no_retry_on_non_transient(self, settings, tmp_path, mocker):
        """Test that _transcribe_single does NOT retry on 401."""
        audio_file = _make_wav(tmp_path / "test.wav")

        mock_response = mocker.MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False
        mock_client.post = mocker.AsyncMock(return_value=mock_response)
        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        with pytest.raises(NeevSTTError, match="HTTP 401"):
            await provider.transcribe(audio_file)

        assert mock_client.post.call_count == 1


class TestSarvamSTTClientLifecycle:
    """Tests for httpx.AsyncClient reuse and cleanup."""

    def test_client_initially_none(self, settings):
        """Test that _client is None before first use."""
        provider = SarvamSTT(settings)
        assert provider._client is None

    def test_get_client_creates_client(self, settings, mocker):
        """Test that _get_client creates an AsyncClient on first call."""
        mock_client = mocker.MagicMock()
        mock_client.is_closed = False
        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        client = provider._get_client()

        assert client is mock_client
        assert provider._client is mock_client

    def test_get_client_reuses_client(self, settings, mocker):
        """Test that _get_client returns the same client on subsequent calls."""
        mock_client = mocker.MagicMock()
        mock_client.is_closed = False
        mock_cls = mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        client1 = provider._get_client()
        client2 = provider._get_client()

        assert client1 is client2
        assert mock_cls.call_count == 1

    def test_get_client_recreates_if_closed(self, settings, mocker):
        """Test that _get_client creates a new client if the old one is closed."""
        mock_client = mocker.MagicMock()
        mock_client.is_closed = False
        mock_cls = mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        provider._get_client()
        assert mock_cls.call_count == 1

        mock_client.is_closed = True
        provider._get_client()
        assert mock_cls.call_count == 2

    async def test_close_closes_client(self, settings, mocker):
        """Test that close() calls aclose() on the client."""
        mock_client = mocker.AsyncMock()
        mock_client.is_closed = False

        provider = SarvamSTT(settings)
        provider._client = mock_client

        await provider.close()

        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    async def test_close_noop_when_no_client(self, settings):
        """Test that close() is safe when no client exists."""
        provider = SarvamSTT(settings)
        await provider.close()  # Should not raise


class TestGetSTTProvider:
    """Tests for the STT provider factory function."""

    def test_get_sarvam_provider(self, settings):
        """Test factory returns SarvamSTT for 'sarvam'."""
        provider = get_stt_provider("sarvam", settings)
        assert isinstance(provider, SarvamSTT)

    def test_unknown_provider_raises(self, settings):
        """Test factory raises NeevConfigError for unknown provider."""
        with pytest.raises(NeevConfigError, match="Unknown STT provider"):
            get_stt_provider("whisper", settings)

    def test_unknown_provider_lists_available(self, settings):
        """Test error message lists available providers."""
        with pytest.raises(NeevConfigError, match="sarvam"):
            get_stt_provider("unknown", settings)


class TestSarvamStreamingTranscribe:
    """Tests for WebSocket streaming transcription of long audio."""

    @pytest.fixture
    def _mock_ws(self, mocker):
        """Create a mock WebSocket connection with configurable responses.

        Sets up a mock websockets.connect context manager. Tests can configure
        responses via self._ws_responses before calling transcribe.
        """
        self._ws_sent = []
        self._ws_responses = []

        mock_ws = mocker.AsyncMock()

        async def mock_send(msg):
            self._ws_sent.append(msg)

        mock_ws.send = mock_send

        async def mock_aiter():
            for resp in self._ws_responses:
                yield resp

        mock_ws.__aiter__ = lambda s: mock_aiter()

        mock_connect = mocker.patch("neev_voice.stt.sarvam.websockets.connect")
        mock_ctx = mocker.AsyncMock()
        mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_ws)
        mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_connect.return_value = mock_ctx
        self._mock_connect = mock_connect
        self._mock_ws = mock_ws

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_success(self, settings, tmp_path):
        """Test streaming transcription returns correct result."""
        audio_file = _make_wav(tmp_path / "long.wav", duration_s=1.0)

        self._ws_responses = [
            json.dumps(
                {
                    "type": "data",
                    "data": {
                        "transcript": "hello world namaste",
                        "language_code": "hi-IN",
                        "language_probability": 0.95,
                    },
                }
            ),
        ]

        provider = SarvamSTT(settings)
        result = await provider._transcribe_streaming(audio_file)

        assert result.text == "hello world namaste"
        assert result.language == "hi-IN"
        assert result.confidence == 0.95
        assert result.provider == "sarvam-streaming"

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_sends_base64_audio(self, settings, tmp_path):
        """Test streaming sends base64-encoded audio data."""
        import base64

        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        self._ws_responses = [
            json.dumps(
                {
                    "type": "data",
                    "data": {"transcript": "test", "language_code": "hi-IN"},
                }
            ),
        ]

        provider = SarvamSTT(settings)
        await provider._transcribe_streaming(audio_file)

        assert len(self._ws_sent) >= 2
        audio_msg = json.loads(self._ws_sent[0])
        assert "audio" in audio_msg
        assert "data" in audio_msg["audio"]
        # Verify it's valid base64
        base64.b64decode(audio_msg["audio"]["data"])

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_sends_flush_signal(self, settings, tmp_path):
        """Test streaming sends flush signal after audio."""
        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        self._ws_responses = [
            json.dumps(
                {
                    "type": "data",
                    "data": {"transcript": "test", "language_code": "hi-IN"},
                }
            ),
        ]

        provider = SarvamSTT(settings)
        await provider._transcribe_streaming(audio_file)

        flush_msg = json.loads(self._ws_sent[-1])
        assert flush_msg == {"type": "flush"}

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_multiple_segments(self, settings, tmp_path):
        """Test streaming merges multiple transcript segments."""
        audio_file = _make_wav(tmp_path / "long.wav", duration_s=1.0)

        self._ws_responses = [
            json.dumps(
                {
                    "type": "data",
                    "data": {
                        "transcript": "first part",
                        "language_code": "hi-IN",
                        "language_probability": 0.9,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "data",
                    "data": {
                        "transcript": "second part",
                        "language_code": "hi-IN",
                        "language_probability": 0.8,
                    },
                }
            ),
        ]

        provider = SarvamSTT(settings)
        result = await provider._transcribe_streaming(audio_file)

        assert result.text == "first part second part"
        assert abs(result.confidence - 0.85) < 1e-6

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_passes_auth_header(self, settings, tmp_path):
        """Test streaming passes API key in headers."""
        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        self._ws_responses = [
            json.dumps(
                {
                    "type": "data",
                    "data": {"transcript": "test", "language_code": "hi-IN"},
                }
            ),
        ]

        provider = SarvamSTT(settings)
        await provider._transcribe_streaming(audio_file)

        call_kwargs = self._mock_connect.call_args.kwargs
        assert call_kwargs["additional_headers"]["Api-Subscription-Key"] == "test-api-key-123"

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_url_contains_params(self, settings, tmp_path):
        """Test streaming URL includes model, mode, and language params."""
        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        self._ws_responses = [
            json.dumps(
                {
                    "type": "data",
                    "data": {"transcript": "test", "language_code": "hi-IN"},
                }
            ),
        ]

        provider = SarvamSTT(settings)
        await provider._transcribe_streaming(audio_file)

        url = self._mock_connect.call_args.args[0]
        assert "model=saaras%3Av3" in url or "model=saaras:v3" in url
        assert "mode=translate" in url
        assert "language-code=hi-IN" in url
        assert "flush_signal=true" in url

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_error_response(self, settings, tmp_path):
        """Test streaming raises NeevSTTError on error response."""
        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        self._ws_responses = [
            json.dumps(
                {
                    "type": "error",
                    "data": {"error": "Invalid audio format", "code": "INVALID_FORMAT"},
                }
            ),
        ]

        provider = SarvamSTT(settings)
        with pytest.raises(NeevSTTError, match="Invalid audio format"):
            await provider._transcribe_streaming(audio_file)

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_skips_event_messages(self, settings, tmp_path):
        """Test streaming ignores VAD event messages."""
        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        self._ws_responses = [
            json.dumps({"type": "events", "data": {"signal_type": "START_SPEECH"}}),
            json.dumps(
                {
                    "type": "data",
                    "data": {"transcript": "actual text", "language_code": "hi-IN"},
                }
            ),
            json.dumps({"type": "events", "data": {"signal_type": "END_SPEECH"}}),
        ]

        provider = SarvamSTT(settings)
        result = await provider._transcribe_streaming(audio_file)

        assert result.text == "actual text"

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_empty_transcript(self, settings, tmp_path):
        """Test streaming returns empty text when no transcripts received."""
        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        self._ws_responses = []

        provider = SarvamSTT(settings)
        result = await provider._transcribe_streaming(audio_file)

        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.usefixtures("_mock_ws")
    async def test_streaming_connection_error(self, settings, tmp_path, mocker):
        """Test streaming raises NeevSTTError on connection failure."""
        import websockets

        audio_file = _make_wav(tmp_path / "test.wav", duration_s=0.5)

        mocker.patch(
            "neev_voice.stt.sarvam.websockets.connect",
            side_effect=websockets.exceptions.WebSocketException("Connection refused"),
        )

        provider = SarvamSTT(settings)
        with pytest.raises(NeevSTTError, match="Streaming STT connection failed"):
            await provider._transcribe_streaming(audio_file)
