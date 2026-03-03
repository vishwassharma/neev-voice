"""Tests for Sarvam AI STT provider."""

import json

import numpy as np
import pytest
from scipy.io import wavfile

from neev_voice.config import NeevSettings, SarvamSTTMode
from neev_voice.stt.base import STTProvider, TranscriptionResult
from neev_voice.stt.sarvam import SarvamSTT, get_stt_provider


@pytest.fixture
def settings():
    """Create test settings with Sarvam API key."""
    return NeevSettings(sarvam_api_key="test-api-key-123")


@pytest.fixture
def settings_no_key():
    """Create test settings without API key."""
    return NeevSettings(sarvam_api_key="")


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
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
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
        """Test SarvamSTT raises ValueError without API key."""
        with pytest.raises(ValueError, match="Sarvam API key is required"):
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
        """Test transcribe raises RuntimeError on API error."""
        audio_file = _make_wav(tmp_path / "test.wav")

        mock_response = mocker.MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = mocker.AsyncMock()
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)

        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        with pytest.raises(RuntimeError, match="Sarvam STT API error"):
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


class TestSarvamSTTChunkedTranscribe:
    """Tests for chunked transcription of audio > 30s."""

    async def test_short_audio_single_api_call(self, settings, tmp_path, mocker):
        """Test audio ≤30s results in a single API call."""
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
        assert mock_client.post.call_count == 1

    async def test_long_audio_multiple_api_calls(self, settings, tmp_path, mocker):
        """Test audio >30s results in multiple API calls and merged result."""
        audio_file = _make_wav(tmp_path / "long.wav", duration_s=60.0)

        call_count = 0
        responses = [
            {"transcript": "first chunk", "language_code": "hi-IN", "confidence": 0.9},
            {"transcript": "second chunk", "language_code": "hi-IN", "confidence": 0.8},
        ]

        def make_response(idx):
            """Create a mock response for the given chunk index."""
            resp = mocker.MagicMock()
            resp.status_code = 200
            resp.json.return_value = responses[idx]
            return resp

        async def mock_post(*args, **kwargs):
            """Return successive mock responses for each chunk."""
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return make_response(idx)

        mock_client = mocker.AsyncMock()
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_client.post = mocker.AsyncMock(side_effect=mock_post)

        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert call_count == 2
        assert result.text == "first chunk second chunk"
        assert abs(result.confidence - 0.85) < 1e-6
        assert result.language == "hi-IN"
        assert result.provider == "sarvam"

    async def test_merged_text_is_space_joined(self, settings, tmp_path, mocker):
        """Test merged text from chunks is space-joined."""
        audio_file = _make_wav(tmp_path / "long.wav", duration_s=45.0)

        call_count = 0
        responses = [
            {"transcript": "hello", "language_code": "hi-IN", "confidence": 0.9},
            {"transcript": "world", "language_code": "hi-IN", "confidence": 0.8},
        ]

        async def mock_post(*args, **kwargs):
            """Return successive mock responses."""
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            resp = mocker.MagicMock()
            resp.status_code = 200
            resp.json.return_value = responses[idx]
            return resp

        mock_client = mocker.AsyncMock()
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_client.post = mocker.AsyncMock(side_effect=mock_post)

        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert result.text == "hello world"

    async def test_merged_confidence_is_averaged(self, settings, tmp_path, mocker):
        """Test merged confidence is the average of chunk confidences."""
        audio_file = _make_wav(tmp_path / "long.wav", duration_s=60.0)

        call_count = 0
        responses = [
            {"transcript": "a", "language_code": "hi-IN", "confidence": 1.0},
            {"transcript": "b", "language_code": "hi-IN", "confidence": 0.5},
        ]

        async def mock_post(*args, **kwargs):
            """Return successive mock responses."""
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            resp = mocker.MagicMock()
            resp.status_code = 200
            resp.json.return_value = responses[idx]
            return resp

        mock_client = mocker.AsyncMock()
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_client.post = mocker.AsyncMock(side_effect=mock_post)

        mocker.patch("neev_voice.stt.sarvam.httpx.AsyncClient", return_value=mock_client)

        provider = SarvamSTT(settings)
        result = await provider.transcribe(audio_file)

        assert abs(result.confidence - 0.75) < 1e-6


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


class TestGetSTTProvider:
    """Tests for the STT provider factory function."""

    def test_get_sarvam_provider(self, settings):
        """Test factory returns SarvamSTT for 'sarvam'."""
        provider = get_stt_provider("sarvam", settings)
        assert isinstance(provider, SarvamSTT)

    def test_unknown_provider_raises(self, settings):
        """Test factory raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown STT provider"):
            get_stt_provider("whisper", settings)

    def test_unknown_provider_lists_available(self, settings):
        """Test error message lists available providers."""
        with pytest.raises(ValueError, match="sarvam"):
            get_stt_provider("unknown", settings)
