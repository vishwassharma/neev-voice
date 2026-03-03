"""Sarvam AI Speech-to-Text provider.

Uses the Sarvam AI REST API with the saaras:v3 model.
Supports multiple transcription modes: translate (Hindi to English),
codemix (Hindi-English mixed), and formal (Devanagari).
Audio longer than the configured max duration (default 30s) is
automatically chunked, transcribed per-chunk, and merged.
"""

from pathlib import Path

import httpx
import numpy as np
import structlog
from scipy.io import wavfile
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from neev_voice.audio.recorder import AudioRecorder, AudioSegment
from neev_voice.config import NeevSettings
from neev_voice.exceptions import NeevConfigError, NeevSTTError
from neev_voice.stt.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)

__all__ = ["SARVAM_STT_URL", "SarvamSTT", "get_stt_provider"]

_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_transient_error(exc: BaseException) -> bool:
    """Check if an exception is a transient HTTP error worth retrying.

    Args:
        exc: The exception to check.

    Returns:
        True if the error is transient (timeout, connection, or server error).
    """
    if isinstance(exc, httpx.ConnectError | httpx.TimeoutException):
        return True
    if isinstance(exc, RuntimeError | NeevSTTError) and "HTTP" in str(exc):
        for code in _TRANSIENT_STATUS_CODES:
            if f"HTTP {code}" in str(exc):
                return True
    return False


SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT(STTProvider):
    """Sarvam AI Speech-to-Text provider.

    Transcribes audio using Sarvam AI's saaras:v3 model with a
    configurable output mode (translate, codemix, or formal).
    Audio exceeding ``settings.stt_max_audio_duration`` is split
    into chunks, each transcribed separately, then merged.

    Attributes:
        settings: Application settings containing the Sarvam API key and STT mode.
    """

    def __init__(self, settings: NeevSettings) -> None:
        """Initialize SarvamSTT with application settings.

        Args:
            settings: Application settings containing the Sarvam API key.

        Raises:
            NeevConfigError: If the Sarvam API key is not configured.
        """
        if not settings.sarvam_api_key:
            raise NeevConfigError(
                "Sarvam API key is required. "
                "Set NEEV_SARVAM_API_KEY env var or get a key from https://dashboard.sarvam.ai"
            )
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared httpx.AsyncClient for connection pooling.

        Returns:
            The shared async HTTP client.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client and release connections."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file using Sarvam AI API.

        If the audio is longer than ``settings.stt_max_audio_duration``,
        it is split into chunks, each chunk is transcribed via
        ``_transcribe_single``, and the results are merged (text
        concatenated, confidence averaged).

        Args:
            audio_path: Path to the WAV audio file to transcribe.

        Returns:
            TranscriptionResult with transcribed text and metadata.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            NeevSTTError: If the API request fails.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        max_duration = self.settings.stt_max_audio_duration
        sample_rate, raw_data = wavfile.read(str(audio_path))
        duration = len(raw_data) / sample_rate
        logger.info("transcribe_started", audio_path=str(audio_path), duration_s=round(duration, 2))

        if duration <= max_duration:
            return await self._transcribe_single(audio_path)

        # Convert int16 WAV data to float32 for AudioSegment
        float_data = raw_data.astype(np.float32) / 32767.0
        segment = AudioSegment(data=float_data, sample_rate=sample_rate, duration=duration)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=max_duration)
        logger.info("transcribe_chunked", num_chunks=len(chunks))

        results: list[TranscriptionResult] = []
        for chunk in chunks:
            chunk_path = AudioRecorder.save_wav(chunk)
            try:
                result = await self._transcribe_single(chunk_path)
                results.append(result)
            finally:
                chunk_path.unlink(missing_ok=True)

        return self._merge_results(results)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_transient_error),
        reraise=True,
    )
    async def _transcribe_single(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe a single audio file via the Sarvam API.

        Retries up to 3 times with exponential backoff on transient
        HTTP errors (429, 500, 502, 503, 504, timeouts, connection errors).

        Args:
            audio_path: Path to the WAV audio file (must be ≤ max duration).

        Returns:
            TranscriptionResult with transcribed text and metadata.

        Raises:
            NeevSTTError: If the API request fails after all retries.
        """
        headers = {
            "api-subscription-key": self.settings.sarvam_api_key,
        }

        client = self._get_client()
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {
                "model": "saaras:v3",
                "language_code": "hi-IN",
                "mode": self.settings.stt_mode.value,
            }

            response = await client.post(
                SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
            )

        if response.status_code != 200:
            logger.error("stt_api_error", status_code=response.status_code)
            raise NeevSTTError(
                f"Sarvam STT API error (HTTP {response.status_code}): {response.text}"
            )

        result = response.json()
        logger.debug(
            "stt_api_success",
            confidence=result.get("confidence", 0.0),
            text_length=len(result.get("transcript", "")),
        )

        return TranscriptionResult(
            text=result.get("transcript", ""),
            language=result.get("language_code", "hi-IN"),
            confidence=result.get("confidence", 0.0),
            provider="sarvam",
        )

    @staticmethod
    def _merge_results(results: list[TranscriptionResult]) -> TranscriptionResult:
        """Merge multiple transcription results into one.

        Concatenates text with spaces, averages confidence scores,
        and keeps the language and provider from the first result.

        Args:
            results: List of TranscriptionResult objects to merge.

        Returns:
            A single merged TranscriptionResult.
        """
        merged_text = " ".join(r.text for r in results if r.text)
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
        return TranscriptionResult(
            text=merged_text,
            language=results[0].language if results else "hi-IN",
            confidence=avg_confidence,
            provider=results[0].provider if results else "sarvam",
        )


def get_stt_provider(name: str, settings: NeevSettings) -> STTProvider:
    """Factory function to create an STT provider by name.

    Args:
        name: Name of the provider (e.g., "sarvam").
        settings: Application settings.

    Returns:
        An instance of the requested STT provider.

    Raises:
        NeevConfigError: If the provider name is not recognized.
    """
    providers = {
        "sarvam": lambda: SarvamSTT(settings),
    }

    if name not in providers:
        available = ", ".join(providers.keys())
        raise NeevConfigError(f"Unknown STT provider '{name}'. Available: {available}")

    return providers[name]()
