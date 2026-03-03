"""Sarvam AI Text-to-Speech provider.

Uses the Sarvam AI Bulbul v3 TTS API to synthesize
Hindi speech from text.
"""

import base64
import tempfile
from pathlib import Path

import httpx
import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from neev_voice.config import NeevSettings
from neev_voice.exceptions import NeevConfigError, NeevTTSError
from neev_voice.tts.base import TTSProvider

logger = structlog.get_logger(__name__)

__all__ = ["SARVAM_TTS_URL", "SarvamTTS"]

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

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
    if isinstance(exc, RuntimeError | NeevTTSError) and "HTTP" in str(exc):
        for code in _TRANSIENT_STATUS_CODES:
            if f"HTTP {code}" in str(exc):
                return True
    return False


class SarvamTTS(TTSProvider):
    """Sarvam AI Text-to-Speech provider using Bulbul v3.

    Converts text to speech using Sarvam AI's TTS API with
    Hindi voice output.  Reuses a single ``httpx.AsyncClient``
    for connection pooling across calls.

    Attributes:
        settings: Application settings containing the Sarvam API key.
    """

    def __init__(self, settings: NeevSettings) -> None:
        """Initialize SarvamTTS with application settings.

        Args:
            settings: Application settings containing the Sarvam API key.

        Raises:
            NeevConfigError: If the Sarvam API key is not configured.
        """
        if not settings.sarvam_api_key:
            raise NeevConfigError(
                "Sarvam API key is required for TTS. "
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_transient_error),
        reraise=True,
    )
    async def synthesize(self, text: str) -> Path:
        """Synthesize text to speech using Sarvam AI API.

        Retries up to 3 times with exponential backoff on transient
        HTTP errors (429, 500, 502, 503, 504, timeouts, connection errors).

        Args:
            text: The text to convert to speech.

        Returns:
            Path to the generated WAV audio file.

        Raises:
            NeevTTSError: If the API request fails after all retries.
        """
        headers = {
            "api-subscription-key": self.settings.sarvam_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "target_language_code": "hi-IN",
            "speaker": "priya",
            "model": "bulbul:v3",
            "output_audio_codec": "wav",
        }

        logger.info("tts_synthesize_started", text_length=len(text))

        client = self._get_client()
        response = await client.post(
            SARVAM_TTS_URL,
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            logger.error("tts_api_error", status_code=response.status_code)
            raise NeevTTSError(
                f"Sarvam TTS API error (HTTP {response.status_code}): {response.text}"
            )

        result = response.json()
        audios = result.get("audios", [])
        if not audios:
            raise NeevTTSError("Sarvam TTS returned no audio data")

        audio_bytes = base64.b64decode(audios[0])

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()

        logger.debug("tts_synthesize_done", audio_bytes=len(audio_bytes), path=tmp.name)
        return Path(tmp.name)
