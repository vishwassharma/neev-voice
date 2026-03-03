"""Sarvam AI Text-to-Speech provider.

Uses the Sarvam AI Bulbul v3 TTS API to synthesize
Hindi speech from text.
"""

import base64
import tempfile
from pathlib import Path

import httpx

from neev_voice.config import NeevSettings
from neev_voice.tts.base import TTSProvider

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


class SarvamTTS(TTSProvider):
    """Sarvam AI Text-to-Speech provider using Bulbul v3.

    Converts text to speech using Sarvam AI's TTS API with
    Hindi voice output.

    Attributes:
        settings: Application settings containing the Sarvam API key.
    """

    def __init__(self, settings: NeevSettings) -> None:
        """Initialize SarvamTTS with application settings.

        Args:
            settings: Application settings containing the Sarvam API key.

        Raises:
            ValueError: If the Sarvam API key is not configured.
        """
        if not settings.sarvam_api_key:
            raise ValueError(
                "Sarvam API key is required for TTS. "
                "Set NEEV_SARVAM_API_KEY env var or get a key from https://dashboard.sarvam.ai"
            )
        self.settings = settings

    async def synthesize(self, text: str) -> Path:
        """Synthesize text to speech using Sarvam AI API.

        Sends text to Sarvam AI's TTS endpoint and saves the
        resulting audio as a WAV file.

        Args:
            text: The text to convert to speech.

        Returns:
            Path to the generated WAV audio file.

        Raises:
            RuntimeError: If the API request fails.
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

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                SARVAM_TTS_URL,
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Sarvam TTS API error (HTTP {response.status_code}): {response.text}"
            )

        result = response.json()
        audios = result.get("audios", [])
        if not audios:
            raise RuntimeError("Sarvam TTS returned no audio data")

        audio_bytes = base64.b64decode(audios[0])

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()

        return Path(tmp.name)
