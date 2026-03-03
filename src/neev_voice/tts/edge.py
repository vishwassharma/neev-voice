"""Edge TTS provider using Microsoft's free edge-tts service.

Provides Hindi text-to-speech using the edge-tts library
with the hi-IN-SwaraNeural voice.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import edge_tts
import structlog

from neev_voice.exceptions import NeevConfigError
from neev_voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from neev_voice.config import NeevSettings

logger = structlog.get_logger(__name__)

__all__ = ["DEFAULT_VOICE", "EdgeTTS", "get_tts_provider"]

DEFAULT_VOICE = "hi-IN-SwaraNeural"


class EdgeTTS(TTSProvider):
    """Edge TTS provider using Microsoft's free TTS service.

    Uses the edge-tts library to synthesize Hindi speech
    with the SwaraNeural voice.

    Attributes:
        voice: The edge-tts voice identifier to use.
    """

    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        """Initialize EdgeTTS with a voice selection.

        Args:
            voice: Edge TTS voice identifier. Defaults to hi-IN-SwaraNeural.
        """
        self.voice = voice

    async def synthesize(self, text: str) -> Path:
        """Synthesize text to speech using edge-tts.

        Generates an MP3 audio file from the given text using
        Microsoft's edge-tts service.

        Args:
            text: The text to convert to speech.

        Returns:
            Path to the generated MP3 audio file.

        Raises:
            RuntimeError: If synthesis fails.
        """
        logger.info("tts_synthesize_started", provider="edge", text_length=len(text))

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        output_path = Path(tmp.name)

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))

        logger.debug("tts_synthesize_done", provider="edge", path=str(output_path))
        return output_path


def get_tts_provider(name: str, settings: NeevSettings | None = None) -> TTSProvider:
    """Factory function to create a TTS provider by name.

    Args:
        name: Name of the provider (e.g., "sarvam", "edge").
        settings: Application settings (required for Sarvam provider).

    Returns:
        An instance of the requested TTS provider.

    Raises:
        NeevConfigError: If the provider name is not recognized.
    """
    from neev_voice.tts.sarvam import SarvamTTS

    def _make_sarvam() -> SarvamTTS:
        if not settings:
            raise NeevConfigError("Settings required for Sarvam TTS provider")
        return SarvamTTS(settings)

    providers: dict[str, Callable[[], TTSProvider]] = {
        "sarvam": _make_sarvam,
        "edge": lambda: EdgeTTS(),
    }

    if name not in providers:
        available = ", ".join(providers.keys())
        raise NeevConfigError(f"Unknown TTS provider '{name}'. Available: {available}")

    return providers[name]()
