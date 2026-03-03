"""Edge TTS provider using Microsoft's free edge-tts service.

Provides Hindi text-to-speech using the edge-tts library
with the hi-IN-SwaraNeural voice.
"""

import tempfile
from pathlib import Path

import edge_tts

from neev_voice.tts.base import TTSProvider

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
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        output_path = Path(tmp.name)

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))

        return output_path


def get_tts_provider(name: str, settings: "NeevSettings | None" = None) -> TTSProvider:  # noqa: F821
    """Factory function to create a TTS provider by name.

    Args:
        name: Name of the provider (e.g., "sarvam", "edge").
        settings: Application settings (required for Sarvam provider).

    Returns:
        An instance of the requested TTS provider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    from neev_voice.tts.sarvam import SarvamTTS

    providers = {
        "sarvam": lambda: (
            SarvamTTS(settings)
            if settings
            else (_ for _ in ()).throw(ValueError("Settings required for Sarvam TTS provider"))
        ),
        "edge": lambda: EdgeTTS(),
    }

    if name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown TTS provider '{name}'. Available: {available}")

    return providers[name]()
