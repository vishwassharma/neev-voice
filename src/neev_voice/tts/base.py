"""Abstract base class for Text-to-Speech providers.

Defines the interface that all TTS providers must implement,
including synthesis and playback functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


class TTSProvider(ABC):
    """Abstract base class for Text-to-Speech providers.

    All TTS providers must implement the synthesize method to convert
    text to audio files.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> Path:
        """Synthesize text to an audio file.

        Args:
            text: The text to convert to speech.

        Returns:
            Path to the generated audio file.

        Raises:
            RuntimeError: If synthesis fails.
        """
        ...

    @staticmethod
    def play_audio(audio_path: Path) -> None:
        """Play an audio file through the default output device.

        Args:
            audio_path: Path to the WAV audio file to play.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        sample_rate, data = wavfile.read(str(audio_path))

        # Convert to float32 for playback
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0

        sd.play(data, sample_rate)
        sd.wait()
