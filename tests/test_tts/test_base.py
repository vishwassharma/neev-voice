"""Tests for TTS base classes."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from neev_voice.tts.base import TTSProvider


class TestTTSProviderABC:
    """Tests for TTSProvider abstract base class."""

    def test_cannot_instantiate(self):
        """Test that TTSProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TTSProvider()

    def test_subclass_must_implement_synthesize(self):
        """Test that subclass without synthesize raises TypeError."""

        class BadProvider(TTSProvider):
            pass

        with pytest.raises(TypeError):
            BadProvider()

    def test_valid_subclass(self):
        """Test that a proper subclass can be instantiated."""

        class GoodProvider(TTSProvider):
            async def synthesize(self, text: str) -> Path:
                return Path("/tmp/test.wav")

        provider = GoodProvider()
        assert isinstance(provider, TTSProvider)


class TestPlayAudio:
    """Tests for TTSProvider.play_audio static method."""

    def test_play_missing_file_raises(self, tmp_path):
        """Test play_audio raises FileNotFoundError for missing file."""
        missing = tmp_path / "missing.wav"
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            TTSProvider.play_audio(missing)

    def test_play_audio_calls_sounddevice(self, tmp_path, mocker):
        """Test play_audio calls sounddevice.play and wait."""
        # Create a valid WAV file
        wav_path = tmp_path / "test.wav"
        data = np.zeros(1600, dtype=np.int16)
        wavfile.write(str(wav_path), 16000, data)

        mock_play = mocker.patch("neev_voice.tts.base.sd.play")
        mock_wait = mocker.patch("neev_voice.tts.base.sd.wait")

        TTSProvider.play_audio(wav_path)

        mock_play.assert_called_once()
        mock_wait.assert_called_once()

    def test_play_audio_converts_int16(self, tmp_path, mocker):
        """Test play_audio converts int16 to float32."""
        wav_path = tmp_path / "test.wav"
        data = np.array([16384, -16384], dtype=np.int16)
        wavfile.write(str(wav_path), 16000, data)

        mock_play = mocker.patch("neev_voice.tts.base.sd.play")
        mocker.patch("neev_voice.tts.base.sd.wait")

        TTSProvider.play_audio(wav_path)

        played_data = mock_play.call_args[0][0]
        assert played_data.dtype == np.float32
