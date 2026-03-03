"""Shared test fixtures for neev-voice test suite.

Provides common fixtures including mock settings, temporary directories,
sample audio data, and mocked provider instances.
"""

import sys
from unittest.mock import MagicMock

if "sounddevice" not in sys.modules:
    _mock_sd = MagicMock()
    _mock_sd.CallbackFlags = MagicMock
    sys.modules["sounddevice"] = _mock_sd

import numpy as np
import pytest

from neev_voice.config import NeevSettings


@pytest.fixture
def default_settings():
    """Create default NeevSettings for testing.

    Returns:
        NeevSettings with test API key and default values.
    """
    return NeevSettings(sarvam_api_key="test-api-key")


@pytest.fixture
def settings_no_key():
    """Create NeevSettings without API key.

    Returns:
        NeevSettings with empty API key.
    """
    return NeevSettings(sarvam_api_key="")


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing.

    Returns:
        Numpy array of 1 second of silence at 16kHz.
    """
    return np.zeros((16000, 1), dtype=np.float32)


@pytest.fixture
def sample_audio_with_speech():
    """Generate sample audio data with simulated speech.

    Returns:
        Numpy array of 1 second of a 440Hz sine wave at 16kHz.
    """
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).reshape(-1, 1)


@pytest.fixture
def temp_wav_file(tmp_path, sample_audio_data):
    """Create a temporary WAV file with sample audio.

    Args:
        tmp_path: Pytest temporary directory.
        sample_audio_data: Sample audio numpy array.

    Returns:
        Path to the temporary WAV file.
    """
    from scipy.io import wavfile

    wav_path = tmp_path / "test.wav"
    audio_int16 = (sample_audio_data * 32767).astype(np.int16)
    wavfile.write(str(wav_path), 16000, audio_int16)
    return wav_path


@pytest.fixture
def sample_document(tmp_path):
    """Create a sample markdown document for discussion testing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the sample document.
    """
    doc_path = tmp_path / "plan.md"
    doc_path.write_text(
        "## Architecture\n"
        "We will use a microservices architecture.\n\n"
        "## Database\n"
        "PostgreSQL will be the primary database.\n\n"
        "## Deployment\n"
        "We will deploy on AWS ECS.\n"
    )
    return doc_path
