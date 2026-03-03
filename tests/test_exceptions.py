"""Tests for the custom exception hierarchy."""

import pytest

from neev_voice.exceptions import (
    NeevConfigError,
    NeevError,
    NeevLLMError,
    NeevSTTError,
    NeevTTSError,
    RecordingCancelledError,
)


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    def test_neev_error_is_exception(self):
        """Test NeevError inherits from Exception."""
        assert issubclass(NeevError, Exception)

    def test_config_error_is_neev_error(self):
        """Test NeevConfigError inherits from NeevError."""
        assert issubclass(NeevConfigError, NeevError)

    def test_stt_error_is_neev_error(self):
        """Test NeevSTTError inherits from NeevError."""
        assert issubclass(NeevSTTError, NeevError)

    def test_tts_error_is_neev_error(self):
        """Test NeevTTSError inherits from NeevError."""
        assert issubclass(NeevTTSError, NeevError)

    def test_llm_error_is_neev_error(self):
        """Test NeevLLMError inherits from NeevError."""
        assert issubclass(NeevLLMError, NeevError)

    def test_recording_cancelled_is_neev_error(self):
        """Test RecordingCancelledError inherits from NeevError."""
        assert issubclass(RecordingCancelledError, NeevError)


class TestExceptionMessages:
    """Tests for exception message propagation."""

    def test_neev_error_message(self):
        """Test NeevError carries a message."""
        err = NeevError("something went wrong")
        assert str(err) == "something went wrong"

    def test_stt_error_message(self):
        """Test NeevSTTError carries a message."""
        err = NeevSTTError("transcription failed")
        assert str(err) == "transcription failed"

    def test_tts_error_message(self):
        """Test NeevTTSError carries a message."""
        err = NeevTTSError("synthesis failed")
        assert str(err) == "synthesis failed"

    def test_recording_cancelled_message(self):
        """Test RecordingCancelledError carries a message."""
        err = RecordingCancelledError("user pressed ESC")
        assert str(err) == "user pressed ESC"


class TestExceptionCatching:
    """Tests for catching exceptions at different hierarchy levels."""

    def test_catch_stt_as_neev_error(self):
        """Test NeevSTTError can be caught as NeevError."""
        with pytest.raises(NeevError):
            raise NeevSTTError("api failed")

    def test_catch_tts_as_neev_error(self):
        """Test NeevTTSError can be caught as NeevError."""
        with pytest.raises(NeevError):
            raise NeevTTSError("api failed")

    def test_catch_config_as_neev_error(self):
        """Test NeevConfigError can be caught as NeevError."""
        with pytest.raises(NeevError):
            raise NeevConfigError("bad config")

    def test_catch_recording_cancelled_as_neev_error(self):
        """Test RecordingCancelledError can be caught as NeevError."""
        with pytest.raises(NeevError):
            raise RecordingCancelledError("cancelled")


class TestBackwardsCompatibility:
    """Tests for backwards-compatible import of RecordingCancelledError."""

    def test_import_from_recorder(self):
        """Test RecordingCancelledError is re-exported from audio.recorder."""
        from neev_voice.audio.recorder import RecordingCancelledError as RCE

        assert RCE is RecordingCancelledError
