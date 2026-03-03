"""Tests for audio recorder module."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neev_voice.audio.keyboard import RecordingState
from neev_voice.audio.recorder import AudioRecorder, AudioSegment, RecordingCancelledError
from neev_voice.config import NeevSettings


@pytest.fixture
def settings():
    """Create test settings."""
    return NeevSettings(
        sarvam_api_key="test-key",
        sample_rate=16000,
        silence_threshold=0.03,
        silence_duration=1.0,
    )


@pytest.fixture
def recorder(settings):
    """Create an AudioRecorder with test settings."""
    return AudioRecorder(settings=settings)


class TestAudioSegment:
    """Tests for AudioSegment dataclass."""

    def test_creation(self):
        """Test AudioSegment creation with basic attributes."""
        data = np.zeros((1600, 1), dtype=np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=0.1)
        assert segment.sample_rate == 16000
        assert segment.duration == 0.1
        assert segment.data.shape == (1600, 1)

    def test_data_type(self):
        """Test AudioSegment accepts different array shapes."""
        data = np.random.randn(8000).astype(np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=0.5)
        assert segment.data.dtype == np.float32


class TestComputeRMS:
    """Tests for RMS computation."""

    def test_silence(self):
        """Test RMS of zero audio is zero."""
        data = np.zeros(1000, dtype=np.float32)
        assert AudioRecorder.compute_rms(data) == 0.0

    def test_known_value(self):
        """Test RMS of constant signal equals the constant."""
        data = np.full(1000, 0.5, dtype=np.float32)
        rms = AudioRecorder.compute_rms(data)
        assert abs(rms - 0.5) < 1e-6

    def test_empty_array(self):
        """Test RMS of empty array returns zero."""
        data = np.array([], dtype=np.float32)
        assert AudioRecorder.compute_rms(data) == 0.0

    def test_sine_wave(self):
        """Test RMS of sine wave is approximately 1/sqrt(2)."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        data = np.sin(2 * np.pi * 440 * t)
        rms = AudioRecorder.compute_rms(data)
        expected = 1.0 / np.sqrt(2)
        assert abs(rms - expected) < 0.01


class TestIsSilent:
    """Tests for silence detection."""

    def test_silent_audio(self, recorder):
        """Test that zero audio is detected as silent."""
        data = np.zeros(1000, dtype=np.float32)
        assert recorder.is_silent(data) is True

    def test_loud_audio(self, recorder):
        """Test that loud audio is not detected as silent."""
        data = np.full(1000, 0.5, dtype=np.float32)
        assert recorder.is_silent(data) is False

    def test_near_threshold(self, recorder):
        """Test audio just below threshold is silent."""
        data = np.full(1000, 0.02, dtype=np.float32)
        assert recorder.is_silent(data) is True

    def test_above_threshold(self, recorder):
        """Test audio just above threshold is not silent."""
        data = np.full(1000, 0.05, dtype=np.float32)
        assert recorder.is_silent(data) is False


class TestAudioCallback:
    """Tests for the audio callback mechanism."""

    def test_frames_collected(self, recorder):
        """Test that callback collects frames."""
        indata = np.random.randn(1024, 1).astype(np.float32) * 0.5
        recorder._audio_callback(indata, 1024, None, None)
        assert len(recorder._frames) == 1
        np.testing.assert_array_equal(recorder._frames[0], indata)

    def test_silence_counter_increments(self, recorder):
        """Test silence counter increments on silent frames."""
        silent_data = np.zeros((1024, 1), dtype=np.float32)
        recorder._audio_callback(silent_data, 1024, None, None)
        expected_duration = 1024 / recorder.settings.sample_rate
        assert abs(recorder._silence_counter - expected_duration) < 1e-6

    def test_silence_counter_resets_on_speech(self, recorder):
        """Test silence counter resets when non-silent audio arrives."""
        # First, accumulate silence
        silent_data = np.zeros((1024, 1), dtype=np.float32)
        recorder._audio_callback(silent_data, 1024, None, None)
        assert recorder._silence_counter > 0

        # Then, send loud audio
        loud_data = np.full((1024, 1), 0.5, dtype=np.float32)
        recorder._audio_callback(loud_data, 1024, None, None)
        assert recorder._silence_counter == 0.0

    def test_multiple_frames(self, recorder):
        """Test multiple callback invocations collect all frames."""
        for _ in range(5):
            indata = np.random.randn(1024, 1).astype(np.float32)
            recorder._audio_callback(indata, 1024, None, None)
        assert len(recorder._frames) == 5


class TestSaveWav:
    """Tests for WAV file saving."""

    def test_save_to_path(self, recorder, tmp_path):
        """Test saving audio segment to a specific path."""
        data = np.random.randn(16000, 1).astype(np.float32) * 0.5
        segment = AudioSegment(data=data, sample_rate=16000, duration=1.0)
        wav_path = tmp_path / "test.wav"
        result = AudioRecorder.save_wav(segment, wav_path)
        assert result == wav_path
        assert wav_path.exists()
        assert wav_path.stat().st_size > 0

    def test_save_to_temp(self, recorder):
        """Test saving audio segment to a temporary file."""
        data = np.random.randn(16000, 1).astype(np.float32) * 0.5
        segment = AudioSegment(data=data, sample_rate=16000, duration=1.0)
        result = AudioRecorder.save_wav(segment)
        assert result.exists()
        assert result.suffix == ".wav"
        # Clean up
        result.unlink()

    def test_wav_content_readable(self, recorder, tmp_path):
        """Test that saved WAV can be read back."""
        from scipy.io import wavfile

        data = np.random.randn(16000, 1).astype(np.float32) * 0.5
        segment = AudioSegment(data=data, sample_rate=16000, duration=1.0)
        wav_path = tmp_path / "test.wav"
        AudioRecorder.save_wav(segment, wav_path)

        rate, read_data = wavfile.read(str(wav_path))
        assert rate == 16000
        assert read_data.dtype == np.int16
        assert len(read_data) == 16000


class MockKeyboardMonitor:
    """Mock KeyboardMonitor for push-to-talk tests.

    Simulates keyboard events by setting recording_event during
    specified frames, then triggering done after a configurable
    number of audio callback invocations.

    Attributes:
        recording_event: Threading event for recording state.
        done_event: Threading event for done state.
        state: Current recording state.
        _on_state_change: Callback for state changes.
        _frames_to_record: Number of frames to simulate recording.
        _frame_count: Counter for callback invocations.
        _loop: Event loop for async signaling.
    """

    def __init__(self, frames_to_record: int = 3) -> None:
        """Initialize mock monitor.

        Args:
            frames_to_record: Number of audio frames to record before stopping.
        """
        self.recording_event = threading.Event()
        self.done_event = threading.Event()
        self.cancelled_event = threading.Event()
        self.state = RecordingState.IDLE
        self._on_state_change = None
        self._frames_to_record = frames_to_record
        self._frame_count = 0
        self._loop = None

    def start(self, loop=None) -> None:
        """Start mock monitor and begin recording immediately.

        Args:
            loop: Event loop (stored for signaling done).
        """
        self._loop = loop
        self.recording_event.set()
        self.state = RecordingState.RECORDING

    def stop(self) -> None:
        """Stop mock monitor."""
        pass

    def tick(self) -> None:
        """Called by test to simulate frame progression.

        After frames_to_record frames, triggers DONE state and
        fires the state change callback.
        """
        self._frame_count += 1
        if self._frame_count >= self._frames_to_record:
            self.recording_event.clear()
            self.done_event.set()
            self.state = RecordingState.DONE
            if self._on_state_change and self._loop:
                self._loop.call_soon_threadsafe(self._on_state_change, RecordingState.DONE)


class TestPushToTalk:
    """Tests for push-to-talk recording."""

    @pytest.fixture
    def ptt_settings(self):
        """Create settings with key_release_timeout."""
        return NeevSettings(
            sarvam_api_key="test-key",
            sample_rate=16000,
            silence_threshold=0.03,
            silence_duration=1.0,
            key_release_timeout=0.15,
        )

    @pytest.fixture
    def ptt_recorder(self, ptt_settings):
        """Create an AudioRecorder with push-to-talk settings."""
        return AudioRecorder(settings=ptt_settings)

    async def test_push_to_talk_collects_frames(self, ptt_recorder):
        """Test that push-to-talk only collects frames when recording."""
        mock_kb = MockKeyboardMonitor(frames_to_record=3)

        audio_data = np.random.randn(1024, 1).astype(np.float32) * 0.3
        call_count = 0

        original_start = mock_kb.start

        def patched_start(loop=None):
            """Start monitor and schedule frame ticks."""
            original_start(loop=loop)
            mock_kb._loop = loop

            def feed_frames():
                nonlocal call_count
                for _ in range(3):
                    ptt_recorder._frames.append(audio_data.copy())
                    call_count += 1
                    mock_kb.tick()

            loop.call_soon_threadsafe(feed_frames)

        mock_kb.start = patched_start

        with patch("sounddevice.InputStream") as mock_stream:
            mock_ctx = MagicMock()
            mock_stream.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            segment = await ptt_recorder.record_push_to_talk(kb_monitor=mock_kb)

        assert segment.data.shape[0] > 0
        assert segment.sample_rate == 16000

    async def test_push_to_talk_no_audio_raises(self, ptt_recorder):
        """Test push-to-talk raises when no audio captured."""
        mock_kb = MockKeyboardMonitor(frames_to_record=0)

        original_start = mock_kb.start

        def patched_start(loop=None):
            """Start monitor and immediately trigger done."""
            original_start(loop=loop)
            mock_kb._loop = loop
            mock_kb.recording_event.clear()
            mock_kb.done_event.set()
            mock_kb.state = RecordingState.DONE
            if mock_kb._on_state_change:
                loop.call_soon_threadsafe(mock_kb._on_state_change, RecordingState.DONE)

        mock_kb.start = patched_start

        with patch("sounddevice.InputStream") as mock_stream:
            mock_ctx = MagicMock()
            mock_stream.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="No audio captured"):
                await ptt_recorder.record_push_to_talk(kb_monitor=mock_kb)

    async def test_push_to_talk_state_callback(self, ptt_recorder):
        """Test push-to-talk fires state change callback."""
        states = []
        mock_kb = MockKeyboardMonitor(frames_to_record=1)

        audio_data = np.random.randn(1024, 1).astype(np.float32) * 0.3
        original_start = mock_kb.start

        def patched_start(loop=None):
            """Start monitor and feed one frame."""
            original_start(loop=loop)
            mock_kb._loop = loop

            def feed():
                ptt_recorder._frames.append(audio_data.copy())
                mock_kb.tick()

            loop.call_soon_threadsafe(feed)

        mock_kb.start = patched_start

        with patch("sounddevice.InputStream") as mock_stream:
            mock_ctx = MagicMock()
            mock_stream.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            await ptt_recorder.record_push_to_talk(
                on_state_change=lambda s: states.append(s),
                kb_monitor=mock_kb,
            )

        assert RecordingState.DONE in states

    async def test_push_to_talk_stops_monitor_on_completion(self, ptt_recorder):
        """Test monitor is stopped after recording completes."""
        mock_kb = MockKeyboardMonitor(frames_to_record=1)
        stop_called = False
        original_stop = mock_kb.stop

        def tracked_stop():
            nonlocal stop_called
            stop_called = True
            original_stop()

        mock_kb.stop = tracked_stop

        audio_data = np.random.randn(1024, 1).astype(np.float32) * 0.3
        original_start = mock_kb.start

        def patched_start(loop=None):
            """Start monitor and feed one frame."""
            original_start(loop=loop)
            mock_kb._loop = loop

            def feed():
                ptt_recorder._frames.append(audio_data.copy())
                mock_kb.tick()

            loop.call_soon_threadsafe(feed)

        mock_kb.start = patched_start

        with patch("sounddevice.InputStream") as mock_stream:
            mock_ctx = MagicMock()
            mock_stream.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            await ptt_recorder.record_push_to_talk(kb_monitor=mock_kb)

        assert stop_called

    async def test_push_to_talk_cancelled_raises(self, ptt_recorder):
        """Test push-to-talk raises RecordingCancelledError when ESC pressed."""
        mock_kb = MockKeyboardMonitor(frames_to_record=0)

        original_start = mock_kb.start

        def patched_start(loop=None):
            """Start monitor and immediately trigger cancelled."""
            original_start(loop=loop)
            mock_kb._loop = loop
            mock_kb.recording_event.clear()
            mock_kb.cancelled_event.set()
            mock_kb.state = RecordingState.CANCELLED
            if mock_kb._on_state_change:
                loop.call_soon_threadsafe(mock_kb._on_state_change, RecordingState.CANCELLED)

        mock_kb.start = patched_start

        with patch("sounddevice.InputStream") as mock_stream:
            mock_ctx = MagicMock()
            mock_stream.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RecordingCancelledError, match="cancelled"):
                await ptt_recorder.record_push_to_talk(kb_monitor=mock_kb)

    async def test_push_to_talk_cancelled_with_frames_still_raises(self, ptt_recorder):
        """Test cancellation raises even if some audio was captured."""
        mock_kb = MockKeyboardMonitor(frames_to_record=2)
        audio_data = np.random.randn(1024, 1).astype(np.float32) * 0.3

        original_start = mock_kb.start

        def patched_start(loop=None):
            """Start monitor, feed frames, then cancel."""
            original_start(loop=loop)
            mock_kb._loop = loop

            def feed_and_cancel():
                ptt_recorder._frames.append(audio_data.copy())
                mock_kb.recording_event.clear()
                mock_kb.cancelled_event.set()
                mock_kb.state = RecordingState.CANCELLED
                if mock_kb._on_state_change:
                    mock_kb._on_state_change(RecordingState.CANCELLED)

            loop.call_soon_threadsafe(feed_and_cancel)

        mock_kb.start = patched_start

        with patch("sounddevice.InputStream") as mock_stream:
            mock_ctx = MagicMock()
            mock_stream.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RecordingCancelledError):
                await ptt_recorder.record_push_to_talk(kb_monitor=mock_kb)


class TestChunkAudio:
    """Tests for AudioRecorder.chunk_audio static method."""

    def test_short_audio_returns_single(self):
        """Test audio shorter than max_duration returns single segment."""
        data = np.random.randn(16000, 1).astype(np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=1.0)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=30.0)
        assert len(chunks) == 1
        assert chunks[0] is segment

    def test_exact_duration_returns_single(self):
        """Test audio exactly at max_duration returns single segment."""
        data = np.random.randn(16000 * 30, 1).astype(np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=30.0)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=30.0)
        assert len(chunks) == 1
        assert chunks[0] is segment

    def test_60s_audio_returns_two_chunks(self):
        """Test 60s audio splits into 2 equal chunks of 30s."""
        data = np.random.randn(16000 * 60, 1).astype(np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=60.0)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=30.0)
        assert len(chunks) == 2
        assert chunks[0].data.shape[0] == 16000 * 30
        assert chunks[1].data.shape[0] == 16000 * 30

    def test_45s_audio_returns_two_unequal_chunks(self):
        """Test 45s audio splits into 30s + 15s chunks."""
        data = np.random.randn(16000 * 45, 1).astype(np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=45.0)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=30.0)
        assert len(chunks) == 2
        assert chunks[0].data.shape[0] == 16000 * 30
        assert chunks[1].data.shape[0] == 16000 * 15

    def test_empty_audio_returns_empty_list(self):
        """Test empty audio returns empty list."""
        data = np.array([], dtype=np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=0.0)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=30.0)
        assert chunks == []

    def test_chunk_sample_rates_preserved(self):
        """Test all chunks have the same sample rate as the original."""
        data = np.random.randn(16000 * 45, 1).astype(np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=45.0)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=30.0)
        for chunk in chunks:
            assert chunk.sample_rate == 16000

    def test_chunk_durations_correct(self):
        """Test chunk durations are calculated correctly."""
        data = np.random.randn(16000 * 45, 1).astype(np.float32)
        segment = AudioSegment(data=data, sample_rate=16000, duration=45.0)
        chunks = AudioRecorder.chunk_audio(segment, max_duration=30.0)
        assert abs(chunks[0].duration - 30.0) < 1e-6
        assert abs(chunks[1].duration - 15.0) < 1e-6
