"""Audio recording module with Voice Activity Detection and Push-to-Talk.

Captures microphone audio using sounddevice with either energy-based silence
detection (RMS threshold) or push-to-talk keyboard control. Push-to-talk
uses KeyboardMonitor to record only while spacebar is held.
"""

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from neev_voice.audio.keyboard import KeyboardMonitor, RecordingState
from neev_voice.config import NeevSettings


class RecordingCancelledError(Exception):
    """Raised when the user cancels a push-to-talk recording with ESC."""


@dataclass
class AudioSegment:
    """Represents a captured audio segment.

    Attributes:
        data: Raw audio data as a numpy array.
        sample_rate: Sample rate in Hz.
        duration: Duration of the audio in seconds.
    """

    data: np.ndarray
    sample_rate: int
    duration: float


@dataclass
class AudioRecorder:
    """Records audio from microphone with VAD-based silence detection.

    Uses energy-based Voice Activity Detection (RMS threshold) to detect
    when the user stops speaking and automatically ends the recording.

    Attributes:
        settings: Application settings containing audio configuration.
        _frames: Internal buffer for collected audio frames.
        _silence_counter: Tracks consecutive silent frames.
        _recording: Whether recording is currently active.
    """

    settings: NeevSettings
    _frames: list = field(default_factory=list, init=False, repr=False)
    _silence_counter: float = field(default=0.0, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def compute_rms(audio_data: np.ndarray) -> float:
        """Compute Root Mean Square of audio data.

        Args:
            audio_data: Audio samples as numpy array.

        Returns:
            RMS value of the audio data.
        """
        if audio_data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))

    def is_silent(self, audio_data: np.ndarray) -> bool:
        """Check if audio data is below the silence threshold.

        Args:
            audio_data: Audio samples as numpy array.

        Returns:
            True if the audio is considered silent.
        """
        return self.compute_rms(audio_data) < self.settings.silence_threshold

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for sounddevice InputStream.

        Collects audio frames and tracks silence duration. Called by
        sounddevice for each audio block.

        Args:
            indata: Recorded audio data.
            frames: Number of frames in this block.
            time_info: Timing information from PortAudio.
            status: Status flags from PortAudio.
        """
        self._frames.append(indata.copy())
        block_duration = frames / self.settings.sample_rate

        if self.is_silent(indata):
            self._silence_counter += block_duration
        else:
            self._silence_counter = 0.0

    async def record_until_silence(self) -> AudioSegment:
        """Record audio from microphone until silence is detected.

        Starts recording and continues until a period of silence
        (configured via settings.silence_duration) is detected.

        Returns:
            AudioSegment containing the recorded audio data.

        Raises:
            RuntimeError: If recording fails or no audio is captured.
        """
        self._frames = []
        self._silence_counter = 0.0
        self._recording = True

        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def callback(
            indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
        ) -> None:
            """Internal callback that signals stop on silence detection."""
            self._audio_callback(indata, frames, time_info, status)
            if self._silence_counter >= self.settings.silence_duration and len(self._frames) > 1:
                loop.call_soon_threadsafe(stop_event.set)

        stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        )

        with stream:
            await stop_event.wait()

        self._recording = False

        if not self._frames:
            raise RuntimeError("No audio captured")

        audio_data = np.concatenate(self._frames, axis=0)
        duration = len(audio_data) / self.settings.sample_rate

        return AudioSegment(
            data=audio_data,
            sample_rate=self.settings.sample_rate,
            duration=duration,
        )

    async def record_push_to_talk(
        self,
        on_state_change: Optional[Callable[[RecordingState], None]] = None,
        kb_monitor: Optional[KeyboardMonitor] = None,
    ) -> AudioSegment:
        """Record audio using push-to-talk (hold SPACEBAR, press ENTER to finish).

        Opens a sounddevice InputStream and only captures frames while
        the spacebar is held (KeyboardMonitor.recording_event is set).
        Pressing Enter finalizes the recording.

        Args:
            on_state_change: Optional callback fired on recording state
                transitions (IDLE -> RECORDING -> PAUSED -> DONE).
            kb_monitor: Optional pre-configured KeyboardMonitor (for testing).
                If None, a new one is created with settings.key_release_timeout.

        Returns:
            AudioSegment containing the captured audio data.

        Raises:
            RuntimeError: If no audio is captured or stdin is not a TTY.
        """
        self._frames = []
        self._recording = True

        loop = asyncio.get_event_loop()
        done_async = asyncio.Event()

        monitor = kb_monitor or KeyboardMonitor(
            release_timeout=self.settings.key_release_timeout,
            on_state_change=on_state_change,
        )

        # Capture the user's callback before overwriting
        user_cb = on_state_change or monitor._on_state_change

        def _combined_cb(state: RecordingState) -> None:
            """Fire both user callback and done/cancel detection."""
            if user_cb:
                user_cb(state)
            if state in (RecordingState.DONE, RecordingState.CANCELLED):
                loop.call_soon_threadsafe(done_async.set)

        monitor._on_state_change = _combined_cb

        def callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            """Audio stream callback — only collect frames when recording."""
            if monitor.recording_event.is_set():
                self._frames.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        )

        try:
            monitor.start(loop=loop)
            with stream:
                await done_async.wait()
        finally:
            monitor.stop()
            self._recording = False

        if monitor.cancelled_event.is_set():
            raise RecordingCancelledError("Recording cancelled by user")

        if not self._frames:
            raise RuntimeError("No audio captured — hold SPACEBAR while speaking")

        audio_data = np.concatenate(self._frames, axis=0)
        duration = len(audio_data) / self.settings.sample_rate

        return AudioSegment(
            data=audio_data,
            sample_rate=self.settings.sample_rate,
            duration=duration,
        )

    @staticmethod
    def chunk_audio(segment: AudioSegment, max_duration: float = 30.0) -> list[AudioSegment]:
        """Split an AudioSegment into chunks of at most max_duration seconds.

        If the segment is shorter than or equal to max_duration, it is
        returned as a single-element list without copying.

        Args:
            segment: The audio segment to split.
            max_duration: Maximum duration per chunk in seconds.

        Returns:
            List of AudioSegment chunks, each at most max_duration seconds.
        """
        if segment.data.size == 0:
            return []

        if segment.duration <= max_duration:
            return [segment]

        chunk_samples = int(max_duration * segment.sample_rate)
        total_samples = segment.data.shape[0]
        chunks: list[AudioSegment] = []

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk_data = segment.data[start:end]
            chunk_duration = (end - start) / segment.sample_rate
            chunks.append(
                AudioSegment(
                    data=chunk_data,
                    sample_rate=segment.sample_rate,
                    duration=chunk_duration,
                )
            )

        return chunks

    @staticmethod
    def save_wav(segment: AudioSegment, path: Path | None = None) -> Path:
        """Save an AudioSegment to a WAV file.

        Args:
            segment: The audio segment to save.
            path: Optional path for the WAV file. If None, a temporary
                file is created.

        Returns:
            Path to the saved WAV file.
        """
        if path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            path = Path(tmp.name)
            tmp.close()

        # Convert float32 [-1.0, 1.0] to int16
        audio_int16 = np.clip(segment.data * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(str(path), segment.sample_rate, audio_int16)
        return path
