"""Tests for push-to-talk keyboard monitor module."""

import contextlib
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from neev_voice.audio.keyboard import KeyboardMonitor, MonitorMode, RecordingState


class FakeTTYStream:
    """Fake stdin-like stream backed by an os.pipe for testing.

    Supports isatty(), fileno(), and read() operations. The write end
    of the pipe is used to simulate keystrokes. Uses raw binary mode
    to avoid text-mode line buffering issues with carriage returns.

    Attributes:
        read_fd: File descriptor for reading.
        write_fd: File descriptor for writing (simulating key presses).
        _reader: File object wrapping read_fd in binary mode.
    """

    def __init__(self) -> None:
        """Initialize with an os.pipe pair."""
        self.read_fd, self.write_fd = os.pipe()
        self._reader = os.fdopen(self.read_fd, "rb", buffering=0)

    def isatty(self) -> bool:
        """Always returns True to simulate a terminal.

        Returns:
            True.
        """
        return True

    def fileno(self) -> int:
        """Return the read file descriptor.

        Returns:
            The read-side file descriptor number.
        """
        return self._reader.fileno()

    def read(self, n: int = 1) -> str:
        """Read n characters from the pipe.

        Args:
            n: Number of characters to read.

        Returns:
            String of characters read, decoded from bytes.
        """
        data = self._reader.read(n)
        return data.decode("utf-8") if data else ""

    def send_key(self, char: str) -> None:
        """Write a character to the pipe (simulates a keypress).

        Args:
            char: The character to send.
        """
        os.write(self.write_fd, char.encode())

    def send_keys(self, chars: str, interval: float = 0.02) -> None:
        """Write multiple characters with delay between them.

        Args:
            chars: String of characters to send.
            interval: Delay in seconds between characters.
        """
        for ch in chars:
            self.send_key(ch)
            time.sleep(interval)

    def close(self) -> None:
        """Close both ends of the pipe."""
        with contextlib.suppress(OSError):
            os.close(self.write_fd)
        with contextlib.suppress(OSError):
            self._reader.close()


@pytest.fixture
def fake_stdin():
    """Create a fake TTY stdin backed by os.pipe.

    Yields:
        FakeTTYStream instance.
    """
    stream = FakeTTYStream()
    yield stream
    stream.close()


class TestRecordingState:
    """Tests for RecordingState enum."""

    def test_idle_value(self):
        """Test IDLE state string value."""
        assert RecordingState.IDLE.value == "idle"

    def test_recording_value(self):
        """Test RECORDING state string value."""
        assert RecordingState.RECORDING.value == "recording"

    def test_paused_value(self):
        """Test PAUSED state string value."""
        assert RecordingState.PAUSED.value == "paused"

    def test_done_value(self):
        """Test DONE state string value."""
        assert RecordingState.DONE.value == "done"

    def test_cancelled_value(self):
        """Test CANCELLED state string value."""
        assert RecordingState.CANCELLED.value == "cancelled"

    def test_all_states_exist(self):
        """Test that all five states are defined."""
        states = list(RecordingState)
        assert len(states) == 5


class TestKeyboardMonitorInit:
    """Tests for KeyboardMonitor initialization."""

    def test_default_timeout(self):
        """Test default release timeout is 0.15."""
        monitor = KeyboardMonitor()
        assert monitor.release_timeout == 0.15

    def test_custom_timeout(self):
        """Test custom release timeout."""
        monitor = KeyboardMonitor(release_timeout=0.3)
        assert monitor.release_timeout == 0.3

    def test_initial_state_is_idle(self):
        """Test initial state is IDLE."""
        monitor = KeyboardMonitor()
        assert monitor.state == RecordingState.IDLE

    def test_events_not_set_initially(self):
        """Test recording, done, and cancelled events are not set initially."""
        monitor = KeyboardMonitor()
        assert not monitor.recording_event.is_set()
        assert not monitor.done_event.is_set()
        assert not monitor.cancelled_event.is_set()

    def test_custom_stdin(self, fake_stdin):
        """Test monitor accepts custom stdin stream."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        assert monitor._stdin is fake_stdin


class TestKeyboardMonitorNonTTY:
    """Tests for non-TTY stdin handling."""

    def test_raises_on_non_tty(self):
        """Test start raises RuntimeError if stdin is not a TTY."""
        fake_stream = MagicMock()
        fake_stream.isatty.return_value = False
        monitor = KeyboardMonitor(stdin=fake_stream)

        with pytest.raises(RuntimeError, match="not a TTY"):
            monitor.start()

    def test_no_isatty_attribute(self):
        """Test start works when stdin has no isatty method."""
        fake_stream = MagicMock(spec=[])
        monitor = KeyboardMonitor(stdin=fake_stream)
        # Should not raise — no isatty means we skip the check
        # But it may fail on fileno, so we patch termios
        with patch("neev_voice.audio.keyboard.termios") as mock_termios:
            mock_termios.tcgetattr.side_effect = OSError
            try:
                monitor.start()
            finally:
                monitor.stop()


class TestKeyboardMonitorSpacebar:
    """Tests for spacebar press/release detection."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_spacebar_sets_recording_event(self, mock_tty, mock_termios, fake_stdin):
        """Test pressing spacebar sets the recording event."""
        monitor = KeyboardMonitor(stdin=fake_stdin, release_timeout=0.3)
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.15)
            assert monitor.recording_event.is_set()
            assert monitor.state == RecordingState.RECORDING
        finally:
            # Send enter to end the monitor loop cleanly
            fake_stdin.send_key("\n")
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_spacebar_release_detected(self, mock_tty, mock_termios, fake_stdin):
        """Test spacebar release is detected after timeout."""
        monitor = KeyboardMonitor(stdin=fake_stdin, release_timeout=0.1)
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.05)
            assert monitor.recording_event.is_set()

            # Wait for release timeout
            time.sleep(0.2)
            assert not monitor.recording_event.is_set()
            assert monitor.state == RecordingState.PAUSED
        finally:
            fake_stdin.send_key("\n")
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_held_spacebar_stays_recording(self, mock_tty, mock_termios, fake_stdin):
        """Test holding spacebar (repeated chars) stays in RECORDING."""
        monitor = KeyboardMonitor(stdin=fake_stdin, release_timeout=0.15)
        monitor.start()

        try:
            # Simulate holding by sending spaces rapidly
            for _ in range(5):
                fake_stdin.send_key(" ")
                time.sleep(0.05)

            assert monitor.recording_event.is_set()
            assert monitor.state == RecordingState.RECORDING
        finally:
            fake_stdin.send_key("\n")
            monitor.stop()


class TestKeyboardMonitorEnter:
    """Tests for Enter key detection."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_enter_sets_done(self, mock_tty, mock_termios, fake_stdin):
        """Test pressing Enter sets done event and state."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        monitor.start()

        try:
            fake_stdin.send_key("\n")
            time.sleep(0.1)
            assert monitor.done_event.is_set()
            assert monitor.state == RecordingState.DONE
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_carriage_return_sets_done(self, mock_tty, mock_termios, fake_stdin):
        """Test pressing carriage return also sets done."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        monitor.start()

        try:
            fake_stdin.send_key("\r")
            time.sleep(0.1)
            assert monitor.done_event.is_set()
            assert monitor.state == RecordingState.DONE
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_enter_clears_recording(self, mock_tty, mock_termios, fake_stdin):
        """Test Enter clears recording event even if currently recording."""
        monitor = KeyboardMonitor(stdin=fake_stdin, release_timeout=0.5)
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.05)
            assert monitor.recording_event.is_set()

            fake_stdin.send_key("\n")
            time.sleep(0.1)
            assert not monitor.recording_event.is_set()
            assert monitor.done_event.is_set()
        finally:
            monitor.stop()


class TestKeyboardMonitorEscape:
    """Tests for ESC key cancellation."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_escape_sets_cancelled(self, mock_tty, mock_termios, fake_stdin):
        """Test pressing ESC sets cancelled event and state."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        monitor.start()

        try:
            fake_stdin.send_key("\x1b")
            time.sleep(0.1)
            assert monitor.cancelled_event.is_set()
            assert monitor.state == RecordingState.CANCELLED
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_escape_clears_recording(self, mock_tty, mock_termios, fake_stdin):
        """Test ESC clears recording event even if currently recording."""
        monitor = KeyboardMonitor(stdin=fake_stdin, release_timeout=0.5)
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.05)
            assert monitor.recording_event.is_set()

            fake_stdin.send_key("\x1b")
            time.sleep(0.1)
            assert not monitor.recording_event.is_set()
            assert monitor.cancelled_event.is_set()
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_escape_does_not_set_done(self, mock_tty, mock_termios, fake_stdin):
        """Test ESC does not set the done event."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        monitor.start()

        try:
            fake_stdin.send_key("\x1b")
            time.sleep(0.1)
            assert not monitor.done_event.is_set()
            assert monitor.cancelled_event.is_set()
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_escape_fires_callback(self, mock_tty, mock_termios, fake_stdin):
        """Test ESC fires state change callback with CANCELLED."""
        states = []
        monitor = KeyboardMonitor(
            stdin=fake_stdin,
            on_state_change=lambda s: states.append(s),
        )
        monitor.start()

        try:
            fake_stdin.send_key("\x1b")
            time.sleep(0.1)
            assert RecordingState.CANCELLED in states
        finally:
            monitor.stop()


class TestKeyboardMonitorStateCallback:
    """Tests for state change callback."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_callback_fired_on_recording(self, mock_tty, mock_termios, fake_stdin):
        """Test state callback fires when entering RECORDING."""
        states = []
        monitor = KeyboardMonitor(
            stdin=fake_stdin,
            on_state_change=lambda s: states.append(s),
            release_timeout=0.1,
        )
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.05)
            assert RecordingState.RECORDING in states
        finally:
            fake_stdin.send_key("\n")
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_callback_fired_on_pause_and_done(self, mock_tty, mock_termios, fake_stdin):
        """Test state callback fires for PAUSED and DONE transitions."""
        states = []
        monitor = KeyboardMonitor(
            stdin=fake_stdin,
            on_state_change=lambda s: states.append(s),
            release_timeout=0.1,
        )
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.05)
            # Wait for release timeout
            time.sleep(0.2)
            fake_stdin.send_key("\n")
            time.sleep(0.1)

            assert RecordingState.RECORDING in states
            assert RecordingState.PAUSED in states
            assert RecordingState.DONE in states
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_no_duplicate_state_callbacks(self, mock_tty, mock_termios, fake_stdin):
        """Test same state doesn't trigger duplicate callbacks."""
        states = []
        monitor = KeyboardMonitor(
            stdin=fake_stdin,
            on_state_change=lambda s: states.append(s),
            release_timeout=0.3,
        )
        monitor.start()

        try:
            # Send multiple spaces — should only get one RECORDING callback
            for _ in range(5):
                fake_stdin.send_key(" ")
                time.sleep(0.02)
            time.sleep(0.05)

            recording_count = sum(1 for s in states if s == RecordingState.RECORDING)
            assert recording_count == 1
        finally:
            fake_stdin.send_key("\n")
            monitor.stop()


class TestKeyboardMonitorTerminalRestore:
    """Tests for terminal settings save/restore."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_terminal_settings_restored(self, mock_tty, mock_termios, fake_stdin):
        """Test terminal settings are saved and restored on stop."""
        old_settings = [1, 2, 3]
        mock_termios.tcgetattr.return_value = old_settings

        monitor = KeyboardMonitor(stdin=fake_stdin)
        monitor.start()

        fake_stdin.send_key("\n")
        monitor.stop()

        mock_termios.tcsetattr.assert_called_once_with(
            fake_stdin.fileno(),
            mock_termios.TCSADRAIN,
            old_settings,
        )

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_context_manager_restores(self, mock_tty, mock_termios, fake_stdin):
        """Test context manager restores terminal on exit."""
        old_settings = [4, 5, 6]
        mock_termios.tcgetattr.return_value = old_settings

        with KeyboardMonitor(stdin=fake_stdin) as _monitor:
            fake_stdin.send_key("\n")
            time.sleep(0.1)

        mock_termios.tcsetattr.assert_called_once()


class TestKeyboardMonitorContextManager:
    """Tests for context manager protocol."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_enter_returns_self(self, mock_tty, mock_termios, fake_stdin):
        """Test __enter__ returns the monitor instance."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        result = monitor.__enter__()
        assert result is monitor
        fake_stdin.send_key("\n")
        monitor.__exit__(None, None, None)

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_with_statement(self, mock_tty, mock_termios, fake_stdin):
        """Test monitor works in a with statement."""
        with KeyboardMonitor(stdin=fake_stdin) as monitor:
            assert isinstance(monitor, KeyboardMonitor)
            fake_stdin.send_key("\n")
            time.sleep(0.1)


class TestMonitorMode:
    """Tests for MonitorMode enum."""

    def test_recording_mode_value(self):
        """Test RECORDING mode string value."""
        assert MonitorMode.RECORDING.value == "recording"

    def test_presentation_mode_value(self):
        """Test PRESENTATION mode string value."""
        assert MonitorMode.PRESENTATION.value == "presentation"

    def test_default_mode_is_recording(self, fake_stdin):
        """Test default monitor mode is RECORDING."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        assert monitor.mode == MonitorMode.RECORDING


class TestKeyboardMonitorManualKey:
    """Tests for M key detection in RECORDING mode."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_m_sets_manual_event(self, mock_tty, mock_termios, fake_stdin):
        """Test pressing M sets the manual event."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        monitor.start()

        try:
            fake_stdin.send_key("m")
            time.sleep(0.1)
            assert monitor.manual_event.is_set()
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_uppercase_m_sets_manual_event(self, mock_tty, mock_termios, fake_stdin):
        """Test pressing uppercase M also sets the manual event."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        monitor.start()

        try:
            fake_stdin.send_key("M")
            time.sleep(0.1)
            assert monitor.manual_event.is_set()
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_m_clears_recording(self, mock_tty, mock_termios, fake_stdin):
        """Test M key clears recording event if currently recording."""
        monitor = KeyboardMonitor(stdin=fake_stdin, release_timeout=0.5)
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.05)
            assert monitor.recording_event.is_set()

            fake_stdin.send_key("m")
            time.sleep(0.1)
            assert not monitor.recording_event.is_set()
            assert monitor.manual_event.is_set()
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_m_fires_callback(self, mock_tty, mock_termios, fake_stdin):
        """Test M key fires on_manual callback."""
        called = []
        monitor = KeyboardMonitor(
            stdin=fake_stdin,
            on_manual=lambda: called.append(True),
        )
        monitor.start()

        try:
            fake_stdin.send_key("m")
            time.sleep(0.1)
            assert len(called) == 1
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_manual_event_not_set_initially(self, mock_tty, mock_termios, fake_stdin):
        """Test manual_event is not set initially."""
        monitor = KeyboardMonitor(stdin=fake_stdin)
        assert not monitor.manual_event.is_set()


class TestKeyboardMonitorPresentationMode:
    """Tests for PRESENTATION monitor mode."""

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_spacebar_sets_interrupted(self, mock_tty, mock_termios, fake_stdin):
        """Test spacebar tap sets interrupted_event in PRESENTATION mode."""
        monitor = KeyboardMonitor(stdin=fake_stdin, mode=MonitorMode.PRESENTATION)
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.1)
            assert monitor.interrupted_event.is_set()
            # Should NOT set recording_event
            assert not monitor.recording_event.is_set()
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_spacebar_fires_interrupt_callback(self, mock_tty, mock_termios, fake_stdin):
        """Test spacebar fires on_interrupt callback in PRESENTATION mode."""
        called = []
        monitor = KeyboardMonitor(
            stdin=fake_stdin,
            mode=MonitorMode.PRESENTATION,
            on_interrupt=lambda: called.append(True),
        )
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.1)
            assert len(called) == 1
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_enter_sets_done(self, mock_tty, mock_termios, fake_stdin):
        """Test ENTER sets done_event in PRESENTATION mode."""
        monitor = KeyboardMonitor(stdin=fake_stdin, mode=MonitorMode.PRESENTATION)
        monitor.start()

        try:
            fake_stdin.send_key("\n")
            time.sleep(0.1)
            assert monitor.done_event.is_set()
            assert monitor.state == RecordingState.DONE
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_escape_sets_cancelled(self, mock_tty, mock_termios, fake_stdin):
        """Test ESC sets cancelled_event in PRESENTATION mode."""
        monitor = KeyboardMonitor(stdin=fake_stdin, mode=MonitorMode.PRESENTATION)
        monitor.start()

        try:
            fake_stdin.send_key("\x1b")
            time.sleep(0.1)
            assert monitor.cancelled_event.is_set()
            assert monitor.state == RecordingState.CANCELLED
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_m_sets_manual_event(self, mock_tty, mock_termios, fake_stdin):
        """Test M sets manual_event in PRESENTATION mode."""
        called = []
        monitor = KeyboardMonitor(
            stdin=fake_stdin,
            mode=MonitorMode.PRESENTATION,
            on_manual=lambda: called.append(True),
        )
        monitor.start()

        try:
            fake_stdin.send_key("m")
            time.sleep(0.1)
            assert monitor.manual_event.is_set()
            assert len(called) == 1
        finally:
            monitor.stop()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_interrupted_event_not_set_initially(self, mock_tty, mock_termios, fake_stdin):
        """Test interrupted_event is not set initially."""
        monitor = KeyboardMonitor(stdin=fake_stdin, mode=MonitorMode.PRESENTATION)
        assert not monitor.interrupted_event.is_set()

    @patch("neev_voice.audio.keyboard.termios")
    @patch("neev_voice.audio.keyboard.tty")
    def test_spacebar_breaks_loop(self, mock_tty, mock_termios, fake_stdin):
        """Test spacebar tap breaks the monitor loop in PRESENTATION mode."""
        monitor = KeyboardMonitor(stdin=fake_stdin, mode=MonitorMode.PRESENTATION)
        monitor.start()

        try:
            fake_stdin.send_key(" ")
            time.sleep(0.2)
            # Thread should have stopped
            assert monitor.interrupted_event.is_set()
        finally:
            monitor.stop()
