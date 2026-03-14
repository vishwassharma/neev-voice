"""Push-to-talk keyboard monitor using raw terminal mode.

Detects spacebar hold/release and Enter key press without external
dependencies. Uses tty.setcbreak() to read stdin character-by-character
and select.select() for non-blocking polling. Spacebar release is
detected via a configurable timeout (no space chars received).

Supports two monitor modes:
- RECORDING: Push-to-talk with spacebar hold/release detection.
- PRESENTATION: Spacebar tap for interrupt, ENTER for next, M for manual input.
"""

from __future__ import annotations

import asyncio
import contextlib
import select
import sys
import termios
import threading
import tty
from collections.abc import Callable
from enum import StrEnum
from typing import IO

__all__ = ["KeyboardMonitor", "MonitorMode", "RecordingState"]


class MonitorMode(StrEnum):
    """Operating mode for the keyboard monitor.

    Attributes:
        RECORDING: Push-to-talk mode with spacebar hold/release detection.
        PRESENTATION: Tap-based mode where spacebar interrupts playback,
            ENTER advances, and M triggers manual text input.
    """

    RECORDING = "recording"
    PRESENTATION = "presentation"


class RecordingState(StrEnum):
    """States for push-to-talk recording.

    Attributes:
        IDLE: No recording activity, waiting for user input.
        RECORDING: Spacebar held, actively capturing audio.
        PAUSED: Spacebar released, recording paused.
        DONE: Enter pressed, recording finalized.
        CANCELLED: ESC pressed, recording cancelled.
    """

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    DONE = "done"
    CANCELLED = "cancelled"


class KeyboardMonitor:
    """Monitors keyboard for push-to-talk recording control.

    Runs a background daemon thread that reads stdin in cbreak mode
    to detect spacebar hold/release and Enter key press. Thread-safe
    communication with async code via threading.Event and
    loop.call_soon_threadsafe().

    In RECORDING mode (default), spacebar hold/release is detected via
    timeout-based release detection. In PRESENTATION mode, spacebar tap
    sets an interrupt event, and 'M'/'m' sets a manual input event.

    Attributes:
        release_timeout: Seconds of no-space-char to detect key release.
        recording_event: Set when spacebar is held (recording active).
        done_event: Set when Enter is pressed (recording finalized).
        cancelled_event: Set when ESC is pressed (recording cancelled).
        manual_event: Set when M/m is pressed (manual input requested).
        interrupted_event: Set when spacebar tapped in PRESENTATION mode.
        mode: Current monitor mode (RECORDING or PRESENTATION).
        state: Current recording state.
    """

    def __init__(
        self,
        release_timeout: float = 0.15,
        on_state_change: Callable[[RecordingState], None] | None = None,
        stdin: IO[str] | None = None,
        mode: MonitorMode = MonitorMode.RECORDING,
        on_manual: Callable[[], None] | None = None,
        on_interrupt: Callable[[], None] | None = None,
    ) -> None:
        """Initialize KeyboardMonitor.

        Args:
            release_timeout: Seconds without space char to detect release.
            on_state_change: Optional callback fired on state transitions.
            stdin: Optional file object to read from (defaults to sys.stdin).
                   Useful for testing with os.pipe().
            mode: Monitor mode (RECORDING for push-to-talk, PRESENTATION
                  for tap-based control).
            on_manual: Optional callback fired when M key is pressed.
            on_interrupt: Optional callback fired when spacebar tapped
                          in PRESENTATION mode.
        """
        self.release_timeout = release_timeout
        self._on_state_change = on_state_change
        self._on_manual = on_manual
        self._on_interrupt = on_interrupt
        self._stdin = stdin or sys.stdin
        self.mode = mode
        self.recording_event = threading.Event()
        self.done_event = threading.Event()
        self.cancelled_event = threading.Event()
        self.manual_event = threading.Event()
        self.interrupted_event = threading.Event()
        self.state = RecordingState.IDLE
        self._thread: threading.Thread | None = None
        self._old_settings: list | None = None
        self._stop_flag = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _set_state(self, new_state: RecordingState) -> None:
        """Update state and fire callback if provided.

        Args:
            new_state: The new recording state to transition to.
        """
        if self.state == new_state:
            return
        self.state = new_state
        if self._on_state_change:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._on_state_change, new_state)
            else:
                self._on_state_change(new_state)

    def _fire_callback(self, callback: Callable[[], None] | None) -> None:
        """Fire a callback, respecting async loop thread-safety.

        Args:
            callback: The callback to fire, or None to skip.
        """
        if callback is None:
            return
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(callback)
        else:
            callback()

    def _monitor_loop(self) -> None:
        """Background thread main loop reading stdin in cbreak mode.

        Polls stdin with select.select() at 50ms intervals. Behavior
        depends on the monitor mode:

        RECORDING mode: Detects spacebar hold (space chars arriving),
        release (timeout with no space chars), Enter, ESC, and M key.

        PRESENTATION mode: Spacebar tap sets interrupted_event (no
        hold/release tracking), Enter sets done_event, ESC sets
        cancelled_event, and M sets manual_event.
        """
        poll_interval = 0.05
        last_space_time: float | None = None

        import time

        while not self._stop_flag.is_set():
            try:
                ready, _, _ = select.select([self._stdin], [], [], poll_interval)
            except (ValueError, OSError, TypeError):
                break

            if ready:
                try:
                    ch = self._stdin.read(1)
                except (OSError, ValueError):
                    break

                if self.mode == MonitorMode.PRESENTATION:
                    self._handle_presentation_key(ch)
                    if self.done_event.is_set() or self.cancelled_event.is_set():
                        break
                    if self.interrupted_event.is_set() or self.manual_event.is_set():
                        break
                else:
                    if ch == " ":
                        last_space_time = time.monotonic()
                        if not self.recording_event.is_set():
                            self.recording_event.set()
                            self._set_state(RecordingState.RECORDING)
                    elif ch in ("\n", "\r"):
                        self.recording_event.clear()
                        self.done_event.set()
                        self._set_state(RecordingState.DONE)
                        break
                    elif ch == "\x1b":
                        self.recording_event.clear()
                        self.cancelled_event.set()
                        self._set_state(RecordingState.CANCELLED)
                        break
                    elif ch in ("m", "M"):
                        self.recording_event.clear()
                        self.manual_event.set()
                        self._fire_callback(self._on_manual)
                        break
            else:
                if (
                    self.mode == MonitorMode.RECORDING
                    and last_space_time is not None
                    and self.recording_event.is_set()
                ):
                    elapsed = time.monotonic() - last_space_time
                    if elapsed >= self.release_timeout:
                        self.recording_event.clear()
                        self._set_state(RecordingState.PAUSED)
                        last_space_time = None

    def _handle_presentation_key(self, ch: str) -> None:
        """Handle a keypress in PRESENTATION mode.

        In presentation mode, spacebar is a single tap (interrupt),
        not a hold/release. No hold detection is performed.

        Args:
            ch: The character read from stdin.
        """
        if ch == " ":
            self.interrupted_event.set()
            self._fire_callback(self._on_interrupt)
        elif ch in ("\n", "\r"):
            self.done_event.set()
            self._set_state(RecordingState.DONE)
        elif ch == "\x1b":
            self.cancelled_event.set()
            self._set_state(RecordingState.CANCELLED)
        elif ch in ("m", "M"):
            self.manual_event.set()
            self._fire_callback(self._on_manual)

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the keyboard monitor background thread.

        Sets stdin to cbreak mode and launches a daemon thread to
        read keystrokes. Terminal settings are saved for restoration
        in stop().

        Args:
            loop: Optional event loop for thread-safe async callbacks.

        Raises:
            RuntimeError: If stdin is not a terminal (not a TTY).
        """
        self._loop = loop

        if hasattr(self._stdin, "isatty") and not self._stdin.isatty():
            raise RuntimeError("Push-to-talk requires a terminal (stdin is not a TTY)")

        if hasattr(self._stdin, "fileno"):
            try:
                self._old_settings = termios.tcgetattr(self._stdin.fileno())
                tty.setcbreak(self._stdin.fileno())
            except (termios.error, OSError):
                self._old_settings = None

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the keyboard monitor and restore terminal settings.

        Signals the background thread to stop, waits for it to finish,
        and restores the original terminal settings.
        """
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._old_settings is not None and hasattr(self._stdin, "fileno"):
            with contextlib.suppress(termios.error, OSError):
                termios.tcsetattr(
                    self._stdin.fileno(),
                    termios.TCSADRAIN,
                    self._old_settings,
                )
            self._old_settings = None

    def __enter__(self) -> KeyboardMonitor:
        """Context manager entry — starts monitoring.

        Returns:
            The KeyboardMonitor instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit — stops monitoring and restores terminal."""
        self.stop()
