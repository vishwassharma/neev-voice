"""Tests for discuss TUI panel builders and DiscussTUI wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from rich.console import Console
from rich.panel import Panel

from neev_voice.audio.keyboard import RecordingState
from neev_voice.discuss.session import SessionManager
from neev_voice.discuss.state import DiscussState
from neev_voice.discuss.tui import (
    EQUALIZER_FRAMES,
    DiscussTUI,
    get_equalizer_frame,
    make_answer_panel,
    make_answer_text_panel,
    make_enquiry_panel,
    make_playback_panel,
    make_prepare_enquiry_panel,
    make_prepare_panel,
    make_presentation_panel,
    make_recording_animated_panel,
    make_recording_panel,
)


class TestMakePresentationPanel:
    """Tests for make_presentation_panel."""

    def test_returns_panel(self) -> None:
        """Returns a Rich Panel."""
        panel = make_presentation_panel("Basics", index=0, total=3)
        assert isinstance(panel, Panel)

    def test_waiting_shows_enter_key(self) -> None:
        """Waiting panel shows ENTER to start."""
        panel = make_presentation_panel("Basics", index=0, total=3, playing=False)
        text = panel.renderable.plain
        assert "ENTER" in text
        assert "start" in text

    def test_playing_shows_playing_label(self) -> None:
        """Playing panel shows Playing label."""
        panel = make_presentation_panel("Basics", index=0, total=3, playing=True)
        text = panel.renderable.plain
        assert "Playing" in text

    def test_title_shows_concept_counter(self) -> None:
        """Panel title shows concept index."""
        panel = make_presentation_panel("Test", index=2, total=5)
        assert "3/5" in panel.title

    def test_description_shown(self) -> None:
        """Description appears in waiting panel."""
        panel = make_presentation_panel("T", description="Some desc", index=0, total=1)
        text = panel.renderable.plain
        assert "Some desc" in text

    def test_playing_border_green(self) -> None:
        """Playing panel has green border."""
        panel = make_presentation_panel("T", playing=True)
        assert panel.border_style == "green"

    def test_waiting_border_cyan(self) -> None:
        """Waiting panel has cyan border."""
        panel = make_presentation_panel("T", playing=False)
        assert panel.border_style == "cyan"


class TestMakeEnquiryPanel:
    """Tests for make_enquiry_panel."""

    def test_returns_panel(self) -> None:
        """Returns a Rich Panel."""
        panel = make_enquiry_panel()
        assert isinstance(panel, Panel)

    def test_shows_key_instructions(self) -> None:
        """Panel shows SPACE, M, and ESC keys."""
        panel = make_enquiry_panel()
        text = panel.renderable.plain
        assert "SPACE" in text
        assert "M" in text
        assert "ESC" in text

    def test_title_is_enquiry(self) -> None:
        """Panel title is Enquiry."""
        panel = make_enquiry_panel()
        assert panel.title == "Enquiry"

    def test_border_yellow(self) -> None:
        """Panel has yellow border."""
        panel = make_enquiry_panel()
        assert panel.border_style == "yellow"


class TestMakeRecordingPanel:
    """Tests for make_recording_panel."""

    def test_idle_state(self) -> None:
        """Idle state shows waiting message."""
        panel = make_recording_panel(RecordingState.IDLE)
        assert "Waiting" in panel.renderable.plain

    def test_recording_state(self) -> None:
        """Recording state shows RECORDING."""
        panel = make_recording_panel(RecordingState.RECORDING)
        assert "RECORDING" in panel.renderable.plain

    def test_paused_state(self) -> None:
        """Paused state shows PAUSED."""
        panel = make_recording_panel(RecordingState.PAUSED)
        assert "PAUSED" in panel.renderable.plain

    def test_done_state(self) -> None:
        """Done state shows Captured."""
        panel = make_recording_panel(RecordingState.DONE)
        assert "Captured" in panel.renderable.plain

    def test_cancelled_state(self) -> None:
        """Cancelled state shows Cancelled."""
        panel = make_recording_panel(RecordingState.CANCELLED)
        assert "Cancelled" in panel.renderable.plain

    def test_title_is_voice_recording(self) -> None:
        """Panel title is Voice Recording."""
        panel = make_recording_panel(RecordingState.IDLE)
        assert panel.title == "Voice Recording"


class TestMakeAnswerPanel:
    """Tests for make_answer_panel."""

    def test_returns_panel(self) -> None:
        """Returns a Rich Panel."""
        panel = make_answer_panel()
        assert isinstance(panel, Panel)

    def test_waiting_shows_answer_ready(self) -> None:
        """Waiting panel shows Answer ready."""
        panel = make_answer_panel(playing=False)
        assert "Answer ready" in panel.renderable.plain

    def test_playing_shows_speaking(self) -> None:
        """Playing panel shows Speaking."""
        panel = make_answer_panel(playing=True)
        assert "Speaking" in panel.renderable.plain

    def test_title_is_answer(self) -> None:
        """Panel title is Answer."""
        panel = make_answer_panel()
        assert panel.title == "Answer"


class TestMakePreparePanel:
    """Tests for make_prepare_panel."""

    def test_returns_panel(self) -> None:
        """Returns a Rich Panel."""
        panel = make_prepare_panel()
        assert isinstance(panel, Panel)

    def test_shows_analyzing(self) -> None:
        """Panel shows analyzing message."""
        panel = make_prepare_panel()
        assert "Analyzing" in panel.renderable.plain

    def test_title_is_preparing(self) -> None:
        """Panel title is Preparing."""
        panel = make_prepare_panel()
        assert panel.title == "Preparing"


class TestMakePrepareEnquiryPanel:
    """Tests for make_prepare_enquiry_panel."""

    def test_returns_panel(self) -> None:
        """Returns a Rich Panel."""
        panel = make_prepare_enquiry_panel()
        assert isinstance(panel, Panel)

    def test_shows_researching(self) -> None:
        """Panel shows researching message."""
        panel = make_prepare_enquiry_panel()
        assert "Researching" in panel.renderable.plain


class TestMakeAnswerTextPanel:
    """Tests for make_answer_text_panel."""

    def test_returns_panel(self) -> None:
        """Returns a Rich Panel."""
        panel = make_answer_text_panel("Some answer")
        assert isinstance(panel, Panel)

    def test_shows_answer_text(self) -> None:
        """Panel displays the answer content."""
        panel = make_answer_text_panel("The answer is 42.")
        assert "The answer is 42." in panel.renderable.plain

    def test_shows_key_instructions(self) -> None:
        """Panel shows SPACE and ESC keys."""
        panel = make_answer_text_panel("answer")
        text = panel.renderable.plain
        assert "SPACE" in text
        assert "ESC" in text

    def test_truncates_long_answer(self) -> None:
        """Long answers are truncated."""
        long_text = "x" * 3000
        panel = make_answer_text_panel(long_text)
        text = panel.renderable.plain
        assert "..." in text

    def test_title_is_answer(self) -> None:
        """Panel title is Answer."""
        panel = make_answer_text_panel("text")
        assert panel.title == "Answer"


class TestEqualizer:
    """Tests for equalizer animation."""

    def test_frames_not_empty(self) -> None:
        """Frames list is not empty."""
        assert len(EQUALIZER_FRAMES) > 0

    def test_get_frame_cycles(self) -> None:
        """get_equalizer_frame cycles through frames."""
        f0 = get_equalizer_frame(0)
        f1 = get_equalizer_frame(1)
        assert f0 != f1
        # Wraps around
        assert get_equalizer_frame(len(EQUALIZER_FRAMES)) == f0


class TestMakePlaybackPanel:
    """Tests for make_playback_panel."""

    def test_returns_panel(self) -> None:
        """Returns a Rich Panel."""
        panel = make_playback_panel(title="Test", speed=1.0, tick=0)
        assert isinstance(panel, Panel)

    def test_shows_equalizer(self) -> None:
        """Panel shows equalizer characters."""
        panel = make_playback_panel(title="T", tick=0)
        text = panel.renderable.plain
        assert "▁" in text or "▃" in text or "▅" in text or "▇" in text

    def test_shows_speed(self) -> None:
        """Panel shows current speed."""
        panel = make_playback_panel(title="T", speed=1.5, tick=0)
        assert "1.5x" in panel.renderable.plain

    def test_shows_speed_keys(self) -> None:
        """Panel shows 1-4 speed key instructions."""
        panel = make_playback_panel(title="T", tick=0)
        text = panel.renderable.plain
        assert "1x" in text
        assert "1.25x" in text
        assert "2x" in text

    def test_concept_title(self) -> None:
        """Concept panel shows Playing x/y title."""
        panel = make_playback_panel(title="T", index=1, total=5, tick=0)
        assert "2/5" in panel.title

    def test_answer_title(self) -> None:
        """Answer panel title says Answer."""
        panel = make_playback_panel(title="T", is_answer=True, tick=0)
        assert panel.title == "Answer"

    def test_shows_action_keys(self) -> None:
        """Panel shows SPACE, ENTER, ESC keys."""
        panel = make_playback_panel(title="T", tick=0)
        text = panel.renderable.plain
        assert "SPACE" in text
        assert "ENTER" in text
        assert "ESC" in text


class TestMakeRecordingAnimatedPanel:
    """Tests for make_recording_animated_panel."""

    def test_recording_shows_equalizer(self) -> None:
        """Recording state shows equalizer animation."""
        panel = make_recording_animated_panel(RecordingState.RECORDING, tick=0)
        text = panel.renderable.plain
        assert "RECORDING" in text
        assert "▁" in text or "▃" in text or "▅" in text or "▇" in text

    def test_recording_red_border(self) -> None:
        """Recording state has red border."""
        panel = make_recording_animated_panel(RecordingState.RECORDING)
        assert panel.border_style == "red"

    def test_paused_no_equalizer(self) -> None:
        """Paused state shows PAUSED without equalizer."""
        panel = make_recording_animated_panel(RecordingState.PAUSED)
        assert "PAUSED" in panel.renderable.plain

    def test_idle_shows_instructions(self) -> None:
        """Idle state shows SPACEBAR instruction."""
        panel = make_recording_animated_panel(RecordingState.IDLE)
        assert "SPACEBAR" in panel.renderable.plain

    def test_done_shows_captured(self) -> None:
        """Done state shows captured."""
        panel = make_recording_animated_panel(RecordingState.DONE)
        assert "Captured" in panel.renderable.plain


class TestDiscussTUI:
    """Tests for DiscussTUI wrapper class."""

    def _make_runner(self, tmp_path: Path) -> MagicMock:
        """Create a mock runner with a real session."""
        mgr = SessionManager(base_dir=tmp_path / "discuss")
        session = mgr.create_session("test-tui", research_path="/r", source_path="/s")
        runner = MagicMock()
        runner.session = session
        runner.run = AsyncMock()
        runner.on_state_enter = None
        return runner

    def test_init_sets_callback(self, tmp_path: Path) -> None:
        """DiscussTUI sets on_state_enter on the runner."""
        runner = self._make_runner(tmp_path)
        DiscussTUI(runner)
        assert runner.on_state_enter is not None
        assert callable(runner.on_state_enter)

    def test_init_accepts_console(self, tmp_path: Path) -> None:
        """DiscussTUI accepts a custom console."""
        runner = self._make_runner(tmp_path)
        console = Console()
        tui = DiscussTUI(runner, console=console)
        assert tui.console is console

    async def test_run_calls_runner(self, tmp_path: Path) -> None:
        """run() delegates to runner.run()."""
        runner = self._make_runner(tmp_path)
        tui = DiscussTUI(runner, console=Console(file=open("/dev/null", "w")))  # noqa: SIM115
        await tui.run()
        runner.run.assert_called_once()

    def test_on_state_enter_prepare_starts_spinner(self, tmp_path: Path) -> None:
        """Prepare state starts a spinner."""
        runner = self._make_runner(tmp_path)
        console = Console(file=open("/dev/null", "w"))  # noqa: SIM115
        tui = DiscussTUI(runner, console=console)
        tui._on_state_enter(DiscussState.PREPARE, {})
        assert tui._active_status is not None
        tui._stop_spinner()

    def test_on_state_enter_enquiry(self, tmp_path: Path) -> None:
        """Enquiry state renders enquiry panel."""
        runner = self._make_runner(tmp_path)
        console = Console(file=open("/dev/null", "w"))  # noqa: SIM115
        tui = DiscussTUI(runner, console=console)
        tui._on_state_enter(DiscussState.ENQUIRY, {})

    def test_on_state_enter_prepare_enquiry_starts_spinner(self, tmp_path: Path) -> None:
        """Prepare-enquiry state starts a spinner."""
        runner = self._make_runner(tmp_path)
        console = Console(file=open("/dev/null", "w"))  # noqa: SIM115
        tui = DiscussTUI(runner, console=console)
        tui._on_state_enter(DiscussState.PREPARE_ENQUIRY, {})
        assert tui._active_status is not None
        tui._stop_spinner()

    def test_on_state_enter_presentation_no_crash(self, tmp_path: Path) -> None:
        """Presentation state does not crash (panel rendered by engine)."""
        runner = self._make_runner(tmp_path)
        console = Console(file=open("/dev/null", "w"))  # noqa: SIM115
        tui = DiscussTUI(runner, console=console)
        tui._on_state_enter(DiscussState.PRESENTATION, {})

    def test_on_state_enter_presentation_enquiry_shows_answer(self, tmp_path: Path) -> None:
        """Presentation-enquiry displays answer text from context."""
        runner = self._make_runner(tmp_path)
        output = open("/dev/null", "w")  # noqa: SIM115
        console = Console(file=output)
        tui = DiscussTUI(runner, console=console)
        # Should not raise; answer is displayed via console.print
        tui._on_state_enter(DiscussState.PRESENTATION_ENQUIRY, {"answer": "Test answer"})

    def test_on_state_enter_presentation_enquiry_no_answer(self, tmp_path: Path) -> None:
        """Presentation-enquiry with no answer does not crash."""
        runner = self._make_runner(tmp_path)
        console = Console(file=open("/dev/null", "w"))  # noqa: SIM115
        tui = DiscussTUI(runner, console=console)
        tui._on_state_enter(DiscussState.PRESENTATION_ENQUIRY, {})
