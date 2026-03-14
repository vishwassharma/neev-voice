"""Tests for discuss TUI panel builders."""

from __future__ import annotations

from rich.panel import Panel

from neev_voice.audio.keyboard import RecordingState
from neev_voice.discuss.tui import (
    make_answer_panel,
    make_enquiry_panel,
    make_presentation_panel,
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
