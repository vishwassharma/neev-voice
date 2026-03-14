"""State machine runner for the discuss subcommand.

Orchestrates the discuss state machine by delegating to the
appropriate engine for each state and managing transitions,
state stack push/pop, and session persistence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog

from neev_voice.discuss.enquiry import EnquiryEngine, EnquiryResult
from neev_voice.discuss.history import SessionHistory
from neev_voice.discuss.prepare import PrepareEngine
from neev_voice.discuss.prepare_enquiry import PrepareEnquiryEngine
from neev_voice.discuss.presentation import PresentationEngine
from neev_voice.discuss.session import SessionInfo, SessionManager
from neev_voice.discuss.state import (
    DiscussState,
    StateSnapshot,
    validate_transition,
)

if TYPE_CHECKING:
    from neev_voice.config import NeevSettings
    from neev_voice.stt.base import STTProvider
    from neev_voice.tts.base import TTSProvider

logger = structlog.get_logger(__name__)

__all__ = ["DiscussRunner"]


class DiscussRunner:
    """Main state machine orchestrator for discuss sessions.

    Runs the discuss state machine loop, delegating to the appropriate
    engine for each state and managing transitions. Persists session
    state after every transition for crash recovery.

    Attributes:
        session: Current discuss session info.
        settings: Application settings.
        session_manager: Session persistence manager.
        tts_provider: TTS provider for presentation audio.
        stt_provider: STT provider for voice enquiry.
    """

    def __init__(
        self,
        session: SessionInfo,
        settings: NeevSettings,
        session_manager: SessionManager,
        tts_provider: TTSProvider | None = None,
        stt_provider: STTProvider | None = None,
        on_state_enter: Callable[[DiscussState, dict], None] | None = None,
    ) -> None:
        """Initialize the state machine runner.

        Args:
            session: Current discuss session info.
            settings: Application settings.
            session_manager: Session persistence manager.
            tts_provider: Optional TTS provider for audio synthesis.
            stt_provider: Optional STT provider for voice transcription.
            on_state_enter: Optional callback fired at the start of each
                state machine iteration with (state, context_dict).
        """
        self.session = session
        self.settings = settings
        self.session_manager = session_manager
        self.tts_provider = tts_provider
        self.stt_provider = stt_provider
        self.on_state_enter = on_state_enter

        # Transient state (not persisted, used between transitions)
        self._current_enquiry: EnquiryResult | None = None
        self._current_answer: str | None = None
        self._restored_state_data: dict | None = None
        self.history = SessionHistory(session_manager.session_dir(session.name))

    async def run(self) -> None:
        """Run the discuss state machine loop.

        Loops through states, delegating to the appropriate handler
        for each state. The loop exits when all concepts are presented,
        the user cancels, or an unrecoverable error occurs.
        """
        logger.info(
            "discuss_runner_started",
            session=self.session.name,
            initial_state=self.session.state,
        )

        while True:
            state = self.session.state
            logger.info("discuss_runner_state", state=state)

            if self.on_state_enter:
                ctx: dict = {}
                if state == DiscussState.PRESENTATION_ENQUIRY and self._current_answer:
                    ctx["answer"] = self._current_answer
                if state == DiscussState.PREPARE_ENQUIRY and self._current_enquiry:
                    ctx["query"] = self._current_enquiry.query or ""
                self.on_state_enter(state, ctx)

            match state:
                case DiscussState.PREPARE:
                    should_continue = await self._handle_prepare()
                case DiscussState.PRESENTATION:
                    should_continue = await self._handle_presentation()
                case DiscussState.ENQUIRY:
                    should_continue = await self._handle_enquiry()
                case DiscussState.PREPARE_ENQUIRY:
                    should_continue = await self._handle_prepare_enquiry()
                case DiscussState.PRESENTATION_ENQUIRY:
                    should_continue = await self._handle_presentation_enquiry()
                case _:
                    logger.error("discuss_runner_unknown_state", state=state)
                    break

            if not should_continue:
                break

        logger.info("discuss_runner_finished", session=self.session.name)

    async def _handle_prepare(self) -> bool:
        """Handle the PREPARE state.

        Runs the prepare engine to analyze documents and extract
        concepts. Transitions to PRESENTATION on completion.

        Returns:
            True to continue the state machine loop.
        """
        prepare_dir = self.session_manager.session_dir(self.session.name) / "prepare"

        engine = PrepareEngine(
            session=self.session,
            settings=self.settings,
            prepare_dir=prepare_dir,
        )
        concepts = await engine.run()

        if not concepts:
            logger.warning("prepare_no_concepts")
            return False

        self.session.concepts = [c.to_dict() for c in concepts]
        self.session.prepare_complete = True
        self._transition(DiscussState.PRESENTATION)
        return True

    async def _handle_presentation(self) -> bool:
        """Handle the PRESENTATION state.

        Presents concepts via TTS. On interrupt (SPACEBAR), pushes
        state to stack and transitions to ENQUIRY. On completion,
        exits the loop. Tracks progress via on_concept_done callback
        for resume after interruption.

        Returns:
            True to continue, False to exit.
        """
        prepare_dir = self.session_manager.session_dir(self.session.name) / "prepare"

        # Determine start index: prefer restored state, then persisted index
        start_index = self.session.presentation_index
        if self._restored_state_data:
            start_index = self._restored_state_data.get("current_concept_index", start_index)
            self._restored_state_data = None

        engine = PresentationEngine(
            session=self.session,
            settings=self.settings,
            tts_provider=self.tts_provider,
            prepare_dir=prepare_dir,
            on_concept_done=self._on_concept_done,
        )

        result = await engine.run(start_index=start_index)

        if result.interrupted:
            # Push current state and transition to enquiry
            self.session.state_stack.push(
                StateSnapshot(
                    state=DiscussState.PRESENTATION,
                    data=result.state_data,
                )
            )
            self._transition(DiscussState.ENQUIRY)
            return True

        if result.cancelled:
            return False

        if result.completed:
            logger.info("presentation_complete")
            self.session.presentation_index = 0
            self.session_manager.save_session(self.session)
            return False

        return True

    def _on_concept_done(self, concept_index: int) -> None:
        """Callback for presentation engine after each concept completes.

        Persists the next concept index to session for resume support.

        Args:
            concept_index: Index of the concept that just completed.
        """
        self.session.presentation_index = concept_index + 1
        self.session_manager.save_session(self.session)

    async def _handle_enquiry(self) -> bool:
        """Handle the ENQUIRY state.

        Captures user enquiry via voice or text. On ESC, pops state
        stack and returns to the previous state. On query capture,
        transitions to PREPARE_ENQUIRY.

        Returns:
            True to continue, False to exit.
        """
        session_dir = self.session_manager.session_dir(self.session.name)

        engine = EnquiryEngine(
            session=self.session,
            settings=self.settings,
            stt_provider=self.stt_provider,
            session_dir=session_dir,
        )

        result = await engine.run()

        if result.escaped:
            # Pop state stack and restore
            snapshot = self.session.state_stack.pop()
            if snapshot:
                self._restore_state(snapshot)
                return True
            # No state to restore
            if self.session.concepts:
                self._transition(DiscussState.PRESENTATION)
                return True
            # Enquiry-only mode (no concepts) — exit
            return False

        if result.query:
            self._current_enquiry = result
            self.history.append("question", result.query)
            self._transition(DiscussState.PREPARE_ENQUIRY)
            return True

        # No query (e.g., editor cancelled) — stay in enquiry
        return True

    async def _handle_prepare_enquiry(self) -> bool:
        """Handle the PREPARE_ENQUIRY state.

        Researches the answer to the user's enquiry. Checks if
        coming from a nested presentation-enquiry path to update
        the previous answer.

        Returns:
            True to continue.
        """
        if not self._current_enquiry or not self._current_enquiry.query:
            logger.warning("prepare_enquiry_no_query")
            self._transition(DiscussState.ENQUIRY)
            return True

        session_dir = self.session_manager.session_dir(self.session.name)

        # Check if this is a follow-up from presentation-enquiry
        previous_state = self.session.state_stack.peek()
        from_pres_enquiry = (
            previous_state is not None and previous_state.state == DiscussState.PRESENTATION_ENQUIRY
        )
        previous_answer = (
            previous_state.data.get("answer") if from_pres_enquiry and previous_state else None
        )

        engine = PrepareEnquiryEngine(
            session=self.session,
            settings=self.settings,
            session_dir=session_dir,
        )

        self._current_answer = await engine.run(
            query=self._current_enquiry.query,
            from_presentation_enquiry=from_pres_enquiry,
            previous_answer=previous_answer,
        )

        if self._current_answer:
            self.history.append("answer", self._current_answer)

        self._transition(DiscussState.PRESENTATION_ENQUIRY)
        return True

    async def _handle_presentation_enquiry(self) -> bool:
        """Handle the PRESENTATION_ENQUIRY state.

        Presents the answer to the user's enquiry via TTS.
        On ENTER (completion), pops stack and returns to presentation.
        On SPACEBAR (interrupt), pushes state and transitions to enquiry.

        Returns:
            True to continue, False to exit.
        """
        if not self._current_answer:
            logger.warning("presentation_enquiry_no_answer")
            # Pop back to previous state
            snapshot = self.session.state_stack.pop()
            if snapshot:
                self._restore_state(snapshot)
            elif self.session.concepts:
                self._transition(DiscussState.PRESENTATION)
            else:
                self._transition(DiscussState.ENQUIRY)
            return True

        prepare_dir = self.session_manager.session_dir(self.session.name) / "prepare"

        engine = PresentationEngine(
            session=self.session,
            settings=self.settings,
            tts_provider=self.tts_provider,
            prepare_dir=prepare_dir,
        )

        result = await engine.run_answer(self._current_answer)

        if result.interrupted:
            # Nested enquiry — push and transition
            self.session.state_stack.push(
                StateSnapshot(
                    state=DiscussState.PRESENTATION_ENQUIRY,
                    data={
                        "answer": self._current_answer,
                        **result.state_data,
                    },
                )
            )
            self._transition(DiscussState.ENQUIRY)
            return True

        if result.cancelled:
            return False

        # Answer presented (ENTER or completed) — pop stack, return to previous state
        snapshot = self.session.state_stack.pop()
        if snapshot and snapshot.state == DiscussState.PRESENTATION:
            self._restore_state(snapshot)
            return True
        if self.session.concepts:
            self._transition(DiscussState.PRESENTATION)
            return True
        # Enquiry-only mode — wait for user choice, loop on replay
        while True:
            user_choice = await self._wait_after_answer()
            if user_choice == "enquiry":
                self._transition(DiscussState.ENQUIRY)
                return True
            if user_choice == "replay":
                # Re-play the same answer (audio is cached)
                result = await engine.run_answer(self._current_answer)
                if result.interrupted:
                    self.session.state_stack.push(
                        StateSnapshot(
                            state=DiscussState.PRESENTATION_ENQUIRY,
                            data={"answer": self._current_answer, **result.state_data},
                        )
                    )
                    self._transition(DiscussState.ENQUIRY)
                    return True
                if result.cancelled:
                    return False
                continue  # Back to wait
            # "exit"
            logger.info("enquiry_only_complete")
            return False

    async def _wait_after_answer(self) -> str:
        """Wait for user to choose next action after answer playback.

        Shows a prompt and monitors keyboard:
        - SPACE → ``"enquiry"`` (ask follow-up)
        - ENTER → ``"exit"``
        - ESC → ``"replay"`` (go back to answer, replay audio)

        Returns:
            ``"enquiry"``, ``"replay"``, or ``"exit"``.
        """
        import asyncio

        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        from neev_voice.audio.keyboard import KeyboardMonitor, MonitorMode

        console = Console()
        lines = Text()
        lines.append("  What would you like to do?\n\n", style="bold")
        lines.append("  SPACE ", style="bold yellow")
        lines.append("ask follow-up  ", style="dim")
        lines.append("ESC ", style="bold magenta")
        lines.append("replay answer  ", style="dim")
        lines.append("ENTER ", style="bold green")
        lines.append("done", style="dim")
        console.print(Panel(lines, title="Next", border_style="cyan"))

        monitor = KeyboardMonitor(mode=MonitorMode.PRESENTATION)
        monitor.start()

        try:
            while True:
                if monitor.interrupted_event.is_set():
                    return "enquiry"
                if monitor.cancelled_event.is_set():
                    return "replay"
                if monitor.done_event.is_set():
                    return "exit"
                await asyncio.sleep(0.05)
        finally:
            monitor.stop()

    def _transition(self, new_state: DiscussState) -> None:
        """Transition to a new state with validation and persistence.

        Args:
            new_state: The state to transition to.

        Raises:
            ValueError: If the transition is not valid.
        """
        old_state = self.session.state
        if not validate_transition(old_state, new_state):
            logger.error(
                "invalid_transition",
                from_state=old_state,
                to_state=new_state,
            )
            raise ValueError(f"Invalid state transition: {old_state} → {new_state}")

        logger.info(
            "discuss_state_transition",
            from_state=old_state,
            to_state=new_state,
        )
        self.session.state = new_state
        self.session_manager.save_session(self.session)

    def _restore_state(self, snapshot: StateSnapshot) -> None:
        """Restore state from a stack snapshot.

        Sets the session state to the snapshot's state and stores
        the state data for use by the handler.

        Args:
            snapshot: The state snapshot to restore from.
        """
        logger.info(
            "discuss_state_restored",
            restored_state=snapshot.state,
        )
        self.session.state = snapshot.state
        self._restored_state_data = snapshot.data
        self.session_manager.save_session(self.session)
