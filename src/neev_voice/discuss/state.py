"""State machine definitions for the discuss subcommand.

Provides the DiscussState enum, StateSnapshot for capturing state
at transition points, and StateStack for managing nested state
transitions with push/pop semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "VALID_TRANSITIONS",
    "DiscussState",
    "StateSnapshot",
    "StateStack",
    "validate_transition",
]


class DiscussState(StrEnum):
    """States for the discuss state machine.

    Attributes:
        PREPARE: Initial research phase using Claude CLI.
        PRESENTATION: TTS presentation of concepts to user.
        ENQUIRY: User asks a question via voice or text.
        PREPARE_ENQUIRY: Research answer to user's enquiry.
        PRESENTATION_ENQUIRY: Present the answer to user's enquiry.
    """

    PREPARE = "prepare"
    PRESENTATION = "presentation"
    ENQUIRY = "enquiry"
    PREPARE_ENQUIRY = "prepare-enquiry"
    PRESENTATION_ENQUIRY = "presentation-enquiry"


VALID_TRANSITIONS: dict[DiscussState, set[DiscussState]] = {
    DiscussState.PREPARE: {DiscussState.PRESENTATION},
    DiscussState.PRESENTATION: {DiscussState.ENQUIRY},
    DiscussState.ENQUIRY: {
        DiscussState.PREPARE_ENQUIRY,
        DiscussState.PRESENTATION,
        DiscussState.PRESENTATION_ENQUIRY,
    },
    DiscussState.PREPARE_ENQUIRY: {DiscussState.PRESENTATION_ENQUIRY},
    DiscussState.PRESENTATION_ENQUIRY: {
        DiscussState.PRESENTATION,
        DiscussState.ENQUIRY,
    },
}
"""Valid state transitions for the discuss state machine.

Maps each state to the set of states it can transition to.
The enquiry state can transition back to presentation or
presentation-enquiry when restoring from the state stack.
"""


def validate_transition(from_state: DiscussState, to_state: DiscussState) -> bool:
    """Check whether a state transition is valid.

    Args:
        from_state: The current state.
        to_state: The proposed next state.

    Returns:
        True if the transition is allowed, False otherwise.
    """
    allowed = VALID_TRANSITIONS.get(from_state, set())
    return to_state in allowed


@dataclass
class StateSnapshot:
    """Captures state information at a transition point.

    Used by StateStack to save and restore state context when
    transitioning between presentation and enquiry states.

    Attributes:
        state: The state being saved.
        data: State-specific data (must be JSON-serializable).
        timestamp: ISO 8601 timestamp of when the snapshot was taken.
    """

    state: DiscussState
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSnapshot:
        """Deserialize a snapshot from a dictionary.

        Args:
            data: Dictionary with state, data, and timestamp keys.

        Returns:
            Reconstructed StateSnapshot instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If state value is not a valid DiscussState.
        """
        return cls(
            state=DiscussState(data["state"]),
            data=data.get("data", {}),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
        )


class StateStack:
    """Stack-based state tracker for nested state transitions.

    Supports push/pop semantics for saving and restoring state
    context during presentation/enquiry nesting. Enables infinite
    nesting depth: presentation -> enquiry -> prepare-enquiry ->
    presentation-enquiry -> enquiry -> ...

    Attributes:
        _stack: Internal list of StateSnapshot objects.
    """

    def __init__(self, snapshots: list[StateSnapshot] | None = None) -> None:
        """Initialize the state stack.

        Args:
            snapshots: Optional initial list of snapshots.
        """
        self._stack: list[StateSnapshot] = list(snapshots) if snapshots else []

    def push(self, snapshot: StateSnapshot) -> None:
        """Push a state snapshot onto the stack.

        Args:
            snapshot: The state snapshot to save.
        """
        self._stack.append(snapshot)

    def pop(self) -> StateSnapshot | None:
        """Pop the top state snapshot from the stack.

        Returns:
            The most recently pushed snapshot, or None if the stack is empty.
        """
        if not self._stack:
            return None
        return self._stack.pop()

    def peek(self) -> StateSnapshot | None:
        """View the top state snapshot without removing it.

        Returns:
            The most recently pushed snapshot, or None if the stack is empty.
        """
        if not self._stack:
            return None
        return self._stack[-1]

    @property
    def is_empty(self) -> bool:
        """Check whether the stack is empty.

        Returns:
            True if no snapshots are on the stack.
        """
        return len(self._stack) == 0

    def __len__(self) -> int:
        """Return the number of snapshots on the stack.

        Returns:
            Number of snapshots currently stored.
        """
        return len(self._stack)

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize the stack to a list of dictionaries.

        Returns:
            List of serialized snapshots, bottom-to-top order.
        """
        return [s.to_dict() for s in self._stack]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> StateStack:
        """Deserialize a stack from a list of dictionaries.

        Args:
            data: List of serialized snapshot dictionaries.

        Returns:
            Reconstructed StateStack instance.
        """
        snapshots = [StateSnapshot.from_dict(d) for d in data]
        return cls(snapshots)
