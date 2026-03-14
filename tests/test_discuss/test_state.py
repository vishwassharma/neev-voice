"""Tests for the discuss state machine definitions."""

from __future__ import annotations

import pytest

from neev_voice.discuss.state import (
    VALID_TRANSITIONS,
    DiscussState,
    StateSnapshot,
    StateStack,
    validate_transition,
)


class TestDiscussState:
    """Tests for DiscussState enum."""

    def test_all_states_exist(self) -> None:
        """All five states are defined."""
        assert len(DiscussState) == 5

    def test_state_values(self) -> None:
        """State string values match expected kebab-case."""
        assert DiscussState.PREPARE == "prepare"
        assert DiscussState.PRESENTATION == "presentation"
        assert DiscussState.ENQUIRY == "enquiry"
        assert DiscussState.PREPARE_ENQUIRY == "prepare-enquiry"
        assert DiscussState.PRESENTATION_ENQUIRY == "presentation-enquiry"

    def test_state_from_value(self) -> None:
        """States can be constructed from their string values."""
        assert DiscussState("prepare") is DiscussState.PREPARE
        assert DiscussState("prepare-enquiry") is DiscussState.PREPARE_ENQUIRY

    def test_invalid_state_raises(self) -> None:
        """Invalid state string raises ValueError."""
        with pytest.raises(ValueError):
            DiscussState("nonexistent")


class TestValidTransitions:
    """Tests for state transition validation."""

    def test_prepare_to_presentation(self) -> None:
        """Prepare can transition to presentation."""
        assert validate_transition(DiscussState.PREPARE, DiscussState.PRESENTATION)

    def test_prepare_cannot_skip(self) -> None:
        """Prepare cannot skip directly to enquiry."""
        assert not validate_transition(DiscussState.PREPARE, DiscussState.ENQUIRY)

    def test_presentation_to_enquiry(self) -> None:
        """Presentation can transition to enquiry."""
        assert validate_transition(DiscussState.PRESENTATION, DiscussState.ENQUIRY)

    def test_enquiry_to_prepare_enquiry(self) -> None:
        """Enquiry can transition to prepare-enquiry."""
        assert validate_transition(DiscussState.ENQUIRY, DiscussState.PREPARE_ENQUIRY)

    def test_enquiry_to_presentation_via_escape(self) -> None:
        """Enquiry can transition back to presentation (ESC pop stack)."""
        assert validate_transition(DiscussState.ENQUIRY, DiscussState.PRESENTATION)

    def test_enquiry_to_presentation_enquiry_via_escape(self) -> None:
        """Enquiry can transition back to presentation-enquiry (ESC pop stack)."""
        assert validate_transition(DiscussState.ENQUIRY, DiscussState.PRESENTATION_ENQUIRY)

    def test_prepare_enquiry_to_presentation_enquiry(self) -> None:
        """Prepare-enquiry transitions to presentation-enquiry."""
        assert validate_transition(DiscussState.PREPARE_ENQUIRY, DiscussState.PRESENTATION_ENQUIRY)

    def test_presentation_enquiry_to_presentation(self) -> None:
        """Presentation-enquiry can transition to presentation."""
        assert validate_transition(DiscussState.PRESENTATION_ENQUIRY, DiscussState.PRESENTATION)

    def test_presentation_enquiry_to_enquiry(self) -> None:
        """Presentation-enquiry can transition to enquiry."""
        assert validate_transition(DiscussState.PRESENTATION_ENQUIRY, DiscussState.ENQUIRY)

    def test_all_states_have_transitions(self) -> None:
        """Every state has at least one valid transition defined."""
        for state in DiscussState:
            assert state in VALID_TRANSITIONS, f"{state} missing from VALID_TRANSITIONS"
            assert len(VALID_TRANSITIONS[state]) > 0

    def test_no_self_transitions(self) -> None:
        """No state can transition to itself."""
        for state in DiscussState:
            assert not validate_transition(state, state)


class TestStateSnapshot:
    """Tests for StateSnapshot dataclass."""

    def test_create_with_defaults(self) -> None:
        """Snapshot can be created with just a state."""
        snap = StateSnapshot(state=DiscussState.PRESENTATION)
        assert snap.state == DiscussState.PRESENTATION
        assert snap.data == {}
        assert snap.timestamp  # non-empty

    def test_create_with_data(self) -> None:
        """Snapshot stores arbitrary data dict."""
        data = {"concept_index": 3, "position": 42}
        snap = StateSnapshot(state=DiscussState.PRESENTATION, data=data)
        assert snap.data["concept_index"] == 3
        assert snap.data["position"] == 42

    def test_to_dict(self) -> None:
        """Snapshot serializes to dictionary."""
        snap = StateSnapshot(
            state=DiscussState.ENQUIRY,
            data={"query": "test"},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = snap.to_dict()
        assert d["state"] == "enquiry"
        assert d["data"] == {"query": "test"}
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"

    def test_from_dict(self) -> None:
        """Snapshot can be deserialized from dictionary."""
        d = {
            "state": "presentation",
            "data": {"idx": 5},
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        snap = StateSnapshot.from_dict(d)
        assert snap.state == DiscussState.PRESENTATION
        assert snap.data == {"idx": 5}
        assert snap.timestamp == "2026-01-01T00:00:00+00:00"

    def test_from_dict_missing_optional_fields(self) -> None:
        """Snapshot from_dict handles missing optional fields."""
        d = {"state": "prepare"}
        snap = StateSnapshot.from_dict(d)
        assert snap.state == DiscussState.PREPARE
        assert snap.data == {}
        assert snap.timestamp  # auto-generated

    def test_from_dict_invalid_state_raises(self) -> None:
        """Snapshot from_dict raises ValueError for invalid state."""
        with pytest.raises(ValueError):
            StateSnapshot.from_dict({"state": "invalid"})

    def test_from_dict_missing_state_raises(self) -> None:
        """Snapshot from_dict raises KeyError for missing state."""
        with pytest.raises(KeyError):
            StateSnapshot.from_dict({"data": {}})

    def test_roundtrip(self) -> None:
        """Snapshot survives to_dict/from_dict roundtrip."""
        original = StateSnapshot(
            state=DiscussState.PRESENTATION_ENQUIRY,
            data={"answer": "42", "nested": True},
            timestamp="2026-03-14T12:00:00+00:00",
        )
        restored = StateSnapshot.from_dict(original.to_dict())
        assert restored.state == original.state
        assert restored.data == original.data
        assert restored.timestamp == original.timestamp


class TestStateStack:
    """Tests for StateStack push/pop/peek operations."""

    def test_empty_stack(self) -> None:
        """New stack is empty."""
        stack = StateStack()
        assert stack.is_empty
        assert len(stack) == 0

    def test_push_and_pop(self) -> None:
        """Push and pop return items in LIFO order."""
        stack = StateStack()
        snap1 = StateSnapshot(state=DiscussState.PRESENTATION, data={"idx": 1})
        snap2 = StateSnapshot(state=DiscussState.PRESENTATION_ENQUIRY, data={"idx": 2})

        stack.push(snap1)
        stack.push(snap2)

        assert len(stack) == 2
        assert not stack.is_empty

        popped = stack.pop()
        assert popped is not None
        assert popped.state == DiscussState.PRESENTATION_ENQUIRY

        popped = stack.pop()
        assert popped is not None
        assert popped.state == DiscussState.PRESENTATION

    def test_pop_empty_returns_none(self) -> None:
        """Pop on empty stack returns None."""
        stack = StateStack()
        assert stack.pop() is None

    def test_peek(self) -> None:
        """Peek returns top without removing it."""
        stack = StateStack()
        snap = StateSnapshot(state=DiscussState.PRESENTATION)
        stack.push(snap)

        assert stack.peek() is snap
        assert len(stack) == 1  # not removed

    def test_peek_empty_returns_none(self) -> None:
        """Peek on empty stack returns None."""
        stack = StateStack()
        assert stack.peek() is None

    def test_init_with_snapshots(self) -> None:
        """Stack can be initialized with existing snapshots."""
        snaps = [
            StateSnapshot(state=DiscussState.PRESENTATION),
            StateSnapshot(state=DiscussState.ENQUIRY),
        ]
        stack = StateStack(snaps)
        assert len(stack) == 2
        assert stack.peek().state == DiscussState.ENQUIRY

    def test_init_copies_list(self) -> None:
        """Stack copies the input list, doesn't reference it."""
        snaps = [StateSnapshot(state=DiscussState.PRESENTATION)]
        stack = StateStack(snaps)
        snaps.append(StateSnapshot(state=DiscussState.ENQUIRY))
        assert len(stack) == 1  # not affected by external mutation

    def test_to_dict(self) -> None:
        """Stack serializes to list of dicts."""
        stack = StateStack()
        stack.push(StateSnapshot(state=DiscussState.PRESENTATION, data={"a": 1}))
        stack.push(StateSnapshot(state=DiscussState.ENQUIRY, data={"b": 2}))

        d = stack.to_dict()
        assert len(d) == 2
        assert d[0]["state"] == "presentation"
        assert d[1]["state"] == "enquiry"

    def test_from_dict(self) -> None:
        """Stack can be deserialized from list of dicts."""
        data = [
            {"state": "presentation", "data": {"x": 1}},
            {"state": "enquiry", "data": {"y": 2}},
        ]
        stack = StateStack.from_dict(data)
        assert len(stack) == 2
        assert stack.peek().state == DiscussState.ENQUIRY

    def test_roundtrip(self) -> None:
        """Stack survives to_dict/from_dict roundtrip."""
        stack = StateStack()
        stack.push(StateSnapshot(state=DiscussState.PRESENTATION, data={"idx": 0}))
        stack.push(
            StateSnapshot(
                state=DiscussState.PRESENTATION_ENQUIRY,
                data={"answer": "yes"},
            )
        )

        restored = StateStack.from_dict(stack.to_dict())
        assert len(restored) == 2

        top = restored.pop()
        assert top.state == DiscussState.PRESENTATION_ENQUIRY
        assert top.data == {"answer": "yes"}

        bottom = restored.pop()
        assert bottom.state == DiscussState.PRESENTATION
        assert bottom.data == {"idx": 0}

    def test_nested_transitions_simulation(self) -> None:
        """Simulate nested presentation -> enquiry -> presentation-enquiry -> enquiry cycle."""
        stack = StateStack()

        # Enter enquiry from presentation
        stack.push(StateSnapshot(state=DiscussState.PRESENTATION, data={"concept": 3}))

        # Enter enquiry from presentation-enquiry (nested)
        stack.push(
            StateSnapshot(
                state=DiscussState.PRESENTATION_ENQUIRY,
                data={"answer_pos": 2},
            )
        )

        assert len(stack) == 2

        # Pop back to presentation-enquiry
        snap = stack.pop()
        assert snap.state == DiscussState.PRESENTATION_ENQUIRY
        assert snap.data["answer_pos"] == 2

        # Pop back to presentation
        snap = stack.pop()
        assert snap.state == DiscussState.PRESENTATION
        assert snap.data["concept"] == 3

        assert stack.is_empty
