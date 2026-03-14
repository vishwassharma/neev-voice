"""Session management for discuss subcommand.

Handles creation, persistence, and retrieval of discuss sessions.
Each session is stored as a JSON file in the scratch pad under
``.scratch/neev/discuss/<session-name>/session.json``.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neev_voice.discuss.migration import CURRENT_SCHEMA_VERSION, migrate_session_data
from neev_voice.discuss.state import DiscussState, StateStack

__all__ = ["SessionInfo", "SessionManager"]


@dataclass
class SessionInfo:
    """Persistent session data for a discuss session.

    Attributes:
        name: Session name (kebab-case).
        research_path: Path to the research input folder.
        source_path: Path to the source code root.
        output_path: Path to the output folder for generated content.
        state: Current state in the discuss state machine.
        state_stack: Stack of saved state snapshots for nested transitions.
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 last-updated timestamp.
        prepare_complete: Whether the prepare phase has finished.
        presentation_index: Index of the next concept to present (resume point).
        schema_version: Schema version for migration tracking.
        concepts: List of extracted concept metadata dicts, or None.
    """

    name: str
    research_path: str
    source_path: str
    output_path: str
    state: DiscussState = DiscussState.PREPARE
    state_stack: StateStack = field(default_factory=StateStack)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    prepare_complete: bool = False
    presentation_index: int = 0
    schema_version: int = CURRENT_SCHEMA_VERSION
    concepts: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize session info to a JSON-compatible dictionary.

        Returns:
            Dictionary with all session fields serialized.
        """
        return {
            "name": self.name,
            "research_path": self.research_path,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "state": self.state.value,
            "state_stack": self.state_stack.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "prepare_complete": self.prepare_complete,
            "presentation_index": self.presentation_index,
            "schema_version": self.schema_version,
            "concepts": self.concepts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionInfo:
        """Deserialize session info from a dictionary.

        Args:
            data: Dictionary with session fields.

        Returns:
            Reconstructed SessionInfo instance.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If state value is invalid.
        """
        return cls(
            name=data["name"],
            research_path=data["research_path"],
            source_path=data["source_path"],
            output_path=data["output_path"],
            state=DiscussState(data["state"]),
            state_stack=StateStack.from_dict(data.get("state_stack", [])),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            prepare_complete=data.get("prepare_complete", False),
            presentation_index=data.get("presentation_index", 0),
            schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
            concepts=data.get("concepts"),
        )


class SessionManager:
    """Manages discuss session lifecycle and persistence.

    Sessions are stored as JSON files in the scratch pad directory
    at ``.scratch/neev/discuss/<session-name>/session.json``.
    Uses atomic writes (temp file + rename) to prevent corruption.

    Attributes:
        DEFAULT_BASE_DIR: Default base directory for session storage.
        base_dir: Active base directory for this manager instance.
    """

    DEFAULT_BASE_DIR: Path = Path(".scratch/neev/discuss")

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize the session manager.

        Args:
            base_dir: Override base directory (defaults to DEFAULT_BASE_DIR).
        """
        self.base_dir = base_dir or self.DEFAULT_BASE_DIR

    def session_dir(self, name: str) -> Path:
        """Get the directory path for a named session.

        Args:
            name: Session name.

        Returns:
            Path to the session directory.
        """
        return self.base_dir / name

    def session_file(self, name: str) -> Path:
        """Get the session.json file path for a named session.

        Args:
            name: Session name.

        Returns:
            Path to the session.json file.
        """
        return self.session_dir(name) / "session.json"

    def output_dir(self, session: SessionInfo) -> Path:
        """Get the output directory for a session.

        Args:
            session: Session info with output_path.

        Returns:
            Path to the session's output directory.
        """
        return Path(session.output_path)

    def create_session(
        self,
        name: str,
        research_path: str | Path,
        source_path: str | Path,
        output_path: str | Path | None = None,
    ) -> SessionInfo:
        """Create a new discuss session.

        Creates the session directory and initializes session.json
        with the PREPARE state.

        Args:
            name: Session name (kebab-case).
            research_path: Path to research input folder.
            source_path: Path to source code root.
            output_path: Optional output folder (defaults to session dir / output).

        Returns:
            The created SessionInfo instance.

        Raises:
            FileExistsError: If a session with this name already exists.
        """
        session_path = self.session_dir(name)
        if session_path.exists() and self.session_file(name).exists():
            raise FileExistsError(f"Session '{name}' already exists at {session_path}")

        effective_output = str(output_path) if output_path else str(session_path / "output")

        session = SessionInfo(
            name=name,
            research_path=str(research_path),
            source_path=str(source_path),
            output_path=effective_output,
        )

        session_path.mkdir(parents=True, exist_ok=True)
        self.save_session(session)
        return session

    def load_session(self, name: str) -> SessionInfo | None:
        """Load a session from disk by name.

        Automatically migrates older schema versions to the current
        version and persists the upgraded session.

        Args:
            name: Session name to load.

        Returns:
            The loaded SessionInfo, or None if not found.
        """
        session_path = self.session_file(name)
        if not session_path.exists():
            return None
        try:
            data = json.loads(session_path.read_text(encoding="utf-8"))
            data, migrated = migrate_session_data(data)
            session = SessionInfo.from_dict(data)
            if migrated:
                self.save_session(session)
            return session
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def save_session(self, session: SessionInfo) -> None:
        """Persist session info to disk using atomic write.

        Writes to a temporary file then renames to prevent corruption
        from interrupted writes.

        Args:
            session: Session info to save.
        """
        session.updated_at = datetime.now(UTC).isoformat()
        target = self.session_file(session.name)
        target.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(session.to_dict(), indent=2, default=str) + "\n"

        # Atomic write: write to temp file in same dir, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=".session_",
            suffix=".tmp",
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(target)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def list_sessions(self) -> list[str]:
        """List all session names in the base directory.

        Returns:
            Sorted list of session names that have valid session.json files.
        """
        if not self.base_dir.exists():
            return []
        return [
            d.name
            for d in sorted(self.base_dir.iterdir())
            if d.is_dir() and (d / "session.json").exists()
        ]

    def get_latest_session(self) -> SessionInfo | None:
        """Get the most recently updated session.

        Returns:
            The session with the latest updated_at timestamp, or None.
        """
        sessions = self.list_sessions()
        if not sessions:
            return None

        latest: SessionInfo | None = None
        for name in sessions:
            session = self.load_session(name)
            if session is None:
                continue
            if latest is None or session.updated_at > latest.updated_at:
                latest = session
        return latest

    def delete_session(self, name: str) -> bool:
        """Delete a session and its directory.

        Args:
            name: Session name to delete.

        Returns:
            True if the session was deleted, False if not found.
        """
        import shutil

        session_path = self.session_dir(name)
        if not session_path.exists():
            return False
        shutil.rmtree(session_path)
        return True
