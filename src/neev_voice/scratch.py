"""Scratch pad module for artifact persistence.

Provides the ScratchPad class that creates timestamped flow directories
under `.scratch/neev/` for persisting audio, transcriptions, enriched
output, and discussion artifacts.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar


class ScratchPad:
    """Manages a timestamped scratch directory for a single flow invocation.

    Each invocation (listen or discuss) creates a unique folder:
    ``{base_dir}/{flow_type}/{YYYYMMDD-HHMMSS}_{uuid8}/``

    Provides well-known file paths and save methods for flow artifacts.

    Attributes:
        DEFAULT_BASE_DIR: Default base directory for scratch pad files.
        flow_type: Type of flow ('listen' or 'discussion').
        base_dir: Base directory for all scratch pad flows.
        flow_dir: Unique directory for this flow invocation.
    """

    DEFAULT_BASE_DIR: ClassVar[Path] = Path(".scratch/neev")

    def __init__(self, flow_type: str, base_dir: Path | None = None) -> None:
        """Initialize a new scratch pad for a flow invocation.

        Creates the flow directory immediately.

        Args:
            flow_type: Type of flow (e.g. 'listen', 'discussion').
            base_dir: Override base directory (defaults to DEFAULT_BASE_DIR).
        """
        self.flow_type = flow_type
        self.base_dir = base_dir or self.DEFAULT_BASE_DIR

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        self.flow_dir = self.base_dir / flow_type / f"{timestamp}_{short_id}"
        self.flow_dir.mkdir(parents=True, exist_ok=True)

    @property
    def audio_path(self) -> Path:
        """Path to the audio recording file.

        Returns:
            Path to audio.wav in the flow directory.
        """
        return self.flow_dir / "audio.wav"

    @property
    def transcription_path(self) -> Path:
        """Path to the transcription text file.

        Returns:
            Path to transcription.txt in the flow directory.
        """
        return self.flow_dir / "transcription.txt"

    @property
    def enriched_path(self) -> Path:
        """Path to the enriched markdown file.

        Returns:
            Path to enriched.md in the flow directory.
        """
        return self.flow_dir / "enriched.md"

    @property
    def metadata_path(self) -> Path:
        """Path to the metadata JSON file.

        Returns:
            Path to metadata.json in the flow directory.
        """
        return self.flow_dir / "metadata.json"

    def save_transcription(self, text: str) -> Path:
        """Save transcription text to the scratch pad.

        Args:
            text: Raw transcription text.

        Returns:
            Path to the saved transcription file.
        """
        self.transcription_path.write_text(text, encoding="utf-8")
        return self.transcription_path

    def save_enriched(self, content: str) -> Path:
        """Save enriched markdown content to the scratch pad.

        Args:
            content: Enriched markdown output from the enrichment agent.

        Returns:
            Path to the saved enriched file.
        """
        self.enriched_path.write_text(content, encoding="utf-8")
        return self.enriched_path

    def save_metadata(self, **kwargs: Any) -> Path:
        """Save metadata as JSON to the scratch pad.

        Includes a timestamp and the flow_type alongside any
        additional keyword arguments.

        Args:
            **kwargs: Arbitrary metadata key-value pairs.

        Returns:
            Path to the saved metadata file.
        """
        data = {
            "flow_type": self.flow_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flow_dir": str(self.flow_dir),
            **kwargs,
        }
        self.metadata_path.write_text(
            json.dumps(data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return self.metadata_path

    def save_section(self, index: int, data: dict[str, Any]) -> Path:
        """Save a discussion section result as JSON.

        Args:
            index: Section index (1-based).
            data: Section result data to serialize.

        Returns:
            Path to the saved section file.
        """
        path = self.flow_dir / f"section_{index:03d}.json"
        path.write_text(
            json.dumps(data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    @property
    def discussion_result_path(self) -> Path:
        """Path to the discussion result markdown file.

        Returns:
            Path to discussion-result.md in the flow directory.
        """
        return self.flow_dir / "discussion-result.md"

    def save_discussion_result(self, content: str) -> Path:
        """Save a discussion result markdown to the scratch pad.

        The discussion-result.md file contains a consolidated view of
        insights, plans, explorations, agreements, and disagreements
        from a document discussion session.

        Args:
            content: Markdown content for the discussion result.

        Returns:
            Path to the saved discussion-result.md file.
        """
        self.discussion_result_path.write_text(content, encoding="utf-8")
        return self.discussion_result_path

    def save_summary(self, data: dict[str, Any]) -> Path:
        """Save a discussion summary as JSON.

        Args:
            data: Summary data to serialize.

        Returns:
            Path to the saved summary file.
        """
        path = self.flow_dir / "summary.json"
        path.write_text(
            json.dumps(data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def get_latest_folder(cls, flow_type: str, base_dir: Path | None = None) -> Path | None:
        """Get the most recent flow directory for a given flow type.

        Folders are sorted lexicographically (timestamp-based names
        ensure chronological order).

        Args:
            flow_type: Type of flow (e.g. 'listen', 'discussion').
            base_dir: Override base directory.

        Returns:
            Path to the latest flow directory, or None if none exist.
        """
        parent = (base_dir or cls.DEFAULT_BASE_DIR) / flow_type
        if not parent.exists():
            return None
        folders = sorted(
            [d for d in parent.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        return folders[-1] if folders else None

    @classmethod
    def list_flows(cls, flow_type: str, base_dir: Path | None = None) -> list[Path]:
        """List all flow directories for a given flow type.

        Args:
            flow_type: Type of flow (e.g. 'listen', 'discussion').
            base_dir: Override base directory.

        Returns:
            Sorted list of flow directory paths (oldest first).
        """
        parent = (base_dir or cls.DEFAULT_BASE_DIR) / flow_type
        if not parent.exists():
            return []
        return sorted(
            [d for d in parent.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
