"""Session portability for export and import of discuss sessions.

Handles packaging sessions into zip archives for transfer between
machines and importing them back with path remapping.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from neev_voice.discuss.session import SessionInfo, SessionManager

logger = structlog.get_logger(__name__)

__all__ = ["export_session", "import_session"]

MANIFEST_VERSION = 1
"""Current manifest format version."""


def export_session(
    session_manager: SessionManager,
    session_name: str,
    output_path: Path | None = None,
) -> Path:
    """Export a discuss session to a zip archive.

    Packages the session directory (session.json, prepare/, output/)
    and copies research documents into a portable zip archive with
    a manifest for import on another machine.

    Args:
        session_manager: Session manager for loading session data.
        session_name: Name of the session to export.
        output_path: Directory to write the zip file. Defaults to
            the current working directory.

    Returns:
        Path to the created zip file.

    Raises:
        FileNotFoundError: If the session does not exist.
    """
    session = session_manager.load_session(session_name)
    if session is None:
        raise FileNotFoundError(f"Session '{session_name}' not found")

    session_dir = session_manager.session_dir(session_name)
    output_path = output_path or Path.cwd()
    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / f"{session_name}.zip"

    manifest = _build_manifest(session)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")

        # Write session.json
        session_file = session_dir / "session.json"
        if session_file.exists():
            zf.write(session_file, "session.json")

        # Write prepare/ directory
        _add_directory_to_zip(zf, session_dir / "prepare", "prepare")

        # Write output/ directory
        output_dir = Path(session.output_path)
        if output_dir.exists() and output_dir.is_dir():
            _add_directory_to_zip(zf, output_dir, "output")

        # Copy research documents
        research_path = Path(session.research_path)
        if research_path.exists() and research_path.is_dir():
            _add_directory_to_zip(zf, research_path, "research")

    logger.info(
        "session_exported",
        session=session_name,
        zip_path=str(zip_path),
    )
    return zip_path


def import_session(
    session_manager: SessionManager,
    zip_path: Path,
    research_dest: Path | None = None,
) -> SessionInfo:
    """Import a discuss session from a zip archive.

    Extracts the zip, creates a session directory in the local discuss
    base, copies research documents, and updates session paths.

    Args:
        session_manager: Session manager for creating sessions.
        zip_path: Path to the zip archive to import.
        research_dest: Directory to place research documents. Defaults
            to ``<session_dir>/research/``.

    Returns:
        The imported SessionInfo with updated local paths.

    Raises:
        FileNotFoundError: If the zip file does not exist.
        ValueError: If the zip is not a valid session export.
        FileExistsError: If a session with the same name already exists.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Read and validate manifest
        if "manifest.json" not in zf.namelist():
            raise ValueError("Invalid session export: missing manifest.json")

        manifest = json.loads(zf.read("manifest.json"))
        session_name = manifest.get("session_name")
        if not session_name:
            raise ValueError("Invalid manifest: missing session_name")

        # Check for existing session
        session_dir = session_manager.session_dir(session_name)
        if session_dir.exists() and (session_dir / "session.json").exists():
            raise FileExistsError(f"Session '{session_name}' already exists")

        # Create session directory
        session_dir.mkdir(parents=True, exist_ok=True)

        # Determine research destination
        effective_research = research_dest or (session_dir / "research")

        # Extract files
        for member in zf.namelist():
            if member == "manifest.json":
                continue

            if (
                member == "session.json"
                or member.startswith("prepare/")
                or member.startswith("output/")
            ):
                zf.extract(member, session_dir)
            elif member.startswith("research/"):
                # Extract research files relative to research_dest
                rel_path = member[len("research/") :]
                if rel_path:
                    dest = effective_research / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    # Load and update session paths
    session = _load_and_remap_session(
        session_dir,
        research_path=str(effective_research),
        output_path=str(session_dir / "output"),
    )
    session_manager.save_session(session)

    logger.info(
        "session_imported",
        session=session_name,
        session_dir=str(session_dir),
    )
    return session


def _build_manifest(session: SessionInfo) -> dict[str, Any]:
    """Build an export manifest dictionary.

    Args:
        session: Session to build manifest for.

    Returns:
        Manifest dictionary with metadata.
    """
    from importlib.metadata import version

    try:
        pkg_version = version("neev-voice")
    except Exception:
        pkg_version = "unknown"

    return {
        "manifest_version": MANIFEST_VERSION,
        "session_name": session.name,
        "exported_at": datetime.now(UTC).isoformat(),
        "neev_voice_version": pkg_version,
        "original_paths": {
            "research_path": session.research_path,
            "source_path": session.source_path,
            "output_path": session.output_path,
        },
    }


def _add_directory_to_zip(zf: zipfile.ZipFile, dir_path: Path, prefix: str) -> None:
    """Recursively add a directory's contents to a zip archive.

    Args:
        zf: Open ZipFile to write to.
        dir_path: Directory to add.
        prefix: Prefix path within the zip archive.
    """
    if not dir_path.exists():
        return
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file():
            arcname = f"{prefix}/{file_path.relative_to(dir_path)}"
            zf.write(file_path, arcname)


def _load_and_remap_session(
    session_dir: Path,
    research_path: str,
    output_path: str,
) -> SessionInfo:
    """Load session.json and remap paths to local equivalents.

    Args:
        session_dir: Directory containing session.json.
        research_path: New local research path.
        output_path: New local output path.

    Returns:
        SessionInfo with updated paths.

    Raises:
        ValueError: If session.json cannot be loaded.
    """
    session_file = session_dir / "session.json"
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        session = SessionInfo.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Invalid session.json in export: {e}") from e

    session.research_path = research_path
    session.output_path = output_path
    # source_path left as original — user must set for their machine
    return session
