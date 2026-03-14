"""Session schema migration for the discuss subcommand.

Provides versioned schema migrations so older session.json files
are automatically upgraded when loaded. Each migration function
transforms raw dict data from version N to N+1. Also handles
migration of concepts.json files in session prepare directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

__all__ = ["CURRENT_SCHEMA_VERSION", "migrate_concepts_file", "migrate_session_data"]

CURRENT_SCHEMA_VERSION = 2
"""Current session schema version.

Version history:
    1 — Original v0.9.0 format (no schema_version field).
    2 — Added ``presentation_index`` and ``schema_version``.
        Normalized concept dicts with ``source_file`` and ``dependencies``.
"""


def migrate_session_data(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Migrate raw session dict to the current schema version.

    Applies sequential migration functions from the session's version
    up to ``CURRENT_SCHEMA_VERSION``. Safe to call on already-current
    data (returns unchanged with ``False``).

    Args:
        data: Raw session dictionary (e.g. from JSON).

    Returns:
        Tuple of (migrated data dict, True if any migration was applied).
    """
    version = data.get("schema_version", 1)

    if version >= CURRENT_SCHEMA_VERSION:
        return data, False

    original_version = version
    while version < CURRENT_SCHEMA_VERSION:
        migration_fn = _MIGRATIONS.get(version)
        if migration_fn is None:
            logger.error(
                "session_migration_missing",
                from_version=version,
                to_version=version + 1,
            )
            break
        data = migration_fn(data)
        version += 1

    logger.info(
        "session_migrated",
        session=data.get("name", "unknown"),
        from_version=original_version,
        to_version=version,
    )
    return data, True


def migrate_concepts_file(concepts_path: Path) -> bool:
    """Migrate a concepts.json file to ensure all fields are present.

    Normalizes each concept dict to include ``source_file`` and
    ``dependencies`` with proper defaults. Re-indexes concepts
    sequentially if indices are inconsistent.

    Args:
        concepts_path: Path to concepts.json file.

    Returns:
        True if the file was modified, False if already normalized.
    """
    if not concepts_path.exists():
        return False

    try:
        data = json.loads(concepts_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("concepts_migration_read_error", path=str(concepts_path))
        return False

    if not isinstance(data, list):
        return False

    modified = False
    for i, concept in enumerate(data):
        if not isinstance(concept, dict):
            continue

        # Ensure sequential index
        if concept.get("index") != i:
            concept["index"] = i
            modified = True

        # Ensure required fields with defaults
        if "source_file" not in concept:
            concept["source_file"] = ""
            modified = True
        if "dependencies" not in concept:
            concept["dependencies"] = []
            modified = True
        if "description" not in concept:
            concept["description"] = concept.get("title", f"Concept {i}")
            modified = True

    if modified:
        content = json.dumps(data, indent=2, default=str) + "\n"
        concepts_path.write_text(content, encoding="utf-8")
        logger.info("concepts_migrated", path=str(concepts_path), count=len(data))

    return modified


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from schema version 1 to 2.

    Adds:
        - ``presentation_index``: defaults to 0.
        - ``schema_version``: set to 2.
        - Normalizes ``concepts`` list entries with missing fields.

    Args:
        data: Session dict at version 1.

    Returns:
        Session dict at version 2.
    """
    data.setdefault("presentation_index", 0)
    data["schema_version"] = 2

    # Normalize concepts in session data
    concepts = data.get("concepts")
    if isinstance(concepts, list):
        for i, c in enumerate(concepts):
            if isinstance(c, dict):
                c.setdefault("source_file", "")
                c.setdefault("dependencies", [])
                c.setdefault("description", c.get("title", f"Concept {i}"))
                if c.get("index") != i:
                    c["index"] = i

    return data


_MIGRATIONS: dict[int, Any] = {
    1: _migrate_v1_to_v2,
}
"""Registry mapping source version to its migration function."""
