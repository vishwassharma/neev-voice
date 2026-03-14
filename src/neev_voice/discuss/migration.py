"""Session schema migration for the discuss subcommand.

Provides versioned schema migrations so older session.json files
are automatically upgraded when loaded. Each migration function
transforms raw dict data from version N to N+1.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

__all__ = ["CURRENT_SCHEMA_VERSION", "migrate_session_data"]

CURRENT_SCHEMA_VERSION = 2
"""Current session schema version.

Version history:
    1 — Original v0.9.0 format (no schema_version field).
    2 — Added ``presentation_index`` and ``schema_version``.
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


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from schema version 1 to 2.

    Adds:
        - ``presentation_index``: defaults to 0.
        - ``schema_version``: set to 2.

    Args:
        data: Session dict at version 1.

    Returns:
        Session dict at version 2.
    """
    data.setdefault("presentation_index", 0)
    data["schema_version"] = 2
    return data


_MIGRATIONS: dict[int, Any] = {
    1: _migrate_v1_to_v2,
}
"""Registry mapping source version to its migration function."""
