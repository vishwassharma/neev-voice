#!/usr/bin/env python3
"""Generate structured release notes from git history between tags.

Parses commit messages using conventional commit prefixes (feat, fix,
refactor, docs, test, chore, ci, perf, style, build) and groups them
into categorized sections. Falls back to extracting the CHANGELOG.md
entry for the given version when available.

Usage:
    # Auto-detect latest tag
    python scripts/generate_release_notes.py

    # Specific version range
    python scripts/generate_release_notes.py --from v0.1.0 --to v0.2.0

    # Extract from CHANGELOG.md
    python scripts/generate_release_notes.py --from-changelog --version 0.2.0

    # Output to file
    python scripts/generate_release_notes.py --output release-notes.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Conventional commit type -> display heading
COMMIT_TYPE_MAP: dict[str, str] = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "refactor": "Refactoring",
    "docs": "Documentation",
    "test": "Tests",
    "chore": "Chores",
    "ci": "CI/CD",
    "perf": "Performance",
    "style": "Style",
    "build": "Build",
}

# Pattern: type(scope): description  OR  type: description
CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?:\s*(?P<desc>.+)$",
    re.IGNORECASE,
)


@dataclass
class Commit:
    """A parsed git commit.

    Attributes:
        hash: Short commit hash.
        subject: Full commit subject line.
        commit_type: Conventional commit type (feat, fix, etc.) or empty.
        scope: Optional scope from conventional commit.
        description: Commit description without the type prefix.
    """

    hash: str
    subject: str
    commit_type: str = ""
    scope: str = ""
    description: str = ""


@dataclass
class ReleaseNotes:
    """Grouped release notes ready for rendering.

    Attributes:
        version: Version string (e.g. "0.2.0").
        from_ref: Git ref for the start of the range.
        to_ref: Git ref for the end of the range.
        grouped: Mapping of category heading to list of commits.
        uncategorized: Commits that don't match conventional format.
    """

    version: str
    from_ref: str
    to_ref: str
    grouped: dict[str, list[Commit]] = field(default_factory=dict)
    uncategorized: list[Commit] = field(default_factory=list)


def run_git(*args: str) -> str:
    """Run a git command and return stripped stdout.

    Args:
        *args: Git subcommand and arguments.

    Returns:
        Stripped stdout string.

    Raises:
        SystemExit: If the git command fails.
    """
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_latest_tag() -> str | None:
    """Get the most recent git tag.

    Returns:
        Tag name string, or None if no tags exist.
    """
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_previous_tag(current_tag: str) -> str | None:
    """Get the tag before the given tag.

    Args:
        current_tag: The current tag to look before.

    Returns:
        Previous tag name, or None if no previous tag exists.
    """
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", f"{current_tag}^"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_commits(from_ref: str | None, to_ref: str) -> list[Commit]:
    """Get commits between two git refs.

    Args:
        from_ref: Start ref (exclusive). If None, gets all commits up to to_ref.
        to_ref: End ref (inclusive).

    Returns:
        List of parsed Commit objects.
    """
    if from_ref:
        log_range = f"{from_ref}..{to_ref}"
    else:
        log_range = to_ref

    raw = run_git("log", log_range, "--pretty=format:%h %s")
    if not raw:
        return []

    commits = []
    for line in raw.splitlines():
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        commit_hash, subject = parts
        commit = Commit(hash=commit_hash, subject=subject)

        match = CONVENTIONAL_RE.match(subject)
        if match:
            commit.commit_type = match.group("type").lower()
            commit.scope = match.group("scope") or ""
            commit.description = match.group("desc")

        commits.append(commit)

    return commits


def parse_commit(subject: str) -> Commit:
    """Parse a single commit subject line into a Commit.

    Args:
        subject: The commit subject line (without hash).

    Returns:
        Parsed Commit with type/scope/description extracted if conventional.
    """
    commit = Commit(hash="", subject=subject)
    match = CONVENTIONAL_RE.match(subject)
    if match:
        commit.commit_type = match.group("type").lower()
        commit.scope = match.group("scope") or ""
        commit.description = match.group("desc")
    return commit


def group_commits(commits: list[Commit]) -> ReleaseNotes:
    """Group commits by conventional commit type.

    Args:
        commits: List of Commit objects to categorize.

    Returns:
        ReleaseNotes with commits sorted into categories.
    """
    notes = ReleaseNotes(version="", from_ref="", to_ref="")

    for commit in commits:
        if commit.commit_type and commit.commit_type in COMMIT_TYPE_MAP:
            heading = COMMIT_TYPE_MAP[commit.commit_type]
            notes.grouped.setdefault(heading, []).append(commit)
        else:
            notes.uncategorized.append(commit)

    return notes


def render_notes(notes: ReleaseNotes) -> str:
    """Render ReleaseNotes to a markdown string.

    Args:
        notes: Grouped release notes to render.

    Returns:
        Formatted markdown string.
    """
    lines: list[str] = []

    if notes.version:
        lines.append(f"# Release {notes.version}")
    else:
        lines.append("# Release Notes")
    lines.append("")

    if notes.from_ref:
        lines.append(f"**Changes:** `{notes.from_ref}` → `{notes.to_ref}`")
        lines.append("")

    # Render grouped sections in a stable order
    section_order = list(COMMIT_TYPE_MAP.values())
    for heading in section_order:
        commits = notes.grouped.get(heading, [])
        if not commits:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for commit in commits:
            scope = f"**{commit.scope}:** " if commit.scope else ""
            desc = commit.description or commit.subject
            lines.append(f"- {scope}{desc} (`{commit.hash}`)")
        lines.append("")

    if notes.uncategorized:
        lines.append("## Other Changes")
        lines.append("")
        for commit in notes.uncategorized:
            lines.append(f"- {commit.subject} (`{commit.hash}`)")
        lines.append("")

    return "\n".join(lines)


def extract_changelog_version(version: str, changelog_path: Path | None = None) -> str | None:
    """Extract a specific version's entry from CHANGELOG.md.

    Parses CHANGELOG.md looking for a heading matching ``## [version]``
    and returns all content up to the next version heading.

    Args:
        version: Version string without 'v' prefix (e.g. "0.2.0").
        changelog_path: Path to CHANGELOG.md. Defaults to repo root.

    Returns:
        Markdown string for that version, or None if not found.
    """
    if changelog_path is None:
        changelog_path = Path("CHANGELOG.md")

    if not changelog_path.exists():
        return None

    content = changelog_path.read_text(encoding="utf-8")
    # Match ## [0.2.0] or ## [0.2.0] - 2026-03-03
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\].*$",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        return None

    start = match.start()
    # Find the next ## [...] heading after this one
    next_heading = re.search(r"^## \[", content[match.end() :], re.MULTILINE)
    if next_heading:
        end = match.end() + next_heading.start()
    else:
        end = len(content)

    return content[start:end].rstrip()


def main() -> None:
    """CLI entry point for release notes generation."""
    parser = argparse.ArgumentParser(
        description="Generate release notes from git history or CHANGELOG.md",
    )
    parser.add_argument(
        "--from",
        dest="from_ref",
        default=None,
        help="Start git ref (exclusive). Auto-detects previous tag if omitted.",
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        default=None,
        help="End git ref (inclusive). Defaults to latest tag or HEAD.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string for the release heading (e.g. 0.2.0).",
    )
    parser.add_argument(
        "--from-changelog",
        action="store_true",
        help="Extract notes from CHANGELOG.md instead of git log.",
    )
    parser.add_argument(
        "--changelog-path",
        default=None,
        help="Path to CHANGELOG.md (default: repo root).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write output to file instead of stdout.",
    )

    args = parser.parse_args()

    # Changelog extraction mode
    if args.from_changelog:
        version = args.version
        if not version:
            tag = get_latest_tag()
            if tag:
                version = tag.lstrip("v")
            else:
                print("Error: --version required when no tags exist", file=sys.stderr)
                sys.exit(1)

        changelog_path = Path(args.changelog_path) if args.changelog_path else None
        entry = extract_changelog_version(version, changelog_path)
        if entry is None:
            print(f"Error: no CHANGELOG.md entry for version {version}", file=sys.stderr)
            sys.exit(1)
        output = entry
    else:
        # Git log mode
        to_ref = args.to_ref or get_latest_tag() or "HEAD"
        from_ref = args.from_ref
        if from_ref is None and to_ref != "HEAD":
            from_ref = get_previous_tag(to_ref)

        version = args.version
        if not version and to_ref != "HEAD":
            version = to_ref.lstrip("v")

        commits = get_commits(from_ref, to_ref)
        if not commits:
            print("No commits found in range.", file=sys.stderr)
            sys.exit(1)

        notes = group_commits(commits)
        notes.version = version or ""
        notes.from_ref = from_ref or ""
        notes.to_ref = to_ref
        output = render_notes(notes)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(f"Release notes written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
