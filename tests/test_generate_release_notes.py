"""Tests for the release notes generation script."""

import sys
from pathlib import Path

# Add scripts dir to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_release_notes import (
    Commit,
    ReleaseNotes,
    extract_changelog_version,
    group_commits,
    parse_commit,
    render_notes,
)


class TestParseCommit:
    """Tests for parse_commit function."""

    def test_conventional_feat(self):
        """Test parsing a conventional feat commit."""
        commit = parse_commit("feat: add push-to-talk recording")
        assert commit.commit_type == "feat"
        assert commit.scope == ""
        assert commit.description == "add push-to-talk recording"

    def test_conventional_fix_with_scope(self):
        """Test parsing a conventional fix commit with scope."""
        commit = parse_commit("fix(audio): correct silence detection threshold")
        assert commit.commit_type == "fix"
        assert commit.scope == "audio"
        assert commit.description == "correct silence detection threshold"

    def test_conventional_refactor(self):
        """Test parsing a conventional refactor commit."""
        commit = parse_commit("refactor: simplify config loading")
        assert commit.commit_type == "refactor"
        assert commit.description == "simplify config loading"

    def test_non_conventional(self):
        """Test parsing a non-conventional commit."""
        commit = parse_commit("Update README with new badges")
        assert commit.commit_type == ""
        assert commit.scope == ""
        assert commit.description == ""
        assert commit.subject == "Update README with new badges"

    def test_ci_type(self):
        """Test parsing a ci commit."""
        commit = parse_commit("ci: add Python 3.13 to matrix")
        assert commit.commit_type == "ci"
        assert commit.description == "add Python 3.13 to matrix"

    def test_docs_type(self):
        """Test parsing a docs commit."""
        commit = parse_commit("docs: update CHANGELOG for v0.2.0")
        assert commit.commit_type == "docs"

    def test_test_type(self):
        """Test parsing a test commit."""
        commit = parse_commit("test: add ESC cancel tests")
        assert commit.commit_type == "test"

    def test_case_insensitive(self):
        """Test parsing is case-insensitive for type."""
        commit = parse_commit("Feat: uppercase type")
        assert commit.commit_type == "feat"


class TestGroupCommits:
    """Tests for group_commits function."""

    def test_groups_by_type(self):
        """Test commits are grouped by conventional type."""
        commits = [
            Commit(
                hash="abc", subject="feat: feature A", commit_type="feat", description="feature A"
            ),
            Commit(hash="def", subject="fix: fix B", commit_type="fix", description="fix B"),
            Commit(
                hash="ghi", subject="feat: feature C", commit_type="feat", description="feature C"
            ),
        ]
        notes = group_commits(commits)
        assert "Features" in notes.grouped
        assert len(notes.grouped["Features"]) == 2
        assert "Bug Fixes" in notes.grouped
        assert len(notes.grouped["Bug Fixes"]) == 1

    def test_uncategorized(self):
        """Test non-conventional commits go to uncategorized."""
        commits = [
            Commit(hash="abc", subject="random change"),
        ]
        notes = group_commits(commits)
        assert len(notes.uncategorized) == 1
        assert len(notes.grouped) == 0

    def test_mixed(self):
        """Test mix of conventional and non-conventional commits."""
        commits = [
            Commit(hash="a", subject="feat: new", commit_type="feat", description="new"),
            Commit(hash="b", subject="misc update"),
        ]
        notes = group_commits(commits)
        assert len(notes.grouped.get("Features", [])) == 1
        assert len(notes.uncategorized) == 1

    def test_empty_commits(self):
        """Test empty commit list produces empty notes."""
        notes = group_commits([])
        assert len(notes.grouped) == 0
        assert len(notes.uncategorized) == 0

    def test_unknown_type_goes_uncategorized(self):
        """Test unknown conventional type goes to uncategorized."""
        commits = [
            Commit(hash="a", subject="wip: half done", commit_type="wip", description="half done"),
        ]
        notes = group_commits(commits)
        assert len(notes.uncategorized) == 1


class TestRenderNotes:
    """Tests for render_notes function."""

    def test_basic_render(self):
        """Test rendering with a version heading."""
        notes = ReleaseNotes(
            version="1.0.0",
            from_ref="v0.9.0",
            to_ref="v1.0.0",
            grouped={
                "Features": [
                    Commit(
                        hash="abc",
                        subject="feat: new feature",
                        commit_type="feat",
                        description="new feature",
                    ),
                ],
            },
        )
        result = render_notes(notes)
        assert "# Release 1.0.0" in result
        assert "## Features" in result
        assert "- new feature (`abc`)" in result

    def test_render_with_scope(self):
        """Test rendering includes scope in bold."""
        notes = ReleaseNotes(
            version="1.0.0",
            from_ref="",
            to_ref="HEAD",
            grouped={
                "Bug Fixes": [
                    Commit(
                        hash="def",
                        subject="fix(cli): broken arg",
                        commit_type="fix",
                        scope="cli",
                        description="broken arg",
                    ),
                ],
            },
        )
        result = render_notes(notes)
        assert "**cli:** broken arg" in result

    def test_render_uncategorized(self):
        """Test rendering includes Other Changes section."""
        notes = ReleaseNotes(
            version="",
            from_ref="",
            to_ref="HEAD",
            uncategorized=[
                Commit(hash="xyz", subject="misc tweak"),
            ],
        )
        result = render_notes(notes)
        assert "## Other Changes" in result
        assert "misc tweak" in result

    def test_render_no_version(self):
        """Test rendering without version uses generic heading."""
        notes = ReleaseNotes(version="", from_ref="", to_ref="HEAD")
        result = render_notes(notes)
        assert "# Release Notes" in result

    def test_render_shows_ref_range(self):
        """Test rendering shows from/to ref range."""
        notes = ReleaseNotes(version="1.0.0", from_ref="v0.9.0", to_ref="v1.0.0")
        result = render_notes(notes)
        assert "`v0.9.0` → `v1.0.0`" in result

    def test_render_section_order(self):
        """Test sections are rendered in consistent order (Features before Bug Fixes)."""
        notes = ReleaseNotes(
            version="1.0.0",
            from_ref="",
            to_ref="HEAD",
            grouped={
                "Bug Fixes": [
                    Commit(hash="a", subject="fix: x", commit_type="fix", description="x"),
                ],
                "Features": [
                    Commit(hash="b", subject="feat: y", commit_type="feat", description="y"),
                ],
            },
        )
        result = render_notes(notes)
        feat_pos = result.index("## Features")
        fix_pos = result.index("## Bug Fixes")
        assert feat_pos < fix_pos


class TestExtractChangelogVersion:
    """Tests for extract_changelog_version function."""

    def test_extracts_version_entry(self, tmp_path):
        """Test extracting a specific version from CHANGELOG.md."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "## [0.2.0] - 2026-03-03\n\n"
            "### Added\n\n"
            "- Feature A\n"
            "- Feature B\n\n"
            "## [0.1.0] - 2026-02-01\n\n"
            "### Added\n\n"
            "- Initial release\n",
            encoding="utf-8",
        )
        result = extract_changelog_version("0.2.0", changelog)
        assert result is not None
        assert "## [0.2.0]" in result
        assert "Feature A" in result
        assert "Feature B" in result
        assert "0.1.0" not in result

    def test_extracts_last_version(self, tmp_path):
        """Test extracting the last version entry (no next heading)."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [0.1.0] - 2026-02-01\n\n### Added\n\n- Initial release\n",
            encoding="utf-8",
        )
        result = extract_changelog_version("0.1.0", changelog)
        assert result is not None
        assert "Initial release" in result

    def test_returns_none_for_missing_version(self, tmp_path):
        """Test returns None when version not found."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [0.1.0]\n\n- stuff\n", encoding="utf-8")
        result = extract_changelog_version("9.9.9", changelog)
        assert result is None

    def test_returns_none_for_missing_file(self, tmp_path):
        """Test returns None when CHANGELOG.md doesn't exist."""
        result = extract_changelog_version("0.1.0", tmp_path / "nope.md")
        assert result is None

    def test_handles_date_in_heading(self, tmp_path):
        """Test handles version heading with date suffix."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "## [1.2.3] - 2026-06-15\n\n### Fixed\n\n- Bug fix\n",
            encoding="utf-8",
        )
        result = extract_changelog_version("1.2.3", changelog)
        assert result is not None
        assert "Bug fix" in result

    def test_does_not_match_partial_version(self, tmp_path):
        """Test 0.2.0 does not match 0.2.0-rc1."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "## [0.2.0-rc1] - 2026-03-01\n\n- rc stuff\n\n"
            "## [0.2.0] - 2026-03-03\n\n- release stuff\n",
            encoding="utf-8",
        )
        result = extract_changelog_version("0.2.0", changelog)
        assert result is not None
        assert "release stuff" in result
        assert "rc stuff" not in result
