#!/usr/bin/env bash
# Pre-push hook: verify CHANGELOG.md has an entry for any version tag being pushed.
# Called by git pre-push with stdin lines: <local-ref> <local-sha> <remote-ref> <remote-sha>
set -euo pipefail

while read -r local_ref local_sha remote_ref remote_sha; do
    # Only check version tags (refs/tags/v*)
    if [[ "$local_ref" != refs/tags/v* ]]; then
        continue
    fi

    tag_name="${local_ref#refs/tags/}"
    version="${tag_name#v}"

    if ! grep -q "## \[${version}\]" CHANGELOG.md; then
        echo "ERROR: CHANGELOG.md has no entry for version ${version}"
        echo "Add a '## [${version}]' section to CHANGELOG.md before pushing tag ${tag_name}"
        exit 1
    fi

    echo "changelog-check: ${tag_name} -> CHANGELOG.md entry found"
done
