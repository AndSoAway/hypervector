#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <base-revision> <head-revision> [maximum-changed-lines]" >&2
  exit 2
fi

base_revision="$1"
head_revision="$2"
maximum_changed_lines="${3:-1000}"

if [[ ! "${maximum_changed_lines}" =~ ^[0-9]+$ ]]; then
  echo "maximum changed lines must be a non-negative integer" >&2
  exit 2
fi

merge_base="$(git merge-base "${base_revision}" "${head_revision}")"
read -r additions deletions < <(
  git diff --numstat "${merge_base}" "${head_revision}" |
    awk '
      $1 ~ /^[0-9]+$/ { additions += $1 }
      $2 ~ /^[0-9]+$/ { deletions += $2 }
      END { print additions + 0, deletions + 0 }
    '
)

changed_lines=$((additions + deletions))
changed_files="$(
  git diff --name-only "${merge_base}" "${head_revision}" |
    awk 'END { print NR + 0 }'
)"

echo "PR size: ${changed_files} files, +${additions}/-${deletions} (${changed_lines} changed lines)"
echo "Review limit: ${maximum_changed_lines} changed lines"

if ((changed_lines > maximum_changed_lines)); then
  echo "::error::PR has ${changed_lines} changed lines; reduce it to ${maximum_changed_lines} or fewer."
  exit 1
fi
