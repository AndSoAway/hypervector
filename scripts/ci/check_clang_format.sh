#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <base-revision> <head-revision>" >&2
  exit 2
fi

base_revision="$1"
head_revision="$2"
clang_format_binary="${CLANG_FORMAT_BINARY:-clang-format}"
git_clang_format_binary="${GIT_CLANG_FORMAT_BINARY:-git-clang-format}"
merge_base="$(git merge-base "${base_revision}" "${head_revision}")"

if ! command -v "${clang_format_binary}" >/dev/null 2>&1; then
  echo "clang-format executable '${clang_format_binary}' was not found" >&2
  exit 2
fi

if ! command -v "${git_clang_format_binary}" >/dev/null 2>&1; then
  echo "git-clang-format executable '${git_clang_format_binary}' was not found" >&2
  exit 2
fi

set +e
format_diff="$(
  "${git_clang_format_binary}" \
    --binary "${clang_format_binary}" \
    --diff \
    --extensions c,cc,cpp,cxx,h,hh,hpp,hxx \
    "${merge_base}" "${head_revision}"
)"
format_status=$?
set -e

if grep -q '^diff --git ' <<<"${format_diff}"; then
  printf '%s\n' "${format_diff}"
  echo "::error::Changed C++ lines are not clang-format compliant."
  exit 1
fi

if ((format_status != 0)); then
  echo "git-clang-format failed with status ${format_status}" >&2
  exit 2
fi

if [[ -n "${format_diff}" ]]; then
  printf '%s\n' "${format_diff}"
fi
