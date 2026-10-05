#!/usr/bin/env sh
# run-gates.sh <vcs>
#
# Body of the `run_gates` deterministic step: the checks Backend CI
# (.github/workflows/backend.yaml) runs on every PR, run locally before
# anything is pushed — line endings, then `poe lint`, `poe types` and
# `poe cov`. A failure stops the run here with the failing tool's output,
# instead of a PR that is red from its first CI run.
set -eu

vcs="${1:?run-gates.sh: vcs argument required}"

# The same files CI checks: everything tracked, minus the saved HTML
# fixtures, which keep the endings they were captured with. Read from the
# files themselves (a non-colocated jj workspace has no git index).
case "$vcs" in
  jj) files=$(jj file list) ;;
  git) files=$(git ls-files) ;;
  *)
    echo "run-gates.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
    exit 1
    ;;
esac
bad=$(printf '%s\n' "$files" | grep -v '^tests/assets/html/' \
  | tr '\n' '\0' | xargs -0 -r grep -lI "$(printf '\r')" || true)
if [ -n "$bad" ]; then
  echo "run-gates.sh: these files have CRLF line endings:" >&2
  printf '%s\n' "$bad" >&2
  exit 1
fi

uv run --frozen poe lint
uv run --frozen poe types
uv run --frozen poe cov
