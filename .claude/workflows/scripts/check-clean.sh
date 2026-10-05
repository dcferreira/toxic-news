#!/usr/bin/env sh
# check-clean.sh <vcs> <context>
#
# Exits 0 iff the working copy has no uncommitted changes (git: `git status
# --porcelain` is empty, untracked non-ignored files included; jj: `jj diff
# -r @ --summary` is empty — jj snapshots every non-ignored file into @, so
# that is the same set). Otherwise prints <context> and the dirty paths to
# stderr and exits 1.
#
# fix-push.sh commits *everything* in the working copy as the round's fix,
# so the tree must be clean both where a round starts (prepare-review.sh)
# and right after the round's reviewers ran (the `reviewers_left_tree_clean`
# step): ai_review_custom runs an arbitrary user-typed command in the
# working copy, and any cache, report or reformatted file it leaves behind
# would otherwise be pushed to the PR as part of "review round N".
set -eu

vcs="${1:?check-clean.sh: vcs argument required}"
context="${2:?check-clean.sh: context argument required}"

case "$vcs" in
  jj) dirty=$(jj diff -r @ --summary) ;;
  git) dirty=$(git status --porcelain) ;;
  *)
    echo "check-clean.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
    exit 1
    ;;
esac

if [ -n "$dirty" ]; then
  {
    printf '%s\n' "$context"
    printf '%s\n' "$dirty"
  } >&2
  exit 1
fi
