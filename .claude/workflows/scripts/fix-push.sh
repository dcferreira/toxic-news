#!/usr/bin/env sh
# fix-push.sh <vcs> <branch> <round>
#
# Body of the `fix_push` deterministic step: commits whatever fix_issues
# left in the working copy as "review round N" and pushes it to the PR's
# branch, under either VCS this repo (or any repo this workflow runs
# against) might use. prepare-review.sh has already checked, at the start
# of this round, that the working copy was clean and sat on the PR head,
# and the reviewers_left_tree_clean step checked (with the same
# check-clean.sh) that the round's reviewers left it clean too — so what
# this commits is fix_issues' work, not a reviewer's cache or report files.
# (A ci_failure round reaches here without re-running reviewers; nothing
# but fix_issues touched the tree since the last clean check.)
#
# Fast-forward only: the PR branch's remote head is fetched first (under
# both VCSes, so a push someone else made since this round started is seen
# before anything is committed locally), and the new commit must descend
# from it, or the step fails without committing or pushing. This is what
# keeps a run from rewriting the PR branch with unrelated history.
#
# Idempotent across a failed/interrupted push (format-spec §B.8 — the
# engine re-runs the step on resume): with a clean working copy the step
# compares the local head (git: HEAD; jj: @-) with the freshly fetched
# remote head. Equal: nothing to do, `unchanged` (fix_issues changed nothing
# in the tree — expected when this round's only non-fixes were declines,
# since declining a finding doesn't touch the tree. The workflow routes
# `unchanged` to unchanged_route, not straight back to review_route: if
# anything is left that the fixer didn't actually decline (a genuine
# not-fixed/partially-fixed item, or — on a CI-originated round — a
# declined CI failure, since CI on this head will stay red regardless),
# retrying is pointless and unchanged_route resolves `stuck`; only a review
# round whose only non-fixes were clean declines goes on to check CI).
# Local ahead of remote (a previous attempt committed the fix but the push
# never landed): skip the commit and just push it. Anything else: not a
# fast-forward, refuse.
#
# jj (including a colocated jj+git repo): `jj commit` finalizes the
# working-copy changes into a new commit and moves @ to a fresh empty one on
# top of it — the finished commit is then @-. Bookmarks do NOT follow @
# automatically past the first commit, so `branch` is explicitly re-pointed
# at @- every round before pushing — without --allow-backwards, so jj itself
# also refuses a backwards/sideways move.
#
# git: the familiar add/commit/push, pushing HEAD to the named branch
# explicitly rather than relying on the checkout already tracking it; a
# plain (non-force) push refuses a non-fast-forward on its own too.
#
# Prints `pushed round=<n+1>` or `unchanged` on its last line (emits: pairs;
# an integer has no whitespace to break that grammar).
set -eu

vcs="${1:?fix-push.sh: vcs argument required}"
branch="${2:?fix-push.sh: branch argument required}"
round="${3:?fix-push.sh: round argument required}"

next_round=$((round + 1))
msg="review round ${next_round}"

not_ff() {
  echo "fix-push.sh: the local change does not descend from origin's ${branch} — refusing to push (it would rewrite the PR branch)" >&2
  exit 1
}

case "$vcs" in
  jj)
    jj git fetch --remote origin --branch "exact:\"${branch}\"" >&2
    # A fetched remote bookmark is untracked by default
    # (git.auto-local-bookmark=false), and `jj git push --bookmark` refuses
    # to push while a non-tracking <branch>@origin exists. Track it before
    # anything is committed; already tracked, this is a harmless no-op.
    jj bookmark track "${branch}@origin" >&2 2>&1
    remote_rev="\"${branch}\"@origin"
    remote_head=$(jj log --no-graph -r "$remote_rev" -T commit_id)
    if [ -n "$(jj diff -r @ --summary)" ]; then
      if [ -z "$(jj log --no-graph -r "(${remote_rev}) & ::@" -T commit_id)" ]; then
        not_ff
      fi
      jj commit -m "$msg" >&2
    else
      local_head=$(jj log --no-graph -r @- -T commit_id)
      if [ "$local_head" = "$remote_head" ]; then
        echo unchanged
        exit 0
      fi
      # A previous attempt committed but did not push: push that commit.
      if [ -z "$(jj log --no-graph -r "(${remote_rev}) & ::@-" -T commit_id)" ]; then
        not_ff
      fi
    fi
    jj bookmark set "$branch" -r @- >&2
    jj git push --remote origin --bookmark "$branch" >&2
    ;;
  git)
    git fetch --quiet origin "refs/heads/${branch}"
    remote_head=$(git rev-parse FETCH_HEAD)
    git add -A
    if ! git diff --cached --quiet; then
      git merge-base --is-ancestor "$remote_head" HEAD || not_ff
      git commit --quiet -m "$msg"
    else
      if [ "$(git rev-parse HEAD)" = "$remote_head" ]; then
        echo unchanged
        exit 0
      fi
      # A previous attempt committed but did not push: push that commit.
      git merge-base --is-ancestor "$remote_head" HEAD || not_ff
    fi
    git push --quiet origin "HEAD:refs/heads/${branch}"
    ;;
  *)
    echo "fix-push.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
    exit 1
    ;;
esac

echo "pushed round=${next_round}"
