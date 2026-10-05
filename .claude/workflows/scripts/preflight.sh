#!/usr/bin/env sh
# preflight.sh <title> <head_branch> <base>
#
# Body of the `preflight` deterministic step: refuses to start a PR that
# could not be raised cleanly, before anything is pushed.
#
# 1. The title is a Conventional Commits subject (check-title.sh).
# 2. The head branch is not the base branch — open-pr.sh pushes the local
#    head to <head_branch>, and pushing a change straight onto main is
#    exactly what a PR exists to avoid.
# 3. The working copy is clean (check-clean.sh). The local head (`HEAD`
#    under git, `@-` under jj — commit your work with `jj commit` first, so
#    @ is empty) is what gets pushed; an uncommitted edit would silently be
#    left out of the PR.
# 4. The local head has commits that origin's <base> does not, so there is
#    something to merge.
#
# Prints {"vcs": "jj"|"git"} on one line (emits: json).
set -eu

title="${1-}"
head_branch="${2:?preflight.sh: head_branch argument required}"
base="${3:?preflight.sh: base argument required}"
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

"$script_dir/check-title.sh" "$title"

if [ "$head_branch" = "$base" ]; then
  echo "preflight.sh: head branch '${head_branch}' is the base branch; push your change to its own branch" >&2
  exit 1
fi

vcs=$("$script_dir/detect-vcs.sh")
"$script_dir/check-clean.sh" "$vcs" "preflight.sh: the working copy has uncommitted changes; commit them (jj commit / git commit) or drop them first:"

case "$vcs" in
  jj)
    jj git fetch --remote origin --branch "exact:\"${base}\"" >&2
    ahead=$(jj log --no-graph -r "\"${base}\"@origin..@-" -T 'commit_id ++ "\n"')
    ;;
  git)
    git fetch --quiet origin "refs/heads/${base}"
    ahead=$(git rev-list FETCH_HEAD..HEAD)
    ;;
esac

if [ -z "$ahead" ]; then
  echo "preflight.sh: the local head has no commits that origin's ${base} lacks; nothing to raise a PR for" >&2
  exit 1
fi

jq -cn --arg vcs "$vcs" '{vcs: $vcs}'
