#!/usr/bin/env sh
# verify-pushed.sh <vcs> <branch> <repo>
#
# `fix_push`'s postcondition: re-observes reality instead of trusting the
# push's exit code — exits 0 iff the PR branch's head on GitHub is the local
# head (`HEAD` under git, `@-` under jj, the same convention
# prepare-review.sh uses). Holds after a push and after an `unchanged` round
# alike. Uses GitHub's git-ref API rather than `gh pr view`'s headRefOid,
# which can lag a push by a moment.
set -eu

vcs="${1:?verify-pushed.sh: vcs argument required}"
branch="${2:?verify-pushed.sh: branch argument required}"
repo="${3:?verify-pushed.sh: repo argument required}"

# Resolved once by fetch-pr.sh from origin's URL. GH_REPO fills in the
# {owner}/{repo} placeholder below (and any other gh call here) instead of
# gh inferring a repo from cwd, which fails outright in a non-colocated jj
# workspace (no .git directory to infer from).
export GH_REPO="$repo"

case "$vcs" in
  jj) local_head=$(jj log --no-graph -r @- -T commit_id) ;;
  git) local_head=$(git rev-parse HEAD) ;;
  *)
    echo "verify-pushed.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
    exit 1
    ;;
esac

remote_head=$(gh api "repos/{owner}/{repo}/git/ref/heads/${branch}" --jq '.object.sha')

if [ "$remote_head" != "$local_head" ]; then
  echo "verify-pushed.sh: origin's ${branch} is ${remote_head}, local head is ${local_head}" >&2
  exit 1
fi
