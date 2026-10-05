#!/usr/bin/env sh
# poll-ci.sh <pr_number> <head_sha> <allow_no_ci> <repo>
#
# `poll:` body for the `wait_for_ci` step: the CI verdict for exactly
# <head_sha> (the head this round reviewed), never for whatever GitHub last
# happened to report.
#
# Reads `gh pr view --json headRefOid,statusCheckRollup` and prints one of
# SUCCESS | FAILURE | PENDING:
#   - gh fails (auth, network, ...)          -> PENDING (logged to stderr)
#   - headRefOid != head_sha                 -> PENDING (GitHub hasn't caught
#     up with the push yet, or someone else pushed; the latter runs into
#     wait_for_ci's timeout -> blocked rather than passing silently)
#   - no checks registered for the head      -> PENDING, unless allow_no_ci is
#     "true" (a repo with no CI at all), then SUCCESS. Checks can take a
#     moment to appear after a push, so "none yet" is not "green".
#   - any check failed, errored, was cancelled, timed out, or needs action
#                                            -> FAILURE
#   - any check still queued/running         -> PENDING
#   - otherwise (success/neutral/skipped)    -> SUCCESS
#
# PENDING is not one of wait_for_ci's declared `outcomes:` — following the
# same convention as docs/examples/wait-for-build/scripts/check_build.sh, an
# unrouted token is simply logged and `pawl poll` re-runs this script on
# the next tick. Always exits 0: design/format-spec.md §B.1 makes a non-zero
# exit unconditionally `failure`, which would end the wait outright.
set -u

pr_number="${1:?poll-ci.sh: pr_number argument required}"
head_sha="${2:?poll-ci.sh: head_sha argument required}"
allow_no_ci="${3:-false}"
repo="${4:?poll-ci.sh: repo argument required}"

# Resolved once by fetch-pr.sh from origin's URL; used here instead of
# letting gh infer a repo from cwd (there is no git repo to infer from in a
# non-colocated jj workspace).
export GH_REPO="$repo"

if ! out=$(gh pr view "$pr_number" --json headRefOid,statusCheckRollup 2>&1); then
  echo "poll-ci.sh: gh pr view failed, will retry: ${out}" >&2
  echo PENDING
  exit 0
fi

verdict=$(printf '%s' "$out" | jq -r --arg sha "$head_sha" --arg allow "$allow_no_ci" '
  def bucket:
    if (.__typename == "StatusContext") or (has("state") and (has("status") | not)) then
      if .state == "SUCCESS" then "pass"
      elif .state == "PENDING" or .state == "EXPECTED" then "pending"
      else "fail" end
    else
      if .status != "COMPLETED" then "pending"
      elif .conclusion == "SUCCESS" or .conclusion == "NEUTRAL" or .conclusion == "SKIPPED" then "pass"
      else "fail" end
    end;
  if .headRefOid != $sha then "PENDING head-mismatch"
  else
    [(.statusCheckRollup // [])[] | bucket] as $b
    | if ($b | length) == 0 then
        (if $allow == "true" then "SUCCESS" else "PENDING no-checks" end)
      elif any($b[]; . == "fail") then "FAILURE"
      elif any($b[]; . == "pending") then "PENDING"
      else "SUCCESS" end
  end
' 2>/dev/null) || verdict="PENDING unparseable"
[ -n "$verdict" ] || verdict="PENDING unparseable"

case "$verdict" in
  "PENDING head-mismatch")
    echo "poll-ci.sh: PR #${pr_number}'s head is not ${head_sha} yet" >&2
    ;;
  "PENDING no-checks")
    echo "poll-ci.sh: no checks registered for ${head_sha} yet (pass allow_no_ci=true for a repo without CI)" >&2
    ;;
  "PENDING unparseable")
    echo "poll-ci.sh: could not parse gh output: ${out}" >&2
    ;;
esac

echo "${verdict%% *}"
exit 0
