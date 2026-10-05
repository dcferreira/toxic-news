#!/usr/bin/env sh
# unchanged-route.sh <findings> <ci_round> <optional_findings>
#
# Body of the `unchanged_route` deterministic step, reached when fix_push
# printed `unchanged`: fix_issues made no change to the tree. That's
# expected when this round's only non-fixes were declines (wont-fix), but
# retrying on an unchanged tree is pointless whenever something the fixer
# could not actually change is still outstanding.
#
# <findings> is fix_issues' returned findings (each item has a
# verify_status). <ci_round> is the `ci_round` state key: true iff this
# round's findings came from ci_failure (a failing CI check), false if they
# came from review_route (reviewer findings) — used only in the stderr
# message below now, not in the `remaining` computation (see next
# paragraph).
#
# `remaining` is every finding whose verify_status does NOT start with
# "wont-fix" — on ANY round, review or CI alike. This used to also treat a
# `fixed:`/`partially-fixed:` claim on a review round as already resolved
# (a held optional item an earlier fix happened to fix, say). That
# distinction is gone: check-fix-result.sh (fix_issues' postcondition) now
# refuses `fixed:`/`partially-fixed:` outright whenever the working copy
# comes out of fix_issues unchanged, so such a claim can no longer reach
# this step on an unchanged tree at all — every non-wont-fix item here is
# genuinely not-fixed/partially-fixed, review round or CI round. One rule
# now covers both.
#
# If any remain, no amount of re-review will change anything the fixer
# couldn't already do: stuck. If none remain and this was a CI round, CI on
# this head will stay red forever (the failures were all declined, not
# fixed): also stuck, for a human to look at. Otherwise (a review round
# with nothing left unresolved — every item was declined): the tree is
# fine to send on — except there may be held minor/nitpick findings
# (<optional_findings>, the `optional_findings` state key) that were never
# asked about, because a blocking round never reaches ask_optional. Sending
# straight to `to_ci` (-> wait_for_ci -> done on green) would silently drop
# them, exactly as it used to before this check existed. So: non-empty
# <optional_findings> routes to `ask` (-> ask_optional) instead of `to_ci`,
# same question ask_optional always asks, just reached from this side of
# the graph too. This stays bounded: both `to_ci` and `ask` eventually lead
# back through ci_failure (setting ci_round true) or fix_issues, so a
# second `unchanged` in the same round is `stuck` either way.
#
# Prints just the TOKEN (to_ci|ask|stuck) on the last stdout line — no
# payload, nothing to write (emits: json, the default, but a TOKEN-only
# line is valid under it per format-spec §B.1). A one-line human-readable
# reason goes to stderr for the logs.
set -eu

findings="${1:?unchanged-route.sh: findings argument required}"
ci_round="${2:?unchanged-route.sh: ci_round argument required}"
optional_findings="${3:?unchanged-route.sh: optional_findings argument required}"

remaining=$(printf '%s' "$findings" | jq -c '[.[] | select((.verify_status // "") | startswith("wont-fix") | not)]')
remaining_count=$(printf '%s' "$remaining" | jq 'length')
optional_count=$(printf '%s' "$optional_findings" | jq 'length')

if [ "$remaining_count" -gt 0 ]; then
  echo "unchanged-route.sh: ${remaining_count} finding(s) left unresolved and the tree didn't change — stuck" >&2
  echo stuck
elif [ "$ci_round" = "true" ]; then
  echo "unchanged-route.sh: all findings were declined but this was a CI-originated round — CI on this head will stay red, stuck" >&2
  echo stuck
elif [ "$optional_count" -gt 0 ]; then
  echo "unchanged-route.sh: nothing left unresolved (all declined) on a review round, but ${optional_count} held minor/nitpick finding(s) were never asked about — ask" >&2
  echo ask
else
  echo "unchanged-route.sh: nothing left unresolved (all declined) on a review round — on to CI" >&2
  echo to_ci
fi
