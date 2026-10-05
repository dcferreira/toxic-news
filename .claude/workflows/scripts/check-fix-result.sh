#!/usr/bin/env sh
# check-fix-result.sh <vcs> <findings> <fix_input_file> <ci_round> <extra_files>
#
# fix_issues' postcondition. fix_issues is an agentic step: nothing here
# stops the fixer from claiming things that aren't true, so this
# independently re-checks its return against reality instead of trusting
# it (format-spec.md §B.8's "postconditions that re-observe reality"):
#
#   (i)   every returned item's verify_status matches the declared prefix
#         grammar (^(fixed|partially-fixed|not-fixed|wont-fix):) — the
#         original check;
#   (ii)  every INPUT item's `id` (review-route.sh/take-optional.sh/
#         ci-failures.sh put one on every finding routed to fix_issues) is
#         still present in the OUTPUT — a fixer that returns fewer items,
#         or `[]`, cannot silently make findings disappear, and cannot
#         rename/renumber an item's id either;
#   (iii) if the working copy has no uncommitted changes, no item may
#         claim `fixed:`/`partially-fixed:` — on jj that's an empty
#         `jj diff -r @ --summary`, on git an empty `git status
#         --porcelain`. Those claims require a change that didn't happen;
#         the fixer must say `wont-fix: already resolved` (or another
#         wont-fix reason) or `not-fixed:` instead. This is what makes
#         unchanged_route safe to treat "declines only" as "nothing left
#         unresolved" (see unchanged-route.sh).
#   (iv)  scope (E, "scope the fixer"): unless <ci_round> is "true" (a
#         CI-originated round's findings don't name real repo paths — their
#         `file` is "CI: <check name>" — so there is nothing to scope
#         against), every path the working copy actually changed (jj:
#         `jj diff --name-only`; git: `git status --porcelain`, including
#         untracked files) must be either the `file` of one of THIS round's
#         input findings, or named in <extra_files> (a JSON array of
#         {file, reason} the fixer returns for a file none of the findings
#         named but the fix genuinely needed). A changed path matching
#         neither fails the step — the fixer is re-prompted (attempts: 3)
#         naming the offending path(s), same as any other failure here.
#
# <findings> is fix_issues' returned array, taken as an ordinary ${key}
# argument like the postcondition it replaces (fix_issues' own output,
# already bounded by the same 128 KiB argv limit every other consumer of
# `findings` lives with — ci-failures.sh keeps it under a documented
# budget). <fix_input_file> is NOT re-passed as an argument: it is the
# path (state key `fix_input_file`, created by fetch-pr.sh, kept current by
# whichever step sets `findings` right before fix_issues — review-route.sh,
# take-optional.sh, ci-failures.sh) to a file holding a snapshot of exactly
# what fix_issues was asked to fix. Putting a second full copy of the same
# findings on this command's argv (input AND output) could overflow the
# 128 KiB argv/env cap on its own — CI findings can already be up to ~64 KiB
# (ci-failures.sh's PAWL_CI_INLINE_BUDGET) — so the input side is read from
# a file instead, the same reason ledger_file/record-dispositions.sh use one.
#
# On failure, prints a human-readable reason to STDOUT (the engine parses a
# postcondition's failure text from stdout first, falling back to stderr
# only if stdout was empty — internal/engine/postcondition.go) naming the
# offending items, since that text is what the next of fix_issues' 3
# attempts sees.
set -eu

vcs="${1:?check-fix-result.sh: vcs argument required}"
findings="${2:?check-fix-result.sh: findings argument required}"
fix_input_file="${3:?check-fix-result.sh: fix_input_file argument required}"
ci_round="${4:?check-fix-result.sh: ci_round argument required}"
extra_files="${5:?check-fix-result.sh: extra_files argument required}"

if [ ! -e "$fix_input_file" ]; then
  echo "check-fix-result.sh: snapshot file ${fix_input_file} does not exist (fix_issues' preceding step should have written it)" >&2
  exit 1
fi

fail=0
reasons=""

add_reason() {
  reasons="${reasons}${reasons:+$(printf '\n')}$1"
}

if ! printf '%s' "$findings" | jq -e 'type == "array"' >/dev/null 2>&1; then
  echo "check-fix-result.sh: fix_issues returned findings that are not a JSON array"
  exit 1
fi

# (i) verify_status prefix grammar.
bad_prefix=$(printf '%s' "$findings" | jq -r '
  [.[] | select(((.verify_status // "") | test("^(fixed|partially-fixed|not-fixed|wont-fix):")) | not)]
  | map((.file // "?") + "/" + (.category // "?") + ": " + (.description // "?") + " (verify_status: " + (.verify_status // "<missing>") + ")")
  | .[]')
if [ -n "$bad_prefix" ]; then
  fail=1
  add_reason "$(printf 'item(s) with a verify_status that does not start with fixed:/partially-fixed:/not-fixed:/wont-fix: :\n%s' "$bad_prefix")"
fi

# (ii) every input item's id is still present in the output.
dropped=$(jq -r -n --slurpfile input "$fix_input_file" --argjson output "$findings" '
  ($output | map(.id)) as $out_ids
  | ($input[0] // []) | map(select((.id as $i | $out_ids | any(. == $i)) | not))
  | map((.id // "?") + " (" + (.file // "?") + "/" + (.category // "?") + ": " + (.description // "?") + ")")
  | .[]')
if [ -n "$dropped" ]; then
  fail=1
  add_reason "$(printf 'input item(s) missing from the returned findings, by id (nothing may be dropped, even to [], and an id may not be changed):\n%s' "$dropped")"
fi

# (iii) an unchanged tree cannot have any fixed:/partially-fixed: item.
case "$vcs" in
  jj) dirty=$(jj diff -r @ --summary) ;;
  git) dirty=$(git status --porcelain) ;;
  *)
    echo "check-fix-result.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
    exit 1
    ;;
esac

if [ -z "$dirty" ]; then
  false_claims=$(printf '%s' "$findings" | jq -r '
    [.[] | select((.verify_status // "") | test("^(fixed|partially-fixed):"))]
    | map((.file // "?") + "/" + (.category // "?") + ": " + (.description // "?") + " (verify_status: " + .verify_status + ")")
    | .[]')
  if [ -n "$false_claims" ]; then
    fail=1
    add_reason "$(printf 'the working copy has NO uncommitted changes, so these item(s) cannot be fixed:/partially-fixed: (use wont-fix: already resolved, another wont-fix reason, or not-fixed: instead):\n%s' "$false_claims")"
  fi
fi

# (iv) scope: every changed path must be named by an input finding or
# extra_files, unless this is a CI-originated round (whose findings' `file`
# is a synthetic "CI: <check>" label, not a repo path).
if [ "$ci_round" != "true" ]; then
  case "$vcs" in
    jj) changed=$(jj diff --name-only) ;;
    git)
      changed=$(git status --porcelain | sed -E 's/^.. //; s/.* -> //')
      ;;
    *)
      echo "check-fix-result.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
      exit 1
      ;;
  esac
  if [ -n "$changed" ]; then
    offenders=$(jq -R -r -s --slurpfile input "$fix_input_file" --argjson extra "$extra_files" '
      (($input[0] // []) | map(.file)) as $named
      | (($extra // []) | map(.file)) as $extra_named
      | ($named + $extra_named) as $allowed
      | split("\n") | map(select(length > 0))
      | map(select((. as $p | $allowed | any(. == $p)) | not))
      | .[]' <<EOF
$changed
EOF
)
    if [ -n "$offenders" ]; then
      fail=1
      add_reason "$(printf 'path(s) changed that no input finding names and extra_files does not list (fix only the listed items, minimally; list any other file genuinely needed in extra_files: {file, reason}):\n%s' "$offenders")"
    fi
  fi
fi

if [ "$fail" -ne 0 ]; then
  printf '%s\n' "$reasons"
  exit 1
fi
