#!/usr/bin/env sh
# take-optional.sh <fix_input_file> <optional_findings> <fix_note>
#
# Body of the `take_optional` deterministic step, reached when ask_optional's
# answer was "address": the held findings (the `optional_findings` state
# key — review-route.sh's rendered view of the ledger's "held" entries, each
# already carrying its ledger `id`) become the findings fix_issues fixes.
# Turns <optional_findings> into `{findings: ..., optional_findings: []}`
# (fix_issues fixes them now; the held list is emptied here — record-
# dispositions.sh updates each item's ledger entry by `id` once fix_issues
# returns, same as any other round) and, like review-route.sh's `blocking`
# branch, overwrites <fix_input_file> (atomically) with the same array: a
# snapshot of exactly what fix_issues is about to be asked to fix, read back
# by check-fix-result.sh (fix_issues' postcondition) to confirm nothing
# given to the fixer gets silently dropped from its returned findings.
#
# <fix_note> is ask_optional's `fix_note` write. Per design/format-spec.md
# §B.5, a plain static-option pick (no `chosen:` route reached) still
# writes that option's own label into the declared `writes:` key — so
# picking the "address" option literally writes fix_note="address", not an
# instruction. fix_issues renders fix_note verbatim as "Instructions from
# the user for this round", so left alone that leaks the option label in as
# if it were a user instruction. Deterministically strip that one case: if
# <fix_note> is exactly the literal string "address" (the option label),
# it's rewritten to "" (no instructions given, just "yes, address them");
# anything else — including the empty string from a submit-time bug, or
# any free-text "Other" answer, however it reads — passes through
# unchanged, since only the literal option label is ambiguous.
#
# Per the top-of-file round-cap comment: this is what keeps "address a held
# nit" from ever re-triggering a full re-review — the round that follows
# fix_push re-enters prepare_review with `last_reviewed_head` already
# advanced past this head, so it comes back as a delta round, scoped to
# just this fix, never a fresh full review.
#
# Prints {"findings": ..., "optional_findings": [], "fix_note": ...} on one
# line (emits: json).
set -eu

fix_input_file="${1:?take-optional.sh: fix_input_file argument required}"
optional_findings="${2:?take-optional.sh: optional_findings argument required}"
# Checked by argument count, not ${3:?}: an empty fix_note is a valid value (see above) and
# must not abort the step; only a missing argument is an error.
[ "$#" -ge 3 ] || { echo "take-optional.sh: fix_note argument required" >&2; exit 1; }
fix_note="$3"

printf '%s' "$optional_findings" | jq -c '.' >"${fix_input_file}.tmp"
mv "${fix_input_file}.tmp" "$fix_input_file"

if [ "$fix_note" = "address" ]; then
  fix_note=""
fi

jq -cn --argjson n "$optional_findings" --arg fix_note "$fix_note" \
  '{findings: $n, optional_findings: [], fix_note: $fix_note}'
