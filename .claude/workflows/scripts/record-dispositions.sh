#!/usr/bin/env sh
# record-dispositions.sh <ledger_file> <findings>
#
# Body of the `record_dispositions` deterministic step (formerly
# `record_declined`), run right after fix_issues and before fix_push:
# updates each finding's ledger entry (matched by `id` — every finding
# routed to fix_issues, from review-route.sh, take-optional.sh or
# ci-failures.sh alike, carries its ledger `id`) from fix_issues' returned
# `verify_status`:
#   verify_status prefix   ->  ledger status
#   fixed:                 ->  fixed
#   partially-fixed:       ->  partially-fixed
#   not-fixed:              ->  not-fixed
#   wont-fix:              ->  declined
# `reason` is set to the full `verify_status` string in every case (not just
# the wont-fix reason) — a "fixed:"/"not-fixed:" report can carry useful
# context too, and keeping the whole string is simpler than re-parsing it
# apart from its prefix.
#
# <findings> is fix_issues' output: this round's findings, each with the
# added `verify_status` check-fix-result.sh already validated. An id in
# <findings> that isn't in the ledger is left alone (should not happen —
# check-fix-result.sh's rule (ii) already requires every INPUT id to
# reappear in the output; this only guards a mismatch some future change
# introduces).
#
# The updated ledger replaces <ledger_file> atomically (temp file + mv).
# Idempotent on a re-run: applying the same verify_status update twice
# yields the same ledger. Prints nothing to stdout: the step writes no
# state key.
set -eu

ledger_file="${1:?record-dispositions.sh: ledger_file argument required}"
findings="${2:?record-dispositions.sh: findings argument required}"

jq -cn --slurpfile ledger "$ledger_file" --argjson findings "$findings" '
  def status_of(vs):
    if (vs | startswith("fixed:")) then "fixed"
    elif (vs | startswith("partially-fixed:")) then "partially-fixed"
    elif (vs | startswith("not-fixed:")) then "not-fixed"
    elif (vs | startswith("wont-fix:")) then "declined"
    else "not-fixed"
    end;
  ($findings | map(select(.id != null)) | map({key: .id, value: .}) | from_entries) as $by_id
  | ($ledger[0] // []) | map(
      .id as $iid
      | if ($by_id | has($iid))
      then . + {status: status_of($by_id[$iid].verify_status), reason: $by_id[$iid].verify_status}
      else .
      end
    )
' >"${ledger_file}.tmp"
mv "${ledger_file}.tmp" "$ledger_file"
