#!/usr/bin/env sh
# skip-optional.sh <ledger_file> <optional_findings>
#
# Body of the `skip_optional` deterministic step, reached when ask_optional's
# answer was "skip": the user chose not to address the held findings. Unlike
# the old declined-list design, there is nothing to append: <optional_findings>
# (review-route.sh's rendered view of the ledger's "held" entries) already
# names each item's ledger `id` — this just flips those SAME ledger entries'
# `status` to "skipped" (with a reason), in place, by id.
#
# Written atomically (temp file + mv); idempotent on a re-run (setting an
# already-"skipped" entry to "skipped" again is a no-op).
#
# Prints {"optional_findings": []} on one line (emits: json, the default):
# the held findings are consumed, so review-route.sh does not carry them
# into the next round (they stay in the ledger forever as "skipped", which
# is itself one of the dedup-triggering statuses — see review-route.sh — so
# a reviewer re-raising the same thing without `reraise_of` is dropped).
set -eu

ledger_file="${1:?skip-optional.sh: ledger_file argument required}"
optional_findings="${2:?skip-optional.sh: optional_findings argument required}"

jq -cn --slurpfile ledger "$ledger_file" --argjson optional "$optional_findings" '
  ($optional | map(.id)) as $skipped_ids
  | ($ledger[0] // []) | map(
      .id as $iid
      | if ($skipped_ids | index($iid)) != null
      then . + {status: "skipped", reason: "skipped by the user"}
      else .
      end
    )
' >"${ledger_file}.tmp"
mv "${ledger_file}.tmp" "$ledger_file"

printf '%s\n' '{"optional_findings": []}'
