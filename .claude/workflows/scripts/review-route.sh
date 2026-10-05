#!/usr/bin/env sh
# review-route.sh <findings_claude> <findings_docs> <ledger_file> <fix_input_file> <round> <review_pass> <delta_file> <blocking_history>
#
# Body of the `review_route` deterministic step: merges the two reviewer
# branches' findings arrays, assigns each an id, records every one of them
# into the findings ledger (<ledger_file> — see fetch-pr.sh for why it's a
# file, not a state key) with a disposition, and routes the round.
#
# ## Ledger and ids
#
# Every finding this step ever sees becomes a ledger entry:
#   {id, round, file, line, category, severity, description, fix,
#    pre_existing, status, reason}
# — plus `reraise_of` and `reraise_valid` (bool), carried through unchanged
# from the finding, whenever the finding named a `reraise_of` at all (see
# "Reraise validation" below). `id` is `r<round>-<n>`, assigned in this
# call. This step's own prior entries for THIS round (if it is being
# re-run after a crash — the engine
# re-runs a deterministic step on resume, format-spec §B.8) are dropped and
# recomputed from scratch before appending — `review_route` only ever runs
# once per round (a round only re-enters it via prepare_review, which bumps
# `round` first), so this makes the step idempotent without needing to
# persist any extra "already ran" marker.
#
# ## Dedup
#
# A fresh finding WITHOUT `reraise_of` is dropped as a duplicate of an
# existing ledger entry (from ANY earlier round) whose status is one of
# declined|skipped|held|suppressed|open, when both:
#   - same `file`, and both `line`s are numbers within +-5 of each other
#     (or exactly one `line` is null/missing/non-numeric and both entries
#     share the same `category` — a location-free finding otherwise never
#     matches a different category; both null still always matches);
#   - same `category`, OR the fresh finding is NO MORE severe than the
#     matched entry (an equal or higher rank on critical(0) > major(1) >
#     medium(2) > minor(3) > nitpick(4)). A fresh finding MORE severe than
#     anything nearby is never swallowed as a duplicate of a lesser one — a
#     new medium next to a held pre-existing minor must still block.
# A string `line` ("12-15", "L12") is read as its leading number.
# A finding with a *valid* `reraise_of` (see "Reraise validation" below)
# bypasses dedup entirely (always kept — it is already an explicit "this
# needs another look" claim from the reviewer, evidenced in its
# `description`). A finding whose `reraise_of` is invalid, or absent, gets
# no such exemption and is deduped normally. A `fixed` ledger entry is
# deliberately NOT one of the dedup-triggering statuses above: a finding
# at/near the same spot re-raised after something was marked fixed means the
# fix didn't hold, and must always get through.
#
# ## Reraise validation
#
# A finding's `reraise_of` (an earlier round's ledger `id`) only counts when
# it names an entry in <ledger_prev> (the ledger as of the START of this
# round — not including anything this call itself is about to add) whose
# `status` is one of fixed|partially-fixed|declined|skipped — the
# dispositions the reviewer prompts invite re-raising against. Anything
# else — an unknown/missing id, or an id whose entry has another status
# (e.g. held/suppressed/open/not-fixed) — is INVALID: the finding is
# treated exactly as if it had no `reraise_of` at all (normal dedup, normal
# delta-scope rules). Either way, when a finding names a `reraise_of` its
# ledger entry keeps the field and gains `reraise_valid: true`/`false`, so a
# human reading the ledger can see which reraises actually held up.
#
# ## Disposition
#
# For each surviving fresh finding, in order:
#   1. Delta-scope enforcement (only when <review_pass> is "delta" — see
#      prepare-review.sh): a finding whose severity is minor/nitpick, OR
#      (unless its category is "stale-docs", OR it carries a *valid*
#      `reraise_of` per "Reraise validation" above — docs_review flags
#      existing docs the delta made stale, which are rarely files the delta
#      touched, and a legitimately re-raised item shouldn't be re-suppressed
#      by scope just because the id it names happens to sit outside the
#      delta) whose `file` is not one of <delta_file>'s `+++ b/<path>`
#      paths, is NOT routed — it becomes a ledger entry with status
#      "suppressed" and a reason, and is otherwise dropped. This is the
#      deterministic backstop for the reviewers' own delta-pass
#      instructions (they are told the same rule in their `description:`);
#      a reviewer that disregards it doesn't get through anyway.
#   2. `pre_existing: true` -> status "held", regardless of severity: a
#      pre-existing finding is never blocking, in any pass.
#   3. Otherwise, severity "minor"/"nitpick" -> "held"; everything else
#      (critical/major/medium, and — fail-safe — missing/unrecognised
#      severity) -> "open" (this round's fixable findings).
#
# ## optional_findings
#
# The `optional_findings` state key is the rendered view of the ledger's
# "held" entries (severity-worst-first: critical > major > medium > minor >
# nitpick, so a pre_existing critical/major/medium item a reviewer held
# still surfaces above a fresh nitpick) — not something merged by hand here;
# the ledger IS the held list's source of truth across rounds. take-optional.sh
# and skip-optional.sh consume it by updating those same ledger entries'
# status, not by resetting a separately-tracked state array.
#
# ## blocking_history / stalled
#
# This step (and only this step — it runs once per review round, never on a
# CI-originated round) appends this round's fixable ("open") count to
# <blocking_history> (the `blocking_history` state key, a JSON array of
# integers). If the array now has >=3 entries, the last 3 are
# non-decreasing, and the first of them is >0 (growth from zero is not a
# stall), the round is routed `stalled`
# instead of `blocking` — fixable findings keep being raised but the count
# never drops, so another round won't help; a human needs to look
# (see review-pr.yaml's `stalled` terminal).
#
# ## Output
#
# On `blocking`, <fix_input_file> is overwritten (atomically) with the
# "open" findings — the snapshot check-fix-result.sh (fix_issues'
# postcondition) reads back to confirm nothing given to the fixer is
# silently dropped. Left untouched otherwise (fix_issues isn't reached
# either way on those outcomes; take-optional.sh writes it for the
# optional_only -> address path instead).
#
# Prints one of four routed lines (TOKEN + JSON payload, emits: json —
# `jq -c` keeps it on the one line the engine parses, format-spec §B.1),
# every payload carrying `blocking_history` (F: "make sure every emitted
# payload includes it"):
#   - blocking:      "blocking      {findings, optional_findings, blocking_history}"
#   - stalled:       "stalled       {findings, optional_findings, blocking_history}"
#   - optional_only: "optional_only {findings: [], optional_findings, blocking_history}"
#   - clean:         "clean         {findings: [], optional_findings: [], blocking_history}"
set -eu

findings_claude="${1:?review-route.sh: findings_claude argument required}"
findings_docs="${2:?review-route.sh: findings_docs argument required}"
ledger_file="${3:?review-route.sh: ledger_file argument required}"
fix_input_file="${4:?review-route.sh: fix_input_file argument required}"
round="${5:?review-route.sh: round argument required}"
review_pass="${6:?review-route.sh: review_pass argument required}"
delta_file="${7:?review-route.sh: delta_file argument required}"
blocking_history="${8:?review-route.sh: blocking_history argument required}"

# In a delta pass, the paths this round is actually scoped to: every
# "+++ b/<path>" target path in the delta diff (deleted files' "+++
# /dev/null" contribute nothing, correctly excluding them from scope).
if [ "$review_pass" = "delta" ]; then
  delta_paths=$(grep -o '^+++ b/.*' "$delta_file" 2>/dev/null | sed 's#^+++ b/##' | jq -R -s 'split("\n") | map(select(length > 0))')
else
  delta_paths='[]'
fi

result=$(jq -cn \
  --argjson a "$findings_claude" \
  --argjson d "$findings_docs" \
  --slurpfile ledger "$ledger_file" \
  --argjson round "$round" \
  --arg review_pass "$review_pass" \
  --argjson delta_paths "$delta_paths" \
  --argjson history "$blocking_history" \
  '
  def sevrank: {"critical":0,"major":1,"medium":2,"minor":3,"nitpick":4}[.severity // ""] // 2;
  def near(a; b): (a - b) as $diff | ($diff | if . < 0 then -. else . end) <= 5;
  def ln: if type == "number" then .
          elif type == "string" then ((capture("(?<n>[0-9]+)").n | tonumber) // null)
          else null end;
  def loc_match($f; $e):
    ($f.line | ln) as $fl | ($e.line | ln) as $el
    | ($f.file == $e.file) and (
      if ($fl == null) or ($el == null)
      then (($fl == null) and ($el == null)) or ($f.category == $e.category)
      else near($fl; $el) end
    );
  def cat_or_sev($f; $e):
    ($f.category == $e.category) or ($f | sevrank) >= ($e | sevrank);
  def is_dedup_target($e): (["declined","skipped","held","suppressed","open"] | index($e.status)) != null;
  def is_dup($f; $ledger_prev):
    $ledger_prev | any(.[]; is_dedup_target(.) and loc_match($f; .) and cat_or_sev($f; .));
  # A reraise_of only counts when it names a $ledger_prev entry (a finding
  # raised earlier this same call has no id yet, so a same-round reraise
  # is never valid) whose status invites re-raising.
  def reraise_ok($rid; $ledger_prev):
    ($rid != "") and ($ledger_prev | any(.[]; . as $e | $e.id == $rid and ((["fixed","partially-fixed","declined","skipped"] | index($e.status)) != null)));
  def has_valid_reraise($f): ($f.reraise_valid // false) == true;

  ($ledger[0] // []) as $ledger_all
  | ($ledger_all | map(select(.round != $round))) as $ledger_prev
  | (($a + $d) | map(. + {pre_existing: (.pre_existing // false)})) as $merged0
  | ($merged0 | map(
      (.reraise_of // "") as $rid
      | if $rid == "" then . else . + {reraise_valid: reraise_ok($rid; $ledger_prev)} end
    )) as $merged
  | ($merged | map(select(has_valid_reraise(.) or (is_dup(.; $ledger_prev) | not)))) as $kept
  | ($kept | to_entries | map(.value + {id: ("r" + ($round | tostring) + "-" + ((.key + 1) | tostring)), round: $round})) as $with_ids
  | ($with_ids | map(
      . as $it
      | if ($review_pass == "delta") and (($it.severity == "minor") or ($it.severity == "nitpick") or (($it.category != "stale-docs") and (has_valid_reraise($it) | not) and ($delta_paths | index($it.file) == null)))
      then . + {status: "suppressed", reason: ("delta pass: " + (if ($it.severity == "minor" or $it.severity == "nitpick") then "minor/nitpick findings are not routed in a delta pass" else "file is outside this delta (" + ($it.file // "?") + ")" end))}
      elif .pre_existing then . + {status: "held", reason: "pre-existing"}
      elif (.severity == "minor") or (.severity == "nitpick")
      then . + {status: "held", reason: null}
      else . + {status: "open", reason: null}
      end
    )) as $disposed
  | ($ledger_prev + $disposed) as $ledger_new
  | ($disposed | map(select(.status == "open"))) as $fixable
  | ($ledger_new | map(select(.status == "held")) | sort_by(sevrank)) as $optional
  | ($fixable | length) as $fixable_count
  | ($optional | length) as $optional_count
  | ($history + [$fixable_count]) as $history_new
  | ( ($history_new | length) >= 3
      and ($history_new[-3:] as $last3 | $last3[0] > 0 and $last3[0] <= $last3[1] and $last3[1] <= $last3[2])
      and ($history_new[-1] > 0)
    ) as $stalled
  | {
      ledger: $ledger_new,
      fixable: $fixable,
      optional: $optional,
      fixable_count: $fixable_count,
      optional_count: $optional_count,
      history: $history_new,
      stalled: $stalled
    }')

printf '%s' "$result" | jq -c '.ledger' >"${ledger_file}.tmp"
mv "${ledger_file}.tmp" "$ledger_file"

fixable_count=$(printf '%s' "$result" | jq '.fixable_count')
optional_count=$(printf '%s' "$result" | jq '.optional_count')
stalled=$(printf '%s' "$result" | jq -r '.stalled')

if [ "$fixable_count" -gt 0 ] && [ "$stalled" = "true" ]; then
  payload=$(printf '%s' "$result" | jq -c '{findings: [], optional_findings: .optional, blocking_history: .history}')
  printf 'stalled %s\n' "$payload"
elif [ "$fixable_count" -gt 0 ]; then
  printf '%s' "$result" | jq -c '.fixable' >"${fix_input_file}.tmp"
  mv "${fix_input_file}.tmp" "$fix_input_file"
  payload=$(printf '%s' "$result" | jq -c '{findings: .fixable, optional_findings: .optional, blocking_history: .history}')
  printf 'blocking %s\n' "$payload"
elif [ "$optional_count" -gt 0 ]; then
  payload=$(printf '%s' "$result" | jq -c '{findings: [], optional_findings: .optional, blocking_history: .history}')
  printf 'optional_only %s\n' "$payload"
else
  payload=$(printf '%s' "$result" | jq -c '{findings: [], optional_findings: [], blocking_history: .history}')
  printf 'clean %s\n' "$payload"
fi
