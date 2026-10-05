#!/usr/bin/env sh
# ci-failures.sh <pr_number> <head_sha> <repo> <fix_input_file> <ledger_file> <round>
#
# Body of the `ci_failure` deterministic step, reached when wait_for_ci saw a
# failing check. Turns each failing check into a finding for fix_issues —
# name, link and, for a GitHub Actions job, the tail of its failed-step log
# (`gh run view --job <id> --log-failed`) — so the fixer is told what
# actually broke instead of being sent back to diff-only reviewers. Every
# finding (including the fallback one below) carries severity: "major" — a
# realistic failure mode that breaks the run, per the severity scale
# review-route.sh's findings share (critical > major > medium > minor >
# nitpick) — never a nitpick.
#
# Also overwrites <fix_input_file> (atomically) with the same findings
# array — a snapshot of exactly what fix_issues is about to be asked to
# fix, read back by check-fix-result.sh (fix_issues' postcondition) to
# confirm nothing given to the fixer gets silently dropped from its
# returned findings (see fetch-pr.sh, which creates the file, for why it's
# a file and not another state key).
#
# Each finding is also appended to <ledger_file> (findings ledger — see
# fetch-pr.sh) as a `status: "open"` entry with an id `r<round>-ci<n>` — the
# `ci` infix keeps it from ever colliding with review-route.sh's plain
# `r<round>-<n>` ids for the same round number (fix_push's `unchanged`
# outcome can send a round back through wait_for_ci/ci_failure without
# bumping `round`, so ci_failure can run more than once at the same round
# number; <n> continues from however many `r<round>-ci*` ids are already in
# the ledger, so a second ci_failure in the same round still gets fresh
# ids). record-dispositions.sh later updates these same entries by id, same
# as any other finding's.
#
# Prints {"findings": [...], "ci_round": true, "fix_note": ""} on one line
# (emits: json; `jq -c` keeps it on the one line the engine parses).
# `ci_round: true` marks this round as CI-originated, so unchanged_route can
# tell (if fix_push later reports the tree unchanged) that a decline here
# means CI on this head will stay red, not that a re-review is worth another
# try. `fix_note: ""` clears any instructions the user gave ask_optional
# (its picked label, e.g. "skip", or free text): those were about the held
# optional findings, not these CI failures, and fix_issues would otherwise
# apply them to this round. This step is the single choke point every
# CI-originated fix round passes through.
# If the failing checks can't be listed any more (e.g. re-run in the
# meantime) it still emits one finding saying so, so a CI failure never
# turns into an empty fix round.
#
# Tied to <head_sha>, like poll-ci.sh: if the PR's head (headRefOid) is no
# longer <head_sha> — someone pushed between wait_for_ci's FAILURE and this
# step — the rollup describes another commit's checks, so the step fails
# (non-zero exit -> blocked) instead of labelling them as <head_sha>'s.
#
# Size: the findings printed here become the `findings` state key, which
# later deterministic steps (record_dispositions, unchanged_route) and
# fix_issues' postcondition get as one shell-quoted word inside their
# `sh -c` argument, and as PAWL_FINDINGS — each capped at 128 KiB
# (MAX_ARG_STRLEN) on Linux. So the failed-step log tail (last
# PAWL_CI_LOG_LINES lines, default 200) goes to a file under the user cache
# dir, whose path the finding names for the fixer to read; the finding
# itself carries only the last PAWL_CI_INLINE_BYTES bytes of it (default
# 2048). Once the findings reach PAWL_CI_INLINE_BUDGET bytes in total
# (default 65536; measured as the engine will pass them — its JSON encoder
# writes each < > & as a 6-byte \u00XX escape and shell-quoting turns each
# ' into 4 bytes, so those are counted at that size), every further failing check is folded into one last
# finding that just names them (at most 4 KiB of names), so however many
# checks fail the output stays well under the limit.
set -eu

pr_number="${1:?ci-failures.sh: pr_number argument required}"
head_sha="${2:?ci-failures.sh: head_sha argument required}"
repo="${3:?ci-failures.sh: repo argument required}"
fix_input_file="${4:?ci-failures.sh: fix_input_file argument required}"
ledger_file="${5:?ci-failures.sh: ledger_file argument required}"
round="${6:?ci-failures.sh: round argument required}"
log_lines="${PAWL_CI_LOG_LINES:-200}"
inline_bytes="${PAWL_CI_INLINE_BYTES:-2048}"
inline_budget="${PAWL_CI_INLINE_BUDGET:-65536}"
cache_dir="${XDG_CACHE_HOME:-${HOME}/.cache}/pawl-review-pr"
mkdir -p "$cache_dir"

# Resolved once by fetch-pr.sh from origin's URL; used here (and by the
# `gh run view` call below) instead of letting gh infer a repo from cwd,
# which fails outright in a non-colocated jj workspace (no .git directory).
export GH_REPO="$repo"

rollup=$(gh pr view "$pr_number" --json headRefOid,statusCheckRollup)

pr_head=$(printf '%s' "$rollup" | jq -r '.headRefOid // ""')
if [ "$pr_head" != "$head_sha" ]; then
  echo "ci-failures.sh: PR #${pr_number}'s head is now ${pr_head:-<unknown>}, not ${head_sha} (the head wait_for_ci saw fail) — its checks belong to another commit; refusing to report them as ${head_sha}'s" >&2
  exit 1
fi

failed=$(printf '%s' "$rollup" | jq -c '
  (.statusCheckRollup // [])[]
  | if (.__typename == "StatusContext") or (has("state") and (has("status") | not)) then
      select(.state != "SUCCESS" and .state != "PENDING" and .state != "EXPECTED")
      | {name: (.context // "status"), url: (.targetUrl // "")}
    else
      select(.status == "COMPLETED"
             and .conclusion != "SUCCESS" and .conclusion != "NEUTRAL" and .conclusion != "SKIPPED")
      | {name: ((.workflowName // "") + (if .workflowName then " / " else "" end) + (.name // "check")),
         url: (.detailsUrl // "")}
    end')

# make_item <inline log text>: one finding for the check in $name/$url,
# naming $log_file (if any) as the full log tail.
make_item() {
  jq -cn --arg name "$name" --arg url "$url" --arg sha "$head_sha" --arg log "$1" --arg log_file "$log_file" '{
    file: ("CI: " + $name),
    line: 0,
    category: "bug",
    description: ("CI check \"" + $name + "\" failed on " + $sha
                  + (if $url != "" then " (" + $url + ")" else "" end)
                  + (if $log_file != "" then ". Full failed-step log tail: " + $log_file else "" end)
                  + ". Failed-step log excerpt:\n" + $log),
    fix: "Reproduce the failure locally and change the code (or the test, if the test is wrong) so this check passes.",
    severity: "major"
  }'
}

# encoded_size <json>: its byte size once the engine has re-encoded it
# (Go's json.Marshal escapes < > & as \u003c etc.) and shell-quoted it
# (' -> '\''), counting both worst cases.
encoded_size() {
  bytes=$(printf '%s' "$1" | wc -c)
  html=$(printf '%s' "$1" | tr -cd '<>&' | wc -c)
  quotes=$(printf '%s' "$1" | tr -cd "'" | wc -c)
  echo $((bytes + 5 * html + 3 * quotes))
}

items=""
used=0
overflow=""
overflow_n=0
nl='
'
while IFS= read -r check; do
  [ -n "$check" ] || continue
  name=$(printf '%s' "$check" | jq -r '.name')
  url=$(printf '%s' "$check" | jq -r '.url')
  job_id=$(printf '%s' "$url" | sed -n 's#.*/actions/runs/[0-9][0-9]*/job/\([0-9][0-9]*\).*#\1#p')
  log_file=""
  if [ -n "$job_id" ] && log=$(gh run view --job "$job_id" --log-failed 2>&1); then
    log_file="${cache_dir}/ci-${pr_number}-${head_sha}-${job_id}.log"
    printf '%s\n' "$log" | tail -n "$log_lines" >"$log_file"
    log=$(tail -c "$inline_bytes" "$log_file")
  elif [ -n "$job_id" ]; then
    log="(could not fetch the job log: $(printf '%s' "$log" | tail -n 5 | tail -c "$inline_bytes"))"
  else
    log="(no GitHub Actions job log available — open the check's link)"
  fi
  item=$(make_item "$log")
  size=$(encoded_size "$item")
  if [ $((used + size)) -gt "$inline_budget" ]; then
    overflow_n=$((overflow_n + 1))
    overflow="${overflow}${overflow:+, }${name}${log_file:+ (log: ${log_file})}"
    continue
  fi
  used=$((used + size))
  items="${items}${item}${nl}"
done <<EOF
$failed
EOF

if [ "$overflow_n" -gt 0 ]; then
  overflow=$(printf '%s' "$overflow" | head -c 4096)
  item=$(jq -cn --arg sha "$head_sha" --arg pr "$pr_number" --arg n "$overflow_n" --arg names "$overflow" '{
    file: "CI",
    line: 0,
    category: "bug",
    description: ($n + " more CI checks failed on " + $sha + " (listed without log excerpts to keep the findings small; the list may be cut short): " + $names),
    fix: ("Inspect `gh pr checks " + $pr + "` and the log files named, and fix whatever is failing."),
    severity: "major"
  }')
  items="${items}${item}${nl}"
fi

if [ -z "$items" ]; then
  items=$(jq -cn --arg sha "$head_sha" --arg pr "$pr_number" '{
    file: "CI",
    line: 0,
    category: "bug",
    description: ("CI reported a failure on " + $sha + " but no failing check could be listed any more (re-run or removed?)."),
    fix: ("Inspect `gh pr checks " + $pr + "`, fix whatever is failing, or leave the tree unchanged if CI is actually green."),
    severity: "major"
  }')
fi

findings=$(printf '%s\n' "$items" | jq -cs '.')

# Assign ledger ids, continuing the r<round>-ci<n> sequence from whatever is
# already in the ledger for this round (see the top-of-file comment).
existing_ci=$(jq -c --arg prefix "r${round}-ci" '[.[] | select((.id // "") | startswith($prefix))] | length' "$ledger_file")
findings=$(printf '%s' "$findings" | jq -c --argjson round "$round" --argjson start "$existing_ci" '
  to_entries | map(.value + {
    id: ("r" + ($round | tostring) + "-ci" + ((.key + 1 + $start) | tostring)),
    round: $round,
    pre_existing: false
  })')

printf '%s' "$findings" >"${fix_input_file}.tmp"
mv "${fix_input_file}.tmp" "$fix_input_file"

jq -cn --slurpfile ledger "$ledger_file" --argjson findings "$findings" '
  ($ledger[0] // []) + ($findings | map(. + {status: "open", reason: null}))
' >"${ledger_file}.tmp"
mv "${ledger_file}.tmp" "$ledger_file"

printf '%s' "$findings" | jq -c '{findings: ., ci_round: true, fix_note: ""}'
