#!/usr/bin/env sh
# prepare-review.sh <vcs> <pr_number> <repo> <base_branch> <last_reviewed_head>
#
# Body of the `prepare_review` deterministic step, the entry point of every
# review round. It makes the round's inputs hard rather than best-effort:
#
# 1. The local working copy must be clean (check-clean.sh). fix-push.sh
#    later commits *everything* in it as the round's fix, so unrelated local
#    edits would otherwise ride along into the PR.
# 2. The local head must BE the PR's head (headRefOid). Otherwise fix_issues
#    would work on a different tree than the diff the reviewers see, and
#    fix-push.sh would push unrelated history onto the PR branch. "Local
#    head" is `HEAD` under git and `@-` under jj (the workflow's convention:
#    @ is the empty working-copy commit on top of the PR head —
#    `jj bookmark track <branch>@origin && jj new <branch>` gets you there;
#    fix-push.sh also tracks it). Right after a push GitHub can briefly
#    still report the previous head, so a mismatch is re-checked a few times
#    before failing.
# 3. The full PR diff is fetched with `gh pr diff` into a file and its path
#    is written to state as `diff_file`. The reviewers read it as a
#    plain-file context: entry, which — unlike a `!cmd` entry, whose failure
#    silently degrades to empty output — fails the branch loudly if the
#    file is missing. A gh error or an empty diff fails this step, so a
#    round can never "review" nothing and conclude the PR is clean. The
#    file lives under the user cache dir, outside the working copy (so
#    fix-push.sh never commits it) and outside /tmp (so it survives a
#    reboot and a resumed run can still read it).
#
# 4. Delta scoping (`review_pass`/`delta_file`), the mechanism that keeps a
#    fix-and-push cycle from re-triggering a full re-review every time:
#      - <last_reviewed_head> (state key `last_reviewed_head`, "" until a
#        round has completed once) empty  -> review_pass=full, delta_file
#        is a small placeholder file (there is nothing narrower yet, and
#        the reviewers are told to review diff_file instead) — NOT
#        diff_file itself, which would otherwise be inlined into each
#        reviewer's context twice (diff_file and delta_file both, per the
#        step `context:` lists in review-pr.yaml) for the price of one.
#        review-route.sh's delta-scope enforcement only reads delta_file's
#        paths when review_pass is "delta" (see its `case` there), so a
#        full pass's placeholder content is never parsed as a diff.
#      - <last_reviewed_head> non-empty   -> review_pass=delta, delta_file
#        is a *second* diff, <prev>..<head>, computed from the local VCS
#        (`jj diff --git --from <prev> --to <head>`, or
#        `git diff <prev> <head>`) — only what changed since the last round
#        this workflow actually reviewed, however many push cycles that
#        took. A failed or empty delta diff fails the step loudly, the same
#        hard-input philosophy as the full diff above: a round must never
#        silently fall back to reviewing nothing.
#    `last_reviewed_head` is then written as the current PR head, so the
#    round that follows THIS one is a delta round scoped to what happens
#    between now and then — including the "address one more nit" case:
#    ask_optional -> take_optional -> fix_issues -> fix_push -> back here
#    lands with `review_pass=delta`, never a full re-review, because
#    last_reviewed_head was already advanced to this head.
#
# Prints {"head_sha": ..., "diff_file": ..., "delta_file": ...,
# "review_pass": ..., "ci_round": false, "fix_note": "",
# "last_reviewed_head": ...} on one line (emits: json). `ci_round: false`
# marks the round that follows as a review round (a fresh round always
# starts here), so unchanged_route can tell it apart from a
# ci_failure-originated one later. `fix_note` is reset to "" here so a
# prior round's ask_optional instructions never leak into a new round.
# PAWL_REVIEW_HEAD_TRIES / PAWL_REVIEW_HEAD_SLEEP tune the head re-check
# (defaults 10 tries, 3s apart); tests set them low.
set -eu

vcs="${1:?prepare-review.sh: vcs argument required}"
pr_number="${2:?prepare-review.sh: pr_number argument required}"
repo="${3:?prepare-review.sh: repo argument required}"
base_branch="${4:?prepare-review.sh: base_branch argument required}"
last_reviewed_head="${5:-}"
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tries_max="${PAWL_REVIEW_HEAD_TRIES:-10}"
sleep_s="${PAWL_REVIEW_HEAD_SLEEP:-3}"

# base_branch isn't needed by the diff logic below (both diffs are either
# gh-relative or relative to the previously-reviewed local revision), but is
# kept as an argument: it is cheap to have on hand and the caller already
# resolves it for other steps.
: "${base_branch}"

# fetch-pr.sh resolved this once from origin's URL; every gh call here uses
# it instead of letting gh infer a repo from cwd, which fails outright in a
# non-colocated jj workspace (no .git directory at all).
export GH_REPO="$repo"

"$script_dir/check-clean.sh" "$vcs" "prepare-review.sh: the working copy has uncommitted changes; the fix step
commits everything in it, so start from a clean checkout of the PR head:"

case "$vcs" in
  jj) local_head=$(jj log --no-graph -r @- -T commit_id) ;;
  git) local_head=$(git rev-parse HEAD) ;;
esac

tries=0
while :; do
  # Not a pipeline: without pipefail a gh failure would surface only as a
  # misleading head mismatch. Fail on the real cause instead.
  if ! view=$(gh pr view "$pr_number" --json headRefOid); then
    echo "prepare-review.sh: gh pr view ${pr_number} failed" >&2
    exit 1
  fi
  if ! pr_head=$(printf '%s' "$view" | jq -er '.headRefOid'); then
    echo "prepare-review.sh: gh pr view ${pr_number} returned no headRefOid: ${view}" >&2
    exit 1
  fi
  if [ "$pr_head" = "$local_head" ]; then
    break
  fi
  tries=$((tries + 1))
  if [ "$tries" -ge "$tries_max" ]; then
    {
      echo "prepare-review.sh: local head ${local_head} is not PR #${pr_number}'s head ${pr_head}."
      echo "Check out the PR head first (git: gh pr checkout ${pr_number}; jj: jj bookmark track <pr-branch>@origin && jj new <pr-branch>)."
    } >&2
    exit 1
  fi
  sleep "$sleep_s"
done

cache_dir="${XDG_CACHE_HOME:-${HOME}/.cache}/pawl-review-pr"
mkdir -p "$cache_dir"
diff_file="${cache_dir}/pr-${pr_number}-${pr_head}.diff"

if ! gh pr diff "$pr_number" >"${diff_file}.tmp"; then
  rm -f "${diff_file}.tmp"
  echo "prepare-review.sh: gh pr diff ${pr_number} failed" >&2
  exit 1
fi
if [ ! -s "${diff_file}.tmp" ]; then
  rm -f "${diff_file}.tmp"
  echo "prepare-review.sh: gh pr diff ${pr_number} returned an empty diff — nothing to review" >&2
  exit 1
fi
mv "${diff_file}.tmp" "$diff_file"

if [ -z "$last_reviewed_head" ]; then
  review_pass="full"
  delta_file="${cache_dir}/pr-${pr_number}-${pr_head}.full-pass.txt"
  echo "This is a full pass: there is no delta. Review ${diff_file} in full." >"$delta_file"
else
  review_pass="delta"
  delta_file="${cache_dir}/pr-${pr_number}-${last_reviewed_head}..${pr_head}.delta.diff"
  case "$vcs" in
    jj)
      if ! jj diff --git --from "$last_reviewed_head" --to "$pr_head" >"${delta_file}.tmp" 2>"${delta_file}.tmp.err"; then
        cat "${delta_file}.tmp.err" >&2
        rm -f "${delta_file}.tmp" "${delta_file}.tmp.err"
        echo "prepare-review.sh: jj diff --git --from ${last_reviewed_head} --to ${pr_head} failed" >&2
        exit 1
      fi
      rm -f "${delta_file}.tmp.err"
      ;;
    git)
      if ! git diff "$last_reviewed_head" "$pr_head" >"${delta_file}.tmp" 2>"${delta_file}.tmp.err"; then
        cat "${delta_file}.tmp.err" >&2
        rm -f "${delta_file}.tmp" "${delta_file}.tmp.err"
        echo "prepare-review.sh: git diff ${last_reviewed_head} ${pr_head} failed" >&2
        exit 1
      fi
      rm -f "${delta_file}.tmp.err"
      ;;
    *)
      echo "prepare-review.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
      exit 1
      ;;
  esac
  if [ ! -s "${delta_file}.tmp" ]; then
    rm -f "${delta_file}.tmp"
    echo "prepare-review.sh: the ${last_reviewed_head}..${pr_head} delta diff is empty — a delta round must have something to review (did fix_push actually push a change?)" >&2
    exit 1
  fi
  mv "${delta_file}.tmp" "$delta_file"
fi

jq -cn --arg head_sha "$pr_head" --arg diff_file "$diff_file" --arg delta_file "$delta_file" \
  --arg review_pass "$review_pass" --arg last_reviewed_head "$pr_head" \
  '{head_sha: $head_sha, diff_file: $diff_file, delta_file: $delta_file, review_pass: $review_pass, ci_round: false, fix_note: "", last_reviewed_head: $last_reviewed_head}'
