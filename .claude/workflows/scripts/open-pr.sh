#!/usr/bin/env sh
# open-pr.sh <vcs> <head_branch> <base> <title> <body>
#
# Body of the `open_pr` deterministic step: pushes the local head (`HEAD`
# under git, `@-` under jj — preflight.sh has already checked the working
# copy is clean) to origin's <head_branch>, then opens a PR from it into
# <base>, or reuses the open one if the branch already has one (a resumed or
# re-run workflow, or a PR raised by hand before this ran) and sets its
# title to <title>. An empty <body> is built from the descriptions of the
# commits being proposed, oldest first.
#
# Fast-forward only, the same rule fix-push.sh follows: if origin already
# has <head_branch>, the local head must descend from it, or the step fails
# without pushing — re-running this never rewrites a PR branch.
#
# owner/repo is resolved from origin's URL, never inferred by gh from cwd
# (a non-colocated jj workspace has no .git for gh to look at) — the same
# parsing fetch-pr.sh does, which the next step re-derives on its own.
#
# Prints {"pr_number": "<n>"} on one line (emits: json).
set -eu

vcs="${1:?open-pr.sh: vcs argument required}"
head_branch="${2:?open-pr.sh: head_branch argument required}"
base="${3:?open-pr.sh: base argument required}"
title="${4:?open-pr.sh: title argument required}"
body="${5-}"

parse_github_repo() {
  url="$1"
  case "$url" in
    git@github.com:*) rest="${url#git@github.com:}" ;;
    https://github.com/*) rest="${url#https://github.com/}" ;;
    ssh://git@github.com/*) rest="${url#ssh://git@github.com/}" ;;
    *) return 1 ;;
  esac
  rest="${rest%/}"
  rest="${rest%.git}"
  case "$rest" in
    */*/*|"") return 1 ;;
    */*) printf '%s\n' "$rest" ;;
    *) return 1 ;;
  esac
}

not_ff() {
  echo "open-pr.sh: the local head does not descend from origin's ${head_branch} — refusing to push (it would rewrite the PR branch)" >&2
  exit 1
}

case "$vcs" in
  jj)
    origin_url=$(jj git remote list 2>/dev/null | awk '$1 == "origin" { print $2; exit }')
    jj git fetch --remote origin --branch "exact:\"${base}\"" --branch "exact:\"${head_branch}\"" >&2
    remote_rev="\"${head_branch}\"@origin"
    if [ -n "$(jj log --no-graph -r "present(${remote_rev})" -T commit_id)" ]; then
      # See fix-push.sh: a fetched remote bookmark is untracked by default,
      # and `jj git push --bookmark` refuses while it stays so.
      jj bookmark track "${head_branch}@origin" >&2 2>&1
      if [ -z "$(jj log --no-graph -r "(${remote_rev}) & ::@-" -T commit_id)" ]; then
        not_ff
      fi
    fi
    jj bookmark set "$head_branch" -r @- >&2
    jj git push --remote origin --bookmark "$head_branch" >&2
    if [ -z "$body" ]; then
      body=$(jj log --no-graph --reversed -r "\"${base}\"@origin..@-" -T 'description ++ "\n"')
    fi
    ;;
  git)
    # The configured URL, not `git remote get-url`, which applies any
    # url.*.insteadOf rewrite and can hide the github.com form.
    origin_url=$(git config --get remote.origin.url) || origin_url=""
    if git fetch --quiet origin "refs/heads/${head_branch}" 2>/dev/null; then
      git merge-base --is-ancestor FETCH_HEAD HEAD || not_ff
    fi
    git push --quiet origin "HEAD:refs/heads/${head_branch}"
    if [ -z "$body" ]; then
      git fetch --quiet origin "refs/heads/${base}"
      body=$(git log --reverse --format='%B' FETCH_HEAD..HEAD)
    fi
    ;;
  *)
    echo "open-pr.sh: unknown vcs '${vcs}' (expected jj or git)" >&2
    exit 1
    ;;
esac

if ! repo=$(parse_github_repo "${origin_url:-}"); then
  echo "open-pr.sh: origin's URL '${origin_url:-}' is not a github.com URL I can parse" >&2
  exit 1
fi

# Capture gh's output before parsing it, so a gh failure is not masked by
# jq's exit status (no pipefail under sh).
if ! existing=$(gh pr list -R "$repo" --head "$head_branch" --state open --json number,url); then
  echo "open-pr.sh: gh pr list failed" >&2
  exit 1
fi
pr_number=$(printf '%s' "$existing" | jq -r '.[0].number // empty')

if [ -n "$pr_number" ]; then
  gh pr edit "$pr_number" -R "$repo" --title "$title" >&2
else
  if ! url=$(gh pr create -R "$repo" --base "$base" --head "$head_branch" --title "$title" --body "$body"); then
    echo "open-pr.sh: gh pr create failed" >&2
    exit 1
  fi
  pr_number="${url##*/}"
fi

case "$pr_number" in
  ''|*[!0-9]*)
    echo "open-pr.sh: could not tell the PR number (got '${pr_number}')" >&2
    exit 1
    ;;
esac

jq -cn --arg pr_number "$pr_number" '{pr_number: $pr_number}'
