#!/usr/bin/env sh
# check-title.sh <title>
#
# Exits 0 iff <title> is a Conventional Commits subject this repo accepts
# for a PR title: `type(scope)!: description`, with an optional scope and
# `!`, a lower-case type from the list below, one space after the colon,
# a non-empty description, and at most 72 characters in all. PRs are
# squash-merged, so the PR title becomes the commit on main — that is why
# it is the title, not each commit, that has to conform. Otherwise says why
# on stderr and exits 1.
set -eu

title="${1-}"
types='build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test'

fail() {
  echo "check-title.sh: PR title '${title}' is not a Conventional Commits subject: $1 (expected e.g. 'fix: flatten the nested Wayback retries'; types: ${types})" >&2
  exit 1
}

if ! printf '%s' "$title" | grep -Eq "^(${types})(\([a-z0-9._/-]+\))?!?: [^ ].*$"; then
  fail "wrong shape"
fi
if [ "${#title}" -gt 72 ]; then
  fail "longer than 72 characters"
fi
