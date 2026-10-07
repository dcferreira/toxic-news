#!/usr/bin/env bash
# Runs the self-fix agent's code as `selfheal`, a user with no sudo, no docker
# and no write access to anything of the runner's (docs/self-fix-design.md,
# Safety).
#
# The agent has bash, and selfheal-check imports the newspapers.py it wrote.
# Run as the runner's own user, that code could rewrite a JavaScript action a
# later step runs, which is handed the job's ACTIONS_RUNTIME_TOKEN, and so
# write an Actions cache entry that Update, which can push, would restore. As
# `selfheal` it can reach neither the actions nor the token.
#
#   sandbox.sh setup CHECKOUT  copy CHECKOUT to the sandbox, prove it isolated
#   sandbox.sh run COMMAND     run COMMAND (a bash string) in the sandbox's copy
#   sandbox.sh get PATH DEST   copy the file PATH, relative to /srv/selfheal,
#                              out as the sandbox user reads it, so a symlink
#                              it left leaks nothing
#
# The copy is at /srv/selfheal/repo; outputs go in /srv/selfheal/out. Each
# `run` starts with `uv sync --frozen`, so the sandbox has its own environment.
set -euo pipefail

SANDBOX_USER=selfheal
ROOT=/srv/selfheal

# PRESERVE_ENV names the variables the command gets, comma-separated: through
# sudo, never on its command line, which every user can read.
as_sandbox() {
  local keep=()
  if [ -n "${PRESERVE_ENV:-}" ]; then
    keep=(--preserve-env="$PRESERVE_ENV")
  fi
  sudo --user "$SANDBOX_USER" --set-home "${keep[@]}" -- "$@"
}

setup() {
  local checkout=$1
  sudo useradd --create-home --shell /bin/bash "$SANDBOX_USER"
  sudo install -d -m 755 "$ROOT"
  sudo cp -a "$checkout" "$ROOT/repo"
  # the runner's environment is not the sandbox's to use
  sudo rm -rf "$ROOT/repo/.venv"
  sudo chown -R "$SANDBOX_USER:" "$ROOT/repo"
  sudo install -d -o "$SANDBOX_USER" -m 755 "$ROOT/out"
  sudo install -m 755 "$(command -v uv)" /usr/local/bin/uv
  # nothing of the runner's home is the sandbox's to read either
  sudo chmod o-rwx "$HOME"

  # GitHub's runners leave some of these world-writable (/opt, for one), which
  # would let the sandbox swap out a tool a later step runs. The sandbox user
  # is in no group but its own, so taking that bit away is enough.
  # (GitHub's Ubuntu image has some 800,000 such paths, so this takes minutes.)
  local opened
  opened=$(sudo find "$HOME" /opt /usr /etc -xdev -perm -o+w -not -type l \
    -print -exec chmod o-w {} + | wc -l)
  echo "Took write access away from others on $opened paths"

  # The isolation is checked on every run, not assumed.
  local writable
  # symlinks are left out: one to /dev/null reads as writable
  writable=$(as_sandbox find "$HOME" /opt /usr /etc -xdev -writable \
    -not -type l -print -quit 2>/dev/null || true)
  if [ -n "$writable" ]; then
    echo "::error::The sandbox user can write to $writable"
    exit 1
  fi
  if as_sandbox ls "$HOME" >/dev/null 2>&1; then
    echo "::error::The sandbox user can read $HOME"
    exit 1
  fi
  if as_sandbox sudo -n true 2>/dev/null; then
    echo "::error::The sandbox user can sudo"
    exit 1
  fi
  if as_sandbox test -r "/proc/$$/environ"; then
    echo "::error::The sandbox user can read the runner's environment"
    exit 1
  fi
  if as_sandbox test -w /var/run/docker.sock; then
    echo "::error::The sandbox user can reach docker"
    exit 1
  fi
  echo "The sandbox user can write nowhere of the runner's"
}

run() {
  as_sandbox bash -c "set -euo pipefail; cd $ROOT/repo; uv sync --frozen --quiet; $1"
}

get() {
  local path=$1 dest=$2
  if as_sandbox test -f "$ROOT/$path"; then
    as_sandbox cat "$ROOT/$path" >"$dest"
  else
    echo "The sandbox left no $path"
  fi
}

"$@"
