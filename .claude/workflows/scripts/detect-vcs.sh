#!/usr/bin/env sh
# detect-vcs.sh
#
# Prints "jj" if the working copy has a .jj directory (including a
# colocated jj+git repo, which is what this repo itself is), otherwise
# "git". Shared by any step that needs to branch its VCS commands, so the
# detection logic lives in exactly one place.
set -eu

if [ -d .jj ]; then
  echo jj
else
  echo git
fi
