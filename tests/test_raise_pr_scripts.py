# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the front-end scripts of the `raise-pr` pawl workflow.

The workflow lives in `.claude/workflows/raise-pr.yaml`; these cover the
steps that run before a PR exists (`preflight`, `open_pr`), against a
throwaway git repo whose `origin` is a local bare repo and with a fake `gh`
on PATH, so nothing here touches GitHub.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / ".claude" / "workflows" / "scripts"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("jq") is None,
    reason="the workflow scripts need git and jq",
)


def run(cmd, cwd, env=None, *, check=True):
    result = subprocess.run(  # noqa: S603
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        pytest.fail(f"{cmd} exited {result.returncode}:\n{result.stderr}")
    return result


def git(cwd, *args):
    return run(["git", *args], cwd).stdout.strip()


# --- check-title.sh -----------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "feat: add a summary page",
        "fix: flatten the nested Wayback retries",
        "ci: check CRLF endings",
        "build(deps): bump ruff",
        "refactor!: drop the old fetcher",
        "test(heal): cover the retry budget",
    ],
)
def test_check_title_accepts_conventional_commits(title, tmp_path):
    result = run([SCRIPTS / "check-title.sh", title], tmp_path, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "title",
    [
        "Add a summary page",
        "feature: add a summary page",
        "fix:no space after the colon",
        "fix: ",
        "Fix: capitalised type",
        "fix(): empty scope",
        "fix: " + "x" * 80,
    ],
)
def test_check_title_rejects_everything_else(title, tmp_path):
    result = run([SCRIPTS / "check-title.sh", title], tmp_path, check=False)
    assert result.returncode != 0
    assert "Conventional Commits" in result.stderr


# --- a git repo with a local origin -------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """A clone of a bare repo, with origin's URL spelled as a github.com URL.

    `url.<bare>.insteadOf` rewrites it back to the bare repo for git, while
    the scripts (which parse owner/repo out of origin's URL) still see
    github.com.
    """
    bare = tmp_path / "origin.git"
    run(
        ["git", "init", "--quiet", "--bare", "--initial-branch=main", str(bare)],
        tmp_path,
    )
    work = tmp_path / "work"
    run(["git", "init", "--quiet", "--initial-branch=main", str(work)], tmp_path)
    git(work, "config", "user.email", "test@example.com")
    git(work, "config", "user.name", "Test")
    git(work, "config", f"url.{bare}.insteadOf", "git@github.com:owner/repo.git")
    git(work, "remote", "add", "origin", "git@github.com:owner/repo.git")
    (work / "README.md").write_text("hello\n")
    git(work, "add", "-A")
    git(work, "commit", "--quiet", "-m", "initial")
    git(work, "push", "--quiet", "origin", "main")
    git(work, "checkout", "--quiet", "-b", "feature")
    (work / "new.txt").write_text("new\n")
    git(work, "add", "-A")
    git(
        work,
        "commit",
        "--quiet",
        "-m",
        "feat: add new.txt",
        "-m",
        "Because we need it.",
    )
    return work


@pytest.fixture
def fake_gh(tmp_path):
    """A `gh` that logs its argv and answers the few calls open-pr.sh makes.

    `FAKE_GH_EXISTING` set to a PR number makes `gh pr list` report an open
    PR for the head branch; otherwise it reports none and `gh pr create`
    "creates" PR 42.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh.log"
    gh = bin_dir / "gh"
    gh.write_text(
        f"""#!/usr/bin/env sh
printf '%s\\n' "$*" >> {log}
case "$1 $2" in
  "pr list")
    if [ -n "${{FAKE_GH_EXISTING:-}}" ]; then
      printf '[{{"number": %s, "url": "https://github.com/owner/repo/pull/%s"}}]\\n' \
        "$FAKE_GH_EXISTING" "$FAKE_GH_EXISTING"
    else
      echo '[]'
    fi ;;
  "pr create") echo "https://github.com/owner/repo/pull/42" ;;
  "pr edit") ;;
  *) echo "fake gh: unexpected call: $*" >&2; exit 1 ;;
esac
"""
    )
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return env, log


# --- preflight.sh -------------------------------------------------------------


def test_preflight_passes_on_a_clean_branch_ahead_of_base(repo):
    result = run(
        [SCRIPTS / "preflight.sh", "feat: add new.txt", "feature", "main"], repo
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {"vcs": "git"}


def test_preflight_rejects_a_bad_title(repo):
    result = run(
        [SCRIPTS / "preflight.sh", "Add new.txt", "feature", "main"], repo, check=False
    )
    assert result.returncode != 0
    assert "Conventional Commits" in result.stderr


def test_preflight_rejects_a_dirty_working_copy(repo):
    (repo / "stray.txt").write_text("uncommitted\n")
    result = run(
        [SCRIPTS / "preflight.sh", "feat: add new.txt", "feature", "main"],
        repo,
        check=False,
    )
    assert result.returncode != 0
    assert "stray.txt" in result.stderr


def test_preflight_rejects_pushing_to_the_base_branch(repo):
    result = run(
        [SCRIPTS / "preflight.sh", "feat: add new.txt", "main", "main"],
        repo,
        check=False,
    )
    assert result.returncode != 0
    assert "base" in result.stderr


def test_preflight_rejects_a_branch_with_nothing_to_merge(repo):
    git(repo, "reset", "--quiet", "--hard", "origin/main")
    result = run(
        [SCRIPTS / "preflight.sh", "feat: add new.txt", "feature", "main"],
        repo,
        check=False,
    )
    assert result.returncode != 0
    assert "no commits" in result.stderr


# --- open-pr.sh ---------------------------------------------------------------


def test_open_pr_pushes_and_creates_the_pr(repo, fake_gh):
    env, log = fake_gh
    result = run(
        [
            SCRIPTS / "open-pr.sh",
            "git",
            "feature",
            "main",
            "feat: add new.txt",
            "The body.",
        ],
        repo,
        env=env,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {"pr_number": "42"}
    assert git(repo, "ls-remote", "origin", "refs/heads/feature").split()[0] == git(
        repo, "rev-parse", "HEAD"
    )
    calls = log.read_text()
    assert "pr create" in calls
    assert "-R owner/repo" in calls
    assert "--title feat: add new.txt" in calls
    assert "--body The body." in calls


def test_open_pr_builds_the_body_from_the_commits_when_none_is_given(repo, fake_gh):
    env, log = fake_gh
    run(
        [SCRIPTS / "open-pr.sh", "git", "feature", "main", "feat: add new.txt", ""],
        repo,
        env=env,
    )
    assert "Because we need it." in log.read_text()


def test_open_pr_reuses_an_open_pr_for_the_branch(repo, fake_gh):
    env, log = fake_gh
    env = {**env, "FAKE_GH_EXISTING": "7"}
    result = run(
        [SCRIPTS / "open-pr.sh", "git", "feature", "main", "feat: add new.txt", ""],
        repo,
        env=env,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {"pr_number": "7"}
    calls = log.read_text()
    assert "pr create" not in calls
    # The title is the workflow's input, so an existing PR is brought in line.
    assert "pr edit 7" in calls
    assert "--title feat: add new.txt" in calls


def test_open_pr_refuses_a_non_fast_forward_push(repo, fake_gh):
    env, _ = fake_gh
    run(
        [SCRIPTS / "open-pr.sh", "git", "feature", "main", "feat: add new.txt", ""],
        repo,
        env=env,
    )
    git(repo, "commit", "--quiet", "--amend", "-m", "feat: rewritten")
    result = run(
        [SCRIPTS / "open-pr.sh", "git", "feature", "main", "feat: add new.txt", ""],
        repo,
        env=env,
        check=False,
    )
    assert result.returncode != 0
