# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for `poe selfheal-pr`, what the self-fix loop's PR job writes."""

import json
import re
from datetime import date
from typing import Any

import pytest
import yaml

from tests.selfheal_check import NEWSPAPERS_PY
from tests.selfheal_fix import _ROOT, attempt_marker
from tests.selfheal_pr import (
    MAX_BODY,
    Evidence,
    branch,
    file_diffs,
    fixed_xpath,
    main,
    outcome,
    render_body,
    render_comment,
    title,
)
from tests.test_selfheal_fix import _task, newspapers_source

BBC_ENTRY = """get_headlines_fn=get_xpath_fn(
            "//h3[@class='media__title' and a]", href_xpath="a"
        ),"""
NEW_XPATH = "//h2[@data-testid='card-headline']"
FIXED_ENTRY = f"""get_headlines_fn=get_dated_xpath_fn(
            DatedXpath(
                from_date=None,
                headline_xpath="//h3[@class='media__title' and a]",
                href_xpath="a",
            ),
            DatedXpath(
                from_date=datetime(2026, 10, 4, tzinfo=timezone.utc),
                headline_xpath={NEW_XPATH!r},
                href_xpath="ancestor::a",
            ),
        ),"""


def _source(entry: str = FIXED_ENTRY) -> str:
    source = newspapers_source()
    assert BBC_ENTRY in source
    return source.replace(BBC_ENTRY, entry, 1)


# --- the XPath to outline ----------------------------------------------------


def test_the_xpath_is_the_one_dated_from_the_tasks_from_date():
    assert fixed_xpath(_source(), _task()) == NEW_XPATH


def test_no_xpath_when_the_fix_dates_none_from_the_tasks_day():
    assert (
        fixed_xpath(_source(), {**_task(), "from_date": "2026-10-01T00:00:00+00:00"})
        is None
    )
    assert fixed_xpath(_source(BBC_ENTRY), _task()) is None


def test_no_xpath_when_it_is_not_a_plain_string():
    entry = FIXED_ENTRY.replace(f"{NEW_XPATH!r}", "XPATH + '//a'")
    assert fixed_xpath(_source(entry), _task()) is None


def test_the_xpath_command_prints_it(tmp_path, capsys):
    (tmp_path / "task.json").write_text(json.dumps(_task()))
    (tmp_path / "newspapers.py").write_text(_source())
    assert (
        main(["xpath", str(tmp_path / "task.json"), str(tmp_path / "newspapers.py")])
        == 0
    )
    assert capsys.readouterr().out == NEW_XPATH + "\n"


# --- names --------------------------------------------------------------------


def test_the_title_is_a_conventional_commit_and_the_branch_is_dated():
    assert title(_task()) == "fix: repair the BBC headline XPath"
    assert branch(_task()) == "selfheal/bbc.com-2026-10-05"


# --- the body -----------------------------------------------------------------

PAGE = "tests/assets/html/bbc.com/2026-10-05.html"
PATCH = f"""diff --git a/{NEWSPAPERS_PY} b/{NEWSPAPERS_PY}
--- a/{NEWSPAPERS_PY}
+++ b/{NEWSPAPERS_PY}
@@ -1 +1 @@
-old
+new ```` fence
diff --git a/{PAGE} b/{PAGE}
new file mode 100644
--- /dev/null
+++ b/{PAGE}
@@ -0,0 +1 @@
+<html>PAGE BYTES</html>
"""


def _evidence(**overrides: Any) -> Evidence:
    headlines = [[f"Story {i}", f"https://bbc.com/{i}"] for i in range(1, 45)]
    headlines[0] = ["@everyone | <img src=x> #1 [link](http://x)", "https://bbc.com/a"]
    fields: dict[str, Any] = {
        "task": _task(),
        "last_good": {
            "date": "2023-05-20",
            "source": "tests/snapshots/test_parse/bbc.com/2023-05-20.txt",
            "headlines": [{"text": "Old story", "url": "https://bbc.com/o"}] * 45,
        },
        "result": {
            "status": "fixed",
            "explanation": "Cards moved.\nNow h2. @dcferreira look",
            "excluded": "Newsletter cards.",
            "session": {"turns": 23, "cost_usd": 0.0412},
        },
        "patch": PATCH,
        "check": {
            "passed": True,
            "results": [{"name": "count", "passed": True, "detail": "44 | in band"}],
            "headlines": headlines,
        },
        "outline": {"matches": 44, "outlined": 44, "hidden": 0},
        "pixels": {"found": 44},
    }
    return Evidence(**{**fields, **overrides})


def _body(
    evidence: Evidence | None = None,
    *,
    issue: int | None = 14,
    screenshot_url: str | None = "https://shot.png",
) -> str:
    return render_body(
        evidence or _evidence(),
        issue=issue,
        run_url="https://run",
        screenshot_url=screenshot_url,
    )


def test_the_body_closes_the_issue_and_shows_the_counts():
    body = _body()
    assert "Closes #14." in body
    assert "| newest recorded page (2023-05-20, old extractor) | 45 | 47 |" in body
    assert "| today (2026-10-05), old extractor | 0 | 47 |" in body
    assert "| today (2026-10-05), new extractor | 44 | 47 |" in body
    assert "![BBC front page](https://shot.png)" in body
    assert "every outline was found in the screenshot's pixels" in body
    assert "deepseek-flash · 23 turns · $0.041" in body


def test_the_body_shows_only_the_newspapers_change_in_a_fence_it_cannot_close():
    body = _body()
    assert "+new ```` fence\n`````" in body
    assert "PAGE BYTES" not in body
    assert "`tests/assets/html/bbc.com/2026-10-05.html`" in body


def test_scraped_and_agent_text_is_inert():
    body = _body()
    for live in ("@everyone", "<img", "| <", "> #1 ", "[link]", "@dcferreira"):
        assert live not in body
    assert "&#64;everyone" in body
    assert "> Cards moved.\n> Now h2. &#64;dcferreira look" in body


def test_the_body_says_when_outlines_are_missing_from_the_pixels():
    body = _body(_evidence(pixels={"found": 40}))
    assert "only 40 of the outlines were found" in body


def test_the_body_without_a_screenshot_says_why():
    body = _body(_evidence(outline=None), screenshot_url=None)
    assert "No screenshot" in body
    assert "![" not in body


def test_the_body_without_a_screenshot_url_still_reports_the_outline():
    body = _body(_evidence(), screenshot_url=None)
    assert "The browser matched" in body
    assert "No screenshot" not in body
    assert "![" not in body


def test_the_body_fits_githubs_limit():
    long = [[f"Story {i} " + "x" * 280, f"https://bbc.com/{i}"] for i in range(400)]
    check = {"passed": True, "results": [], "headlines": long}
    body = _body(_evidence(check=check))
    assert len(body) <= MAX_BODY
    assert body.endswith("the run has the rest)_\n")


def test_the_last_good_day_is_named_when_there_is_one():
    task = {**_task(), "last_good": "2026-10-01"}
    good = {"date": "2026-10-01", "source": "data branch", "headlines": []}
    body = _body(_evidence(task=task, last_good=good))
    assert "last good on 2026-10-01" in body
    assert "| last good (2026-10-01, old extractor) | 0 | 47 |" in body


# --- the issue comment ----------------------------------------------------------


def test_a_comment_carries_the_attempt_marker():
    comment = render_comment(_evidence(), outcome="pr", run_url="https://run", pr=31)
    assert "opened #31" in comment
    assert comment.rstrip().endswith(attempt_marker(date(2026, 10, 5)))


@pytest.mark.parametrize("outcome", ["gave_up", "refused", "failed", "no_patch"])
def test_a_comment_on_an_attempt_without_a_pr_quotes_the_agent(outcome):
    comment = render_comment(_evidence(), outcome=outcome, run_url="https://run")
    assert "tries again in a week" in comment
    assert "> Cards moved." in comment
    assert "<!-- selfheal-attempt 2026-10-05 -->" in comment


# --- reading the evidence -----------------------------------------------------


def test_evidence_is_read_from_its_directory_missing_parts_and_all(tmp_path):
    (tmp_path / "task.json").write_text(json.dumps(_task()))
    (tmp_path / "last_good.json").write_text(json.dumps({"date": "x", "headlines": []}))
    (tmp_path / "check.json").write_text("not json")
    (tmp_path / "patch.diff").write_bytes(b"diff \xff\r\n")
    evidence = Evidence.read(tmp_path)
    assert evidence.check is None
    assert evidence.result == {}
    assert not evidence.passed
    assert evidence.patch == "diff �\r\n"


def test_file_diffs_splits_a_patch_by_path():
    parts = file_diffs(PATCH)
    assert list(parts) == [NEWSPAPERS_PY, "tests/assets/html/bbc.com/2026-10-05.html"]
    assert parts[NEWSPAPERS_PY].endswith("+new ```` fence\n")


# --- what came of the attempt -------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "problems", "wanted"),
    [
        ({}, [], "pr"),
        ({"result": {}}, [], "none"),
        ({"result": {"status": "deferred"}}, [], "none"),
        ({"result": {"status": "gave_up"}}, [], "gave_up"),
        ({"patch": ""}, [], "no_patch"),
        ({}, ["scope"], "refused"),
        ({"check": None}, [], "failed"),
        ({"check": {"passed": False, "results": [], "headlines": []}}, [], "failed"),
    ],
)
def test_the_outcome_opens_a_pr_only_for_a_passing_patch_within_scope(
    changes: dict[str, Any], problems: list[str], wanted: str
):
    assert outcome(_evidence(**changes), problems) == wanted


def test_the_outcome_command_gates_the_patch_itself(tmp_path, capsys):
    evidence = _evidence()
    (tmp_path / "task.json").write_text(json.dumps(evidence.task))
    (tmp_path / "last_good.json").write_text(json.dumps(evidence.last_good))
    (tmp_path / "result.json").write_text(json.dumps(evidence.result))
    (tmp_path / "check.json").write_text(json.dumps(evidence.check))
    # passes selfheal-check, but reaches beyond BBC
    (tmp_path / "patch.diff").write_text(PATCH.replace("bbc.com", "foxnews.com"))
    assert main(["outcome", "--evidence", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "refused"


# --- the workflow's own guarantees ----------------------------------------------

WORKFLOW = yaml.safe_load((_ROOT / ".github/workflows/selfheal.yml").read_text())
JOBS: dict[str, Any] = WORKFLOW["jobs"]


def _steps(job: str) -> list[dict[str, Any]]:
    return JOBS[job]["steps"]


def test_only_the_pr_job_can_write():
    assert WORKFLOW["permissions"] == {}
    for name, job in JOBS.items():
        writes = {k for k, v in job.get("permissions", {}).items() if v == "write"}
        wanted = {"contents", "pull-requests", "issues"} if name == "pr" else set()
        assert writes == wanted, name


def test_only_the_fix_job_holds_the_deepseek_key():
    for name, job in JOBS.items():
        assert ("DEEPSEEK_API_KEY" in yaml.safe_dump(job)) == (name == "fix"), name
    assert JOBS["fix"]["environment"] == "selfheal"


def test_the_agent_and_the_checker_only_run_in_the_sandbox():
    """Both run the patch's code: the agent has bash, the checker imports it."""
    ran = []
    for name, job in JOBS.items():
        for step in job["steps"]:
            run = step.get("run", "")
            if "poe selfheal-fix agent" in run or "poe selfheal-check" in run:
                assert run.startswith(".github/selfheal-sandbox.sh run"), step["name"]
                ran.append(name)
    assert ran == ["fix", "check"]


def test_no_job_with_write_access_or_trusted_code_holds_the_patch_in_its_checkout():
    """`check` and `pr` run their own Python after the patch is applied."""
    for job in ("check", "pr"):
        for step in _steps(job):
            for line in step.get("run", "").splitlines():
                if " apply" in line and "git" in line:
                    assert re.search(r'git -C "\$RUNNER_TEMP/\w+" apply', line), line


def test_no_job_writes_or_restores_a_cache():
    """A cache entry on main would reach Update, which can push."""
    for name, job in JOBS.items():
        for step in job["steps"]:
            uses = step.get("uses", "")
            assert not uses.startswith("actions/cache"), name
            if uses.startswith("astral-sh/setup-uv"):
                assert step["with"]["enable-cache"] is False, name


def test_an_automatic_run_needs_the_switch_and_main():
    condition = JOBS["prepare"]["if"]
    assert "vars.SELFHEAL_AUTO" in condition
    assert "github.event.workflow_run.head_branch == 'main'" in condition
    assert WORKFLOW[True]["workflow_run"] == {
        "workflows": ["Update"],
        "types": ["completed"],
    }


def test_a_rerun_of_the_pr_job_can_push_its_screenshot_again():
    """A rerun pushes the same screenshot: a path of its own."""
    (step,) = [s for s in _steps("pr") if s.get("id") == "screenshot"]
    assert 'path="$OUTLET/$day-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT.png"' in step["run"]
