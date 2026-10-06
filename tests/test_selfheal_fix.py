# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for `poe selfheal-fix`, the fix job of the self-fix loop."""

import json
import re
import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.fixtures import latest_fixture
from tests.selfheal_check import NEWSPAPERS_PY
from tests.selfheal_fix import (
    _ROOT,
    broken_outlets,
    build_prompt,
    duration,
    gate,
    history,
    last_good_headlines,
    off_peak_left,
    omp_command,
    patched_newspapers,
    read_result,
    run_agent,
    summarise_session,
    write_inputs,
)
from toxic_news.fetchers import Headline
from toxic_news.health import OutletHealth, Verdict, write_health
from toxic_news.models import Scores
from toxic_news.newspapers import Newspaper, newspapers
from toxic_news.store import write_headlines

BBC = next(n for n in newspapers if n.name == "BBC")
FOX = next(n for n in newspapers if n.name == "Fox News")
TODAY = date(2026, 10, 5)


def _health(newspaper: Newspaper, verdict: Verdict, headlines: int = 0) -> OutletHealth:
    return OutletHealth(
        newspaper=newspaper.name,
        url=str(newspaper.url),
        status=200,
        fetch_error=None,
        parse_error=None,
        body_bytes=500_000,
        headlines=headlines,
        expected=newspaper.expected_headlines,
        verdict=verdict,
    )


def _days(data_dir: Path, verdicts: dict[date, dict[str, Verdict]]) -> None:
    by_name = {n.name: n for n in newspapers}
    for day, outlets in verdicts.items():
        write_health(
            data_dir, day, [_health(by_name[name], v) for name, v in outlets.items()]
        )


def _headline(newspaper: Newspaper, text: str, day: date) -> Headline:
    return Headline(
        newspaper=newspaper.name,
        language="en",
        text=text,
        url=f"{newspaper.url}/{abs(hash(text))}",
        date=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
        scores=Scores(
            toxicity=0,
            severe_toxicity=0,
            obscene=0,
            identity_attack=0,
            insult=0,
            threat=0,
            sexual_explicit=0,
            positive=0,
            neutral=1,
            negative=0,
        ),
    )


OK, XPATH, OTHER = Verdict.OK, Verdict.XPATH, Verdict.OTHER


# --- which outlets to fix -----------------------------------------------------


def test_broken_outlets_are_those_with_an_xpath_verdict_today(tmp_path):
    _days(tmp_path, {TODAY: {"BBC": XPATH, "Fox News": OK, "Reuters": OTHER}})
    assert broken_outlets(tmp_path, TODAY) == [BBC]


def test_no_health_report_today_means_nothing_to_fix(tmp_path):
    assert broken_outlets(tmp_path, TODAY) == []


# --- when it broke ------------------------------------------------------------


def test_history_finds_the_first_bad_day_after_the_last_good_one(tmp_path):
    _days(
        tmp_path,
        {
            date(2026, 10, 1): {"BBC": OK},
            date(2026, 10, 2): {"BBC": OK},
            date(2026, 10, 3): {"BBC": XPATH},
            # a missing day or a blocked fetch does not end the broken streak
            date(2026, 10, 5): {"BBC": OTHER},
            date(2026, 10, 6): {"BBC": XPATH},
        },
    )
    found = history(tmp_path, BBC, date(2026, 10, 6))
    assert found.last_good == date(2026, 10, 2)
    assert found.first_bad == date(2026, 10, 3)


def test_history_without_a_good_day_starts_at_the_first_bad_one_recorded(tmp_path):
    _days(
        tmp_path,
        {
            date(2026, 9, 29): {"BBC": OTHER},
            date(2026, 9, 30): {"BBC": XPATH},
            TODAY: {"BBC": XPATH},
        },
    )
    found = history(tmp_path, BBC, TODAY)
    assert found.last_good is None
    assert found.first_bad == date(2026, 9, 30)


def test_history_of_an_outlet_broken_only_today(tmp_path):
    _days(tmp_path, {TODAY: {"BBC": XPATH}})
    found = history(tmp_path, BBC, TODAY)
    assert (found.last_good, found.first_bad) == (None, TODAY)


# --- what a real headline looks like ------------------------------------------


def test_last_good_headlines_come_from_the_data_branch(tmp_path):
    day = date(2026, 10, 2)
    write_headlines(
        tmp_path,
        day,
        [
            _headline(BBC, "A real BBC story", day),
            _headline(FOX, "A Fox story", day),
        ],
    )
    found = last_good_headlines(tmp_path, BBC, day)
    assert found["date"] == "2026-10-02"
    assert found["source"] == "data branch, headlines/2026/10/02.csv"
    assert [h["text"] for h in found["headlines"]] == ["A real BBC story"]


def test_last_good_headlines_fall_back_to_the_newest_fixture(tmp_path):
    fixture = latest_fixture("bbc.com")
    found = last_good_headlines(tmp_path, BBC, None)
    assert found["date"] == fixture.date.isoformat()
    assert found["source"] == "tests/snapshots/test_parse/bbc.com/2023-05-20.txt"
    snapshot = json.loads(fixture.snapshot_path.read_text())
    assert found["headlines"][0] == {"text": snapshot[0][0], "url": snapshot[0][1]}


# --- the inputs the agent gets ------------------------------------------------


def test_write_inputs_lays_out_the_task(tmp_path):
    data, raw, out = tmp_path / "data", tmp_path / "raw", tmp_path / "out"
    _days(data, {date(2026, 10, 4): {"BBC": XPATH}, TODAY: {"BBC": XPATH}})
    raw.mkdir()
    (raw / "bbc.com.html").write_bytes(b"<html>today</html>")

    inputs = write_inputs(
        BBC, raw_html_dir=raw, data_dir=data, today=TODAY, out_dir=out
    )

    assert inputs == out / "bbc.com"
    assert (inputs / "today.html").read_bytes() == b"<html>today</html>"
    task = json.loads((inputs / "task.json").read_text())
    assert task == {
        "outlet": "BBC",
        "slug": "bbc.com",
        "url": "https://bbc.com",
        "today": "2026-10-05",
        "expected_headlines": 47,
        "headlines_today": 0,
        "first_bad": "2026-10-04",
        "last_good": None,
        "from_date": "2026-10-04T00:00:00+00:00",
        "fixture": "tests/assets/html/bbc.com/2026-10-05.html",
        "snapshot": "tests/snapshots/test_parse/bbc.com/2026-10-05.txt",
        "extractor": task["extractor"],
    }
    assert task["extractor"].startswith("Newspaper(\n")
    assert 'name="BBC"' in task["extractor"]
    last_good = json.loads((inputs / "last_good.json").read_text())
    assert last_good["source"].startswith("tests/snapshots/test_parse/bbc.com/")


def test_write_inputs_needs_the_fetched_page(tmp_path):
    _days(tmp_path, {TODAY: {"BBC": XPATH}})
    with pytest.raises(SystemExit, match=r"bbc\.com\.html"):
        write_inputs(
            BBC,
            raw_html_dir=tmp_path,
            data_dir=tmp_path,
            today=TODAY,
            out_dir=tmp_path / "out",
        )


# --- the gate a patch passes before any of its code runs ----------------------

NEWSPAPERS = Path(NEWSPAPERS_PY)


def _repo(tmp_path: Path) -> Path:
    """Return a git repository holding this checkout's `newspapers.py`."""
    repo = tmp_path / "repo"
    (repo / NEWSPAPERS).parent.mkdir(parents=True)
    shutil.copyfile(_ROOT / NEWSPAPERS, repo / NEWSPAPERS)
    existing = repo / "tests/assets/html/bbc.com/2023-05-20.html"
    existing.parent.mkdir(parents=True)
    existing.write_text("<html>old</html>\n")
    (repo / ".gitignore").write_text("/selfheal/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return repo


def _git(repo: Path, *args: str) -> str:
    git = shutil.which("git")
    assert git is not None
    return subprocess.run(  # noqa: S603
        [git, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _diff(repo: Path) -> str:
    _git(repo, "add", "-N", ".")
    return _git(repo, "diff", "--binary", "--no-renames", "HEAD")


BBC_OLD_EXTRACTOR = """get_xpath_fn(
            "//h3[@class='media__title' and a]", href_xpath="a"
        ),"""

#: Not a real fix: these tests only need some edit to BBC's entry.
STUB_XPATH = "//h2[@data-test='stub-only']"


def _dated_bbc(from_date: str = "datetime(2026, 10, 4, tzinfo=timezone.utc)") -> str:
    return f"""get_dated_xpath_fn(
            DatedXpath(
                from_date=None,
                headline_xpath="//h3[@class='media__title' and a]",
                href_xpath="a",
            ),
            DatedXpath(
                from_date={from_date},
                headline_xpath="{STUB_XPATH}",
                href_xpath="a",
            ),
        ),"""


def _fix_bbc(
    repo: Path, *, extractor: str | None = None, day: str = "2026-10-05"
) -> None:
    source = (repo / NEWSPAPERS).read_text()
    assert BBC_OLD_EXTRACTOR in source
    (repo / NEWSPAPERS).write_text(
        source.replace(BBC_OLD_EXTRACTOR, extractor or _dated_bbc())
    )
    page = repo / f"tests/assets/html/bbc.com/{day}.html"
    page.write_text("<html>today</html>\n")
    snapshot = repo / f"tests/snapshots/test_parse/bbc.com/{day}.txt"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("[]")


def _base(repo: Path) -> str:
    return _git(repo, "show", f"HEAD:{NEWSPAPERS_PY}")


def test_the_gate_passes_a_fix_of_the_outlet(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    assert gate(_task(), _diff(repo), _base(repo)) == []


def test_the_gate_reads_the_fix_without_importing_it(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    patched = patched_newspapers(_diff(repo), _base(repo))
    assert STUB_XPATH in patched
    assert BBC_OLD_EXTRACTOR not in patched


def test_the_gate_refuses_an_empty_patch():
    assert gate(_task(), "", "") == ["the patch is empty"]


@pytest.mark.parametrize(
    "path",
    [
        "tests/selfheal_check.py",
        "pyproject.toml",
        ".github/workflows/selfheal.yml",
        "tests/assets/html/foxnews.com/2026-10-05.html",
        "tests/assets/html/bbc.com/20261005.html",
        "tests/snapshots/test_parse/bbc.com/2026-10-05.json",
    ],
)
def test_the_gate_refuses_files_outside_the_outlet(tmp_path, path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text("x\n")
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any(path in p for p in problems), problems


def test_the_gate_refuses_editing_an_existing_page(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    (repo / "tests/assets/html/bbc.com/2023-05-20.html").write_text("<html/>\n")
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("2023-05-20.html" in p for p in problems), problems


def test_the_gate_refuses_deleting_a_file(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    (repo / "tests/assets/html/bbc.com/2023-05-20.html").unlink()
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("delete" in p for p in problems), problems


def test_the_gate_refuses_a_symlink(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    page = repo / "tests/assets/html/bbc.com/2026-10-05.html"
    page.unlink()
    page.symlink_to("/etc/passwd")
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("120000" in p for p in problems), problems


def test_the_gate_refuses_a_mode_change(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    (repo / NEWSPAPERS).chmod(0o755)
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("mode change" in p for p in problems), problems


def test_the_gate_refuses_a_rename(tmp_path):
    repo = _repo(tmp_path)
    patch = (
        "diff --git a/tests/assets/html/bbc.com/2023-05-20.html "
        "b/tests/selfheal_check.py\n"
        "similarity index 100%\n"
        "rename from tests/assets/html/bbc.com/2023-05-20.html\n"
        "rename to tests/selfheal_check.py\n"
    )
    problems = gate(_task(), patch, _base(repo))
    assert any("rename" in p for p in problems), problems


def test_the_gate_refuses_a_patch_without_git_headers(tmp_path):
    repo = _repo(tmp_path)
    patch = (
        "--- a/tests/selfheal_check.py\n"
        "+++ b/tests/selfheal_check.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    problems = gate(_task(), patch, _base(repo))
    assert any("tests/selfheal_check.py" in p for p in problems), problems


def test_the_gate_refuses_editing_another_outlet(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    source = (repo / NEWSPAPERS).read_text()
    (repo / NEWSPAPERS).write_text(
        source.replace("expected_headlines=180", "expected_headlines=181")
    )
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("changes other outlets" in p for p in problems), problems


def test_the_gate_refuses_a_page_of_another_day(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo, day="2099-01-01")
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("2099-01-01" in p for p in problems), problems


def test_the_gate_refuses_another_from_date(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo, extractor=_dated_bbc("datetime(2099, 1, 1, tzinfo=timezone.utc)"))
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("from_date" in p for p in problems), problems


def test_the_gate_refuses_a_fix_without_a_dated_xpath(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo, extractor=f'get_xpath_fn("{STUB_XPATH}", href_xpath="a"),')
    problems = gate(_task(), _diff(repo), _base(repo))
    assert any("from_date" in p for p in problems), problems


def test_the_gate_refuses_a_patch_that_does_not_apply(tmp_path):
    repo = _repo(tmp_path)
    _fix_bbc(repo)
    patch = _diff(repo)
    base = _base(repo).replace(
        BBC_OLD_EXTRACTOR, "get_xpath_fn('//x', href_xpath='a'),"
    )
    problems = gate(_task(), patch, base)
    assert any("does not apply" in p for p in problems), problems


# --- the agent ----------------------------------------------------------------


def _assistant(cost: float, stop: str = "toolUse", **extra) -> str:
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "..."}],
        "stopReason": stop,
        "usage": {
            "input": 1000,
            "output": 100,
            "cacheRead": 900,
            "cacheWrite": 0,
            "totalTokens": 2000,
            "cost": {"total": cost},
        },
        **extra,
    }
    return json.dumps({"type": "message_end", "message": message})


def test_the_session_summary_adds_up_every_turn():
    events = "\n".join(
        [
            json.dumps({"type": "session", "id": "x"}),
            json.dumps({"type": "turn_start"}),
            json.dumps({"type": "message_end", "message": {"role": "user"}}),
            _assistant(0.01),
            json.dumps({"type": "turn_end"}),
            json.dumps({"type": "turn_start"}),
            _assistant(0.02, stop="stop"),
            json.dumps({"type": "turn_end"}),
            "not json",
        ]
    )
    assert summarise_session(events) == {
        "turns": 2,
        "input_tokens": 2000,
        "cache_read_tokens": 1800,
        "output_tokens": 200,
        "cost_usd": 0.03,
        "stop_reason": "stop",
        "error": None,
    }


def test_the_session_summary_keeps_the_last_error():
    events = _assistant(0.0, stop="error", errorMessage="No API key for provider")
    summary = summarise_session(events)
    assert summary["stop_reason"] == "error"
    assert summary["error"] == "No API key for provider"


def test_omp_runs_hermetically_on_the_prompt():
    cmd = omp_command("omp", "deepseek/deepseek-flash", "15m", "Fix it.")
    assert cmd[0] == "omp"
    assert cmd[-1] == "Fix it."
    for flag in (
        "-p",
        "--no-session",
        "--no-skills",
        "--no-rules",
        "--no-extensions",
        "--auto-approve",
    ):
        assert flag in cmd
    assert cmd[cmd.index("--mode") + 1] == "json"
    assert cmd[cmd.index("--tools") + 1] == "read,edit,bash"
    assert cmd[cmd.index("--model") + 1] == "deepseek/deepseek-flash"
    assert cmd[cmd.index("--max-time") + 1] == "15m"


def _task(slug: str = "bbc.com", day: str = "2026-10-05") -> dict[str, object]:
    return {
        "outlet": "BBC",
        "slug": slug,
        "url": "https://bbc.com",
        "today": day,
        "expected_headlines": 47,
        "headlines_today": 0,
        "first_bad": "2026-10-04",
        "last_good": None,
        "from_date": "2026-10-04T00:00:00+00:00",
        "fixture": f"tests/assets/html/{slug}/{day}.html",
        "snapshot": f"tests/snapshots/test_parse/{slug}/{day}.txt",
        "extractor": "Newspaper(...)",
    }


def test_the_prompt_names_the_task_and_its_files():
    prompt = build_prompt(_task(), Path("selfheal/bbc.com"))
    for wanted in (
        "BBC",
        "selfheal/bbc.com/today.html",
        "selfheal/bbc.com/task.json",
        "selfheal/bbc.com/last_good.json",
        "selfheal/bbc.com/result.json",
        "tests/assets/html/bbc.com/2026-10-05.html",
        "datetime(2026, 10, 4, tzinfo=timezone.utc)",
        "poe selfheal-check bbc.com --page selfheal/bbc.com/today.html",
        "python -m tests.selfheal_fix snapshot bbc.com",
    ):
        assert wanted in prompt, wanted
    assert "$" not in prompt


@pytest.mark.parametrize(
    ("written", "status"),
    [
        (
            {
                "status": "fixed",
                "old_xpath": "//h3",
                "new_xpath": "//h2",
                "from_date": "2026-10-04T00:00:00+00:00",
                "explanation": "The site moved to h2.",
            },
            "fixed",
        ),
        ({"status": "gave_up", "explanation": "No headlines in the page."}, "gave_up"),
        ({"status": "done"}, "gave_up"),
        (None, "gave_up"),
    ],
)
def test_the_agent_result_is_read_or_defaults_to_giving_up(tmp_path, written, status):
    path = tmp_path / "result.json"
    if written is not None:
        path.write_text(json.dumps(written))
    result = read_result(path)
    assert result["status"] == status
    assert set(result) == {
        "status",
        "old_xpath",
        "new_xpath",
        "from_date",
        "explanation",
    }
    if written is None:
        assert "no result.json" in str(result["explanation"])


def test_a_long_explanation_is_cut_short(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "gave_up", "explanation": "x" * 5000}))
    assert len(str(read_result(path)["explanation"])) <= 1000


#: A Saturday noon: off-peak whatever the hour.
OFF_PEAK = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)

SECRET = "sk-not-a-real-key-0123456789"  # noqa: S105 (a fake key)


def test_run_agent_turns_the_session_into_a_patch_and_a_result(tmp_path):
    repo = _repo(tmp_path)
    inputs = repo / "selfheal" / "bbc.com"
    inputs.mkdir(parents=True)
    (inputs / "today.html").write_text("<html>today</html>\n")
    (inputs / "task.json").write_text(json.dumps(_task()))
    out = tmp_path / "out"
    calls = []

    def fake_run(cmd: list[str], cwd: Path) -> tuple[int, str]:
        calls.append(cmd)
        if cmd[0] == "omp":
            _fix_bbc(cwd)
            (cwd / "selfheal/bbc.com/result.json").write_text(
                json.dumps({"status": "fixed", "explanation": f"key {SECRET}"})
            )
            return 0, _assistant(0.04, stop="stop", leaked=SECRET) + "\n"
        return 0, ""

    result = run_agent(
        "bbc.com",
        root=repo,
        inputs=inputs,
        out_dir=out,
        omp="omp",
        model="deepseek/deepseek-flash",
        max_time="15m",
        run=fake_run,
        redact_secrets=[SECRET],
        now=OFF_PEAK,
    )

    for kept in out.iterdir():
        assert SECRET not in kept.read_text(), kept.name
    assert "key ***" in result["explanation"]
    page = repo / "tests/assets/html/bbc.com/2026-10-05.html"
    assert page.read_text() == "<html>today</html>\n"
    assert calls[0][0] == "omp"
    assert calls[1][-3:] == ["tests.selfheal_fix", "snapshot", "bbc.com"]
    patch = (out / "patch.diff").read_text()
    assert STUB_XPATH in patch
    assert "selfheal/" not in patch
    assert gate(_task(), patch, _base(repo)) == []
    assert result["status"] == "fixed"
    assert result["outlet"] == "bbc.com"
    assert result["session"]["cost_usd"] == 0.04
    assert result["omp_exit_code"] == 0
    assert json.loads((out / "result.json").read_text()) == result
    assert (out / "session.jsonl").read_text().startswith('{"type": "message_end"')


def test_run_agent_records_a_failed_session(tmp_path):
    repo = _repo(tmp_path)
    inputs = repo / "selfheal" / "bbc.com"
    inputs.mkdir(parents=True)
    (inputs / "today.html").write_text("<html>today</html>\n")
    (inputs / "task.json").write_text(json.dumps(_task()))

    def fake_run(cmd: list[str], cwd: Path) -> tuple[int, str]:
        if cmd[0] == "omp":
            return 1, _assistant(0.0, stop="error", errorMessage="boom") + "\n"
        return 0, ""

    result = run_agent(
        "bbc.com",
        root=repo,
        inputs=inputs,
        out_dir=tmp_path / "out",
        omp="omp",
        model="m",
        max_time="1m",
        run=fake_run,
        now=OFF_PEAK,
    )
    assert result["status"] == "gave_up"
    assert result["omp_exit_code"] == 1
    assert result["session"]["error"] == "boom"


# --- DeepSeek's off-peak hours ------------------------------------------------


def _utc(day: int, hour: int, minute: int = 0) -> datetime:
    """Return a time in the week of Monday 2026-10-05."""
    return datetime(2026, 10, day, hour, minute, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("now", "left"),
    [
        # Monday: peak 01:00-04:00 and 06:00-10:00
        (_utc(5, 0, 30), timedelta(minutes=30)),
        (_utc(5, 1), None),
        (_utc(5, 3, 59), None),
        (_utc(5, 4), timedelta(hours=2)),
        (_utc(5, 9, 59), None),
        (_utc(5, 10), timedelta(hours=15)),
        (_utc(5, 16, 30), timedelta(hours=8, minutes=30)),
        # Friday evening runs until Monday's first peak
        (_utc(9, 10), timedelta(days=2, hours=15)),
        # weekends are off-peak all day
        (_utc(10, 3), timedelta(days=1, hours=22)),
        (_utc(11, 8), timedelta(hours=17)),
    ],
)
def test_off_peak_left(now, left):
    assert off_peak_left(now) == left


def test_off_peak_left_needs_an_aware_time():
    with pytest.raises(ValueError, match="UTC"):
        off_peak_left(datetime(2026, 10, 5, 12))  # noqa: DTZ001


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("90", timedelta(seconds=90)),
        ("45s", timedelta(seconds=45)),
        ("15m", timedelta(minutes=15)),
        ("1h", timedelta(hours=1)),
    ],
)
def test_duration_reads_omp_max_time(text, parsed):
    assert duration(text) == parsed


def test_duration_refuses_what_omp_would_not_read():
    with pytest.raises(ValueError, match="5d"):
        duration("5d")


def _agent_inputs(tmp_path: Path) -> tuple[Path, Path]:
    repo = _repo(tmp_path)
    inputs = repo / "selfheal" / "bbc.com"
    inputs.mkdir(parents=True)
    (inputs / "today.html").write_text("<html>today</html>\n")
    (inputs / "task.json").write_text(json.dumps(_task()))
    return repo, inputs


@pytest.mark.parametrize(
    "now",
    [
        _utc(5, 2),  # in a peak window
        _utc(5, 0, 50),  # 10 minutes before one: a 15-minute session would cross it
    ],
)
def test_run_agent_defers_a_session_that_would_touch_peak_hours(tmp_path, now):
    repo, inputs = _agent_inputs(tmp_path)
    out = tmp_path / "out"

    def fake_run(cmd: list[str], cwd: Path) -> tuple[int, str]:
        pytest.fail(f"nothing should run, but {cmd[0]} did")

    result = run_agent(
        "bbc.com",
        root=repo,
        inputs=inputs,
        out_dir=out,
        omp="omp",
        model="m",
        max_time="15m",
        run=fake_run,
        now=now,
    )

    assert result["status"] == "deferred"
    assert "off-peak" in result["explanation"]
    assert (out / "patch.diff").read_text() == ""
    assert json.loads((out / "result.json").read_text()) == result
    assert not (repo / "tests/assets/html/bbc.com/2026-10-05.html").exists()


def _crons(workflow: Path) -> list[str]:
    cron = re.compile(r"""^\s*- cron: ["']([^"']+)["']""", re.MULTILINE)
    return cron.findall(workflow.read_text())


def test_the_daily_update_leaves_hours_of_off_peak_for_the_fix_job():
    """Step 4 chains the fix job to the daily Update, so Update has to run
    off-peak with time to spare, on every day of the week."""
    daily = [
        c for c in _crons(_ROOT / ".github/workflows/update.yml") if c.endswith("* * *")
    ]
    assert daily, "no daily Update schedule"
    for cron in daily:
        minute, hour = (int(field) for field in cron.split()[:2])
        for day in range(5, 12):
            left = off_peak_left(_utc(day, hour, minute))
            assert left is not None, (cron, day)
            assert left >= timedelta(hours=4), (cron, day, left)


def test_the_fix_job_is_never_scheduled_in_peak_hours():
    for cron in _crons(_ROOT / ".github/workflows/selfheal.yml"):
        minute, hour = (int(field) for field in cron.split()[:2])
        for day in range(5, 12):
            assert off_peak_left(_utc(day, hour, minute)) is not None, (cron, day)
