# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`poe selfheal-fix`: the fix job of the self-fix loop, one outlet at a time.

The `Self-fix` workflow runs it in three steps, each in its own job:

`prepare`
    writes, for every outlet the day's health report calls `xpath` (or the one
    asked for), the agent's inputs under `<out-dir>/<slug>/`: `today.html`,
    the page the Update run fetched; `task.json`, what is broken and since
    when; `last_good.json`, what a real headline on the site looks like.
`agent`
    adds today's page as a fixture, runs omp on the fixed prompt in
    `tests/selfheal_prompt.md`, then writes `patch.diff`, `result.json` and
    the session's events to `--out-dir`. It holds the DeepSeek key and nothing
    else; the agent works offline on the saved page.
`gate`
    is the first thing a trusted checkout does with a patch, before any of its
    code runs: it refuses a patch touching anything but the outlet's entry in
    `newspapers.py` and its new page and snapshot. Only then is the patch
    applied and judged by `poe selfheal-check`.

`agent` also tells the harness failing from the agent giving up. A session
that left no result after a provider error (DeepSeek's `402 Insufficient
Balance`, say) or a failing exit is `infra_error`, not `gave_up`, with an
empty patch, and the command exits 1; so is one the balance endpoint says
DeepSeek will refuse, which is asked before omp starts.

`off-peak`
    fails unless DeepSeek stays off-peak for `--needs` more; `agent` does the
    same check before omp starts, and defers its outlet if it fails.

`snapshot` (re)writes the parse snapshot of an outlet's newest page; the agent
runs it after each edit, and `agent` once more at the end.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from string import Template
from typing import Any, Protocol

from tests.fixtures import (
    HTML_ROOT,
    Fixture,
    latest_fixture,
    slug_of,
)
from tests.selfheal_check import (
    HTML_DIR,
    NEWSPAPERS_PY,
    SNAPSHOT_DIR,
    Change,
    _entry,
    _layout,
    _parse,
    check_scope,
    resolve_outlet,
)
from toxic_news.health import Verdict, read_health
from toxic_news.newspapers import Newspaper, newspapers
from toxic_news.store import headlines_path, read_headlines
from toxic_news.triage import GitHub, label_of

_ROOT = HTML_ROOT.parents[2]

#: How far back to look for the day an outlet broke.
LOOKBACK_DAYS = 60


# --- prepare ------------------------------------------------------------------


def broken_outlets(data_dir: Path, today: date) -> list[Newspaper]:
    """Return the outlets `today`'s health report calls XPath errors."""
    broken = {
        o.newspaper for o in read_health(data_dir, today) if o.verdict == Verdict.XPATH
    }
    return [n for n in newspapers if n.name in broken]


#: An outlet gets at most one fix session in this long, whatever came of it.
ATTEMPT_EVERY = timedelta(days=7)
#: Who posts the attempt markers; anyone can comment on a public repository.
BOT_LOGIN = "github-actions[bot]"
_ATTEMPT = re.compile(r"<!-- selfheal-attempt (\d{4}-\d{2}-\d{2}) -->")


def attempt_marker(day: date) -> str:
    """Return the hidden marker an issue comment records an attempt on `day` with."""
    return f"<!-- selfheal-attempt {day.isoformat()} -->"


def infra_marker(day: date) -> str:
    """Return the hidden marker an issue comment records an infrastructure failure with.

    Unlike `attempt_marker`, it holds nothing back: a failure of DeepSeek or
    the harness says nothing about the outlet, so the next run tries it again.
    """
    return f"<!-- selfheal-infra-error {day.isoformat()} -->"


@dataclass(frozen=True)
class Record:
    """What GitHub knows of an outlet's fixes: its last attempt and open PR."""

    last_attempt: date | None = None
    open_pr: int | None = None


class Labelled(Protocol):
    """The GitHub calls `record_of` needs."""

    def labelled(self, label: str) -> list[dict[str, Any]]:
        """Return the issues and pull requests carrying `label`, in any state."""
        ...

    def comments(self, number: int) -> list[dict[str, Any]]:
        """Return the comments of an issue."""
        ...


def record_of(github: Labelled, newspaper: Newspaper) -> Record:
    """Read `newspaper`'s last attempt and open selfheal PR off its label."""
    items = github.labelled(label_of(str(newspaper.url)))
    open_prs = [
        i["number"] for i in items if "pull_request" in i and i["state"] == "open"
    ]
    attempts = [
        date.fromisoformat(day)
        for item in items
        if "pull_request" not in item
        for comment in github.comments(item["number"])
        if comment["user"]["login"] == BOT_LOGIN
        for day in _ATTEMPT.findall(comment["body"] or "")
    ]
    return Record(
        last_attempt=max(attempts, default=None),
        open_pr=min(open_prs, default=None),
    )


def _report_before(data_dir: Path, today: date) -> date | None:
    """Return the day of the newest health report before `today`, if any."""
    days = [
        date(int(path.parts[-3]), int(path.parts[-2]), int(path.stem))
        for path in (data_dir / "health").glob("*/*/*.json")
    ]
    return max((d for d in days if d < today), default=None)


def select(
    outlets: Sequence[Newspaper],
    *,
    data_dir: Path,
    today: date,
    records: Callable[[Newspaper], Record],
    forced: bool,
) -> tuple[list[Newspaper], list[str]]:
    """Return which of the broken `outlets` get a session, and why the rest wait.

    An outlet waits while a selfheal PR of it is open. Unless `forced`, it also
    waits until it was `xpath` in the run before today's too, and for a week
    after its last attempt, whatever came of it.
    """
    before = _report_before(data_dir, today)
    broken_before = (
        set() if before is None else {n.name for n in broken_outlets(data_dir, before)}
    )
    chosen, skipped = [], []
    for newspaper in outlets:
        record = records(newspaper)
        last = record.last_attempt
        if record.open_pr is not None:
            skipped.append(
                f"{newspaper.name}: selfheal PR #{record.open_pr} is still open"
            )
        elif forced:
            chosen.append(newspaper)
        elif newspaper.name not in broken_before:
            skipped.append(
                f"{newspaper.name}: broken in one run only; waiting for a second"
            )
        elif last is not None and today - last < ATTEMPT_EVERY:
            skipped.append(f"{newspaper.name}: attempted on {last}; at most one a week")
        else:
            chosen.append(newspaper)
    return chosen, skipped


@dataclass(frozen=True)
class History:
    """When an outlet broke: its first bad day, and the good day before it."""

    first_bad: date
    last_good: date | None


def history(
    data_dir: Path,
    newspaper: Newspaper,
    today: date,
    lookback_days: int = LOOKBACK_DAYS,
) -> History:
    """Return when `newspaper`, broken on `today`, broke.

    Walks the health reports back from `today` to the last `ok` day; the first
    bad day is the earliest `xpath` one after it. Days without a report, or on
    which the fetch itself failed, neither end the streak nor start it.
    """
    first_bad = today
    for back in range(lookback_days + 1):
        day = today - timedelta(days=back)
        verdicts = [
            o.verdict
            for o in read_health(data_dir, day)
            if o.newspaper == newspaper.name
        ]
        if Verdict.OK in verdicts:
            return History(first_bad=first_bad, last_good=day)
        if Verdict.XPATH in verdicts:
            first_bad = day
    return History(first_bad=first_bad, last_good=None)


def last_good_headlines(
    data_dir: Path, newspaper: Newspaper, last_good: date | None
) -> dict[str, Any]:
    """Return the headlines of `newspaper`'s last good day.

    They come from the `data` branch; an outlet broken since before it began
    falls back to the snapshot of its newest recorded page.
    """
    if last_good is not None:
        rows = [
            h
            for h in read_headlines(data_dir, last_good)
            if h.newspaper == newspaper.name
        ]
        if rows:
            path = headlines_path(Path(), last_good).as_posix()
            return {
                "date": last_good.isoformat(),
                "source": f"data branch, {path}",
                "headlines": [{"text": h.text, "url": str(h.url)} for h in rows],
            }
    fixture = latest_fixture(slug_of(newspaper))
    snapshot = json.loads(fixture.snapshot_path.read_text())
    return {
        "date": fixture.date.isoformat(),
        "source": fixture.snapshot_path.relative_to(_ROOT).as_posix(),
        "headlines": [{"text": text, "url": url} for text, url in snapshot],
    }


def extractor_source(newspaper: Newspaper, source: str) -> str:
    """Return the source of `newspaper`'s `Newspaper(...)` entry in `source`."""
    entry = _entry(ast.parse(source), slug_of(newspaper))
    if entry is None:
        msg = f"No entry for {newspaper.name} in {NEWSPAPERS_PY}"
        raise SystemExit(msg)
    return ast.get_source_segment(source, entry) or ""


def write_inputs(
    newspaper: Newspaper,
    *,
    raw_html_dir: Path,
    data_dir: Path,
    today: date,
    out_dir: Path,
) -> Path:
    """Write the agent's inputs for `newspaper` to `<out_dir>/<slug>/`."""
    slug = slug_of(newspaper)
    page = raw_html_dir / f"{slug}.html"
    if not page.is_file():
        msg = f"The Update run saved no page for {newspaper.name}: no {page}"
        raise SystemExit(msg)
    found = history(data_dir, newspaper, today)
    health = [o for o in read_health(data_dir, today) if o.newspaper == newspaper.name]
    task = {
        "outlet": newspaper.name,
        "slug": slug,
        "url": str(newspaper.url),
        "today": today.isoformat(),
        "expected_headlines": newspaper.expected_headlines,
        "headlines_today": health[0].headlines if health else None,
        "first_bad": found.first_bad.isoformat(),
        "last_good": None if found.last_good is None else found.last_good.isoformat(),
        "from_date": datetime.combine(
            found.first_bad, time(tzinfo=timezone.utc)
        ).isoformat(),
        "fixture": str(HTML_DIR / slug / f"{today.isoformat()}.html"),
        "snapshot": str(SNAPSHOT_DIR / slug / f"{today.isoformat()}.txt"),
        "extractor": extractor_source(newspaper, (_ROOT / NEWSPAPERS_PY).read_text()),
    }
    inputs = out_dir / slug
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(page, inputs / "today.html")
    (inputs / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    good = last_good_headlines(data_dir, newspaper, found.last_good)
    (inputs / "last_good.json").write_text(json.dumps(good, indent=2) + "\n")
    return inputs


# --- gate ---------------------------------------------------------------------


#: Patches are read and written as bytes and held as text decoded like this,
#: which gives back the same bytes: a saved page keeps its CRLF endings and
#: any bytes that are not UTF-8, so the page the fix adds is the page fetched.
_PATCH_ENCODING = "utf-8"
_PATCH_ERRORS = "surrogateescape"


def read_patch(path: Path) -> str:
    """Return the patch at `path`, byte for byte."""
    return path.read_bytes().decode(_PATCH_ENCODING, _PATCH_ERRORS)


def write_patch(path: Path, patch: str) -> None:
    """Write `patch` to `path`, byte for byte."""
    path.write_bytes(patch.encode(_PATCH_ENCODING, _PATCH_ERRORS))


def _git_apply(args: list[str], patch: str, cwd: Path) -> tuple[int, str, str]:
    """Run `git apply` on `patch` in `cwd`; return its exit code, stdout, stderr.

    Its warnings (say, the trailing whitespace a saved page is full of) go to
    stderr, apart from what it reports.
    """
    git = shutil.which("git")
    if git is None:
        msg = "git is needed to read the patch"
        raise SystemExit(msg)
    done = subprocess.run(  # noqa: S603 (git apply, reading the patch from stdin)
        [git, "apply", *args, "-"],
        input=patch.encode(_PATCH_ENCODING, _PATCH_ERRORS),
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return (
        done.returncode,
        done.stdout.decode(_PATCH_ENCODING, _PATCH_ERRORS),
        done.stderr.decode(_PATCH_ENCODING, "replace"),
    )


def patched_newspapers(patch: str, base_source: str) -> str:
    """Return `newspapers.py` with `patch` applied, without importing any of it.

    Raises `ValueError` if the patch does not apply to `base_source`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / NEWSPAPERS_PY
        path.parent.mkdir(parents=True)
        path.write_text(base_source)
        code, _, errors = _git_apply([f"--include={NEWSPAPERS_PY}"], patch, Path(tmp))
        if code != 0:
            raise ValueError(errors.strip())
        return path.read_text()


#: The only summary line `git apply --summary` may print for a fix: a new,
#: plain file. Renames, copies, deletions and mode changes are all refused.
_CREATE = "create mode 100644 "


def _touched(patch: str) -> tuple[list[Change], list[str]]:
    """Return what `patch` changes, as git would apply it, and what it may not."""
    with tempfile.TemporaryDirectory() as tmp:
        code, output, errors = _git_apply(["--numstat", "--summary"], patch, Path(tmp))
    if code != 0:
        return [], [f"the patch does not parse: {errors.strip()}"]
    created, problems, paths = set(), [], []
    for line in output.splitlines():
        if line.startswith(" "):
            summary = line.strip()
            if summary.startswith(_CREATE):
                created.add(summary.removeprefix(_CREATE))
            else:
                problems.append(f"`{summary}` is not allowed")
        elif line:
            path = line.split("\t", 2)[-1]
            if path.startswith('"') or "=>" in path:
                problems.append(f"`{path}` is not allowed")
            else:
                paths.append(path)
    changes = [Change(p, "A" if p in created else "M") for p in paths]
    return changes, problems


def _from_dates(source: str, slug: str) -> list[str]:
    """Return the `from_date` of every `DatedXpath` in `slug`'s own code."""
    layout = _layout(ast.parse(source), slug)
    own = [layout.entry, *layout.helpers] if layout.entry is not None else []
    return [
        ast.unparse(keyword.value)
        for node in own
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "DatedXpath"
        for keyword in call.keywords
        if keyword.arg == "from_date"
    ]


def _expected_from_date(task: dict[str, Any]) -> str:
    day = datetime.fromisoformat(task["from_date"])
    return f"datetime({day.year}, {day.month}, {day.day}, tzinfo=timezone.utc)"


def gate(task: dict[str, Any], patch: str, base_source: str) -> list[str]:
    """Return why `patch` may not be applied as the fix `task` asked for.

    What the patch touches is read by git itself, the way it would apply it.
    It has to add exactly the task's page and snapshot, keep to
    `selfheal-check`'s scope rule, and add one `DatedXpath` from the task's
    `from_date`. `newspapers.py` is only read as text: none of the patch's
    code runs here. Empty means it may be applied.
    """
    if not patch.strip():
        return ["the patch is empty"]
    slug = task["slug"]
    changes, problems = _touched(patch)
    if not changes:
        return problems or ["the patch changes nothing"]
    wanted = {Change(task["fixture"], "A"), Change(task["snapshot"], "A")}
    problems += [
        f"{c.path} ({c.status}) is not today's page or snapshot"
        for c in changes
        if c not in wanted and c.path != NEWSPAPERS_PY
    ]
    problems += [f"{c.path} is not added" for c in sorted(wanted - set(changes))]
    try:
        new_source = patched_newspapers(patch, base_source)
    except ValueError as e:
        return [*problems, f"the patch does not apply to {NEWSPAPERS_PY}: {e}"]
    scope = check_scope(slug, changes, base_source, new_source)
    if not scope.passed:
        problems.append(scope.detail)
    elif (
        added := sorted(
            # `None` dates the old XPath, when the fix makes the entry dated
            set(_from_dates(new_source, slug))
            - set(_from_dates(base_source, slug))
            - {"None"}
        )
    ) != [expected := _expected_from_date(task)]:
        problems.append(
            f"the fix adds DatedXpath from_date {', '.join(added) or 'none'}, "
            f"expected {expected}"
        )
    return problems


# --- snapshot -----------------------------------------------------------------


def write_snapshot(newspaper: Newspaper) -> Fixture:
    """Write the parse snapshot of `newspaper`'s newest page, as `test_parse` would."""
    fixture = latest_fixture(slug_of(newspaper))
    fixture.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.snapshot_path.write_text(json.dumps(_parse(newspaper, fixture), indent=2))
    return fixture


# --- agent --------------------------------------------------------------------

#: The fixed prompt the agent works from, filled in with its task.
PROMPT = Path(__file__).with_name("selfheal_prompt.md")

#: The tools the agent gets: no web search, no subagents, no browser.
TOOLS = "read,edit,bash"

#: Runs a command in a directory, returning its exit code and standard output.
DirRunner = Callable[[list[str], Path], tuple[int, str]]


def _run_in(cmd: list[str], cwd: Path) -> tuple[int, str]:
    done = subprocess.run(  # noqa: S603 (omp and this module, as the fix job runs them)
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    return done.returncode, done.stdout


def omp_command(omp: str, model: str, max_time: str, prompt: str) -> list[str]:
    """Return the omp command line that runs one fix session on `prompt`.

    It prints the session's events as JSON lines, keeps no session on disk,
    and loads no skills, rules or extensions, so the prompt is all it is told.
    """
    return [
        omp,
        "-p",
        "--mode",
        "json",
        "--model",
        model,
        "--tools",
        TOOLS,
        "--max-time",
        max_time,
        "--no-session",
        "--no-title",
        "--no-skills",
        "--no-rules",
        "--no-extensions",
        "--no-lsp",
        "--auto-approve",
        prompt,
    ]


def build_prompt(task: dict[str, object], inputs: Path) -> str:
    """Fill the fixed prompt in with `task`, whose inputs are under `inputs`."""
    from_date = datetime.fromisoformat(str(task["from_date"]))
    return Template(PROMPT.read_text()).substitute(
        outlet=task["outlet"],
        url=task["url"],
        slug=task["slug"],
        first_bad=task["first_bad"],
        expected=task["expected_headlines"],
        fixture=task["fixture"],
        snapshot=task["snapshot"],
        inputs=inputs.as_posix(),
        from_date=task["from_date"],
        from_date_args=f"{from_date.year}, {from_date.month}, {from_date.day}",
    )


def summarise_session(events: str) -> dict[str, object]:
    """Return the turns, tokens and cost of a session from omp's JSON events."""
    turns = input_tokens = cache_read = output_tokens = 0
    cost = 0.0
    stop_reason = error = None
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn_end":
            turns += 1
        message = event.get("message")
        if event.get("type") != "message_end" or not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        input_tokens += usage.get("input", 0)
        cache_read += usage.get("cacheRead", 0)
        output_tokens += usage.get("output", 0)
        cost += (usage.get("cost") or {}).get("total", 0.0)
        stop_reason = message.get("stopReason")
        error = message.get("errorMessage")
    return {
        "turns": turns,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "stop_reason": stop_reason,
        "error": error,
    }


#: The longest explanation kept from the agent; it is asked for five sentences.
MAX_EXPLANATION_CHARS = 1000

_RESULT_FIELDS = ("old_xpath", "new_xpath", "from_date")

#: What it says it left out of the headlines, and why, kept for the reviewer.
_EXCLUDED = "excluded"


def _written_result(path: Path) -> dict[str, Any] | None:
    """Return what the agent wrote at `path`, or None if it left no usable result."""
    try:
        written = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(written, dict) or written.get("status") not in {
        "fixed",
        "gave_up",
    }:
        return None
    return written


def read_result(path: Path) -> dict[str, str | None]:
    """Return the agent's account of its fix, or a `gave_up` if it left none.

    Only its status and XPaths are used; the headlines the fix finds are
    always recomputed from the patch.
    """
    written = _written_result(path)
    if written is None:
        reason = "no result.json" if not path.exists() else "an invalid result.json"
        return {
            "status": "gave_up",
            **dict.fromkeys(_RESULT_FIELDS),
            "explanation": f"The agent left {reason}.",
            _EXCLUDED: None,
        }
    result: dict[str, str | None] = {"status": written["status"]}
    for key in _RESULT_FIELDS:
        value = written.get(key)
        result[key] = None if value is None else str(value)
    explanation = str(written.get("explanation") or "")
    result["explanation"] = explanation[:MAX_EXPLANATION_CHARS]
    excluded = written.get(_EXCLUDED)
    result[_EXCLUDED] = (
        None if excluded is None else str(excluded)[:MAX_EXPLANATION_CHARS]
    )
    return result


# --- infrastructure failures ----------------------------------------------------

#: The status of a session that failed for want of DeepSeek or the harness, as
#: opposed to `gave_up`, the agent's own verdict on the outlet.
INFRA_ERROR = "infra_error"

#: DeepSeek's balance endpoint; asking is no model call.
BALANCE_URL = "https://api.deepseek.com/user/balance"
#: The longest provider error kept; the rest is noise.
MAX_ERROR_CHARS = 200
_REQUEST_ID = re.compile(r"\s*\(request_id: [^)]*\)")

HTTP_PAYMENT_REQUIRED = 402
HTTP_UNAUTHORIZED = 401

#: Asks the balance endpoint with an API key; returns the HTTP status and body.
BalanceFetcher = Callable[[str], tuple[int, str]]


def _fetch_balance(key: str) -> tuple[int, str]:
    request = urllib.request.Request(
        BALANCE_URL, headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as reply:  # noqa: S310 (a fixed https URL)
            return reply.status, reply.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def balance_problem(key: str, fetch: BalanceFetcher = _fetch_balance) -> str | None:
    """Return why DeepSeek will refuse `key`'s calls, or None if it may work.

    Only a clear no counts: DeepSeek saying the balance is not available, or
    refusing the key. Any other answer, or none, leaves the session to try.
    """
    if not key:
        return None
    try:
        status, body = fetch(key)
    except OSError:
        return None
    refused = {
        HTTP_PAYMENT_REQUIRED: "DeepSeek API: 402 Insufficient Balance",
        HTTP_UNAUTHORIZED: "DeepSeek API: 401, the API key was refused",
    }
    if status in refused:
        return refused[status]
    try:
        available = json.loads(body).get("is_available")
    except (ValueError, AttributeError):
        return None
    return (
        "DeepSeek API: the account has no balance available"
        if available is False
        else None
    )


#: A session with more turns than this really ran, so it is no infra failure.
MAX_INFRA_TURNS = 1
#: A provider error that is DeepSeek's or the network's fault, never the agent's:
#: it is infrastructure after any number of turns.
_PROVIDER_FAULT = re.compile(
    r"\b(?:401|402|403|429|5\d\d)\b|insufficient balance|rate.?limit", re.IGNORECASE
)


def infra_problem(session: dict[str, object], code: int) -> str | None:
    """Return what went wrong when a session that left no result failed to run.

    `session` is `summarise_session`'s summary and `code` omp's exit code. A
    provider error or a failing exit says the harness broke, but only while the
    agent had not really run: once it has taken turns, a timeout or an overflow
    is the agent failing, which stays a give-up, unless the provider error is
    a fault of DeepSeek's (a 402, 5xx, auth or rate limit).
    The error is cut to its first line,
    with DeepSeek's request id dropped, and is redacted by the caller.
    """
    turns = session.get("turns")
    error = session.get("error")
    provider_fault = bool(_PROVIDER_FAULT.search(str(error or "")))
    if isinstance(turns, int) and turns > MAX_INFRA_TURNS and not provider_fault:
        return None
    if error or session.get("stop_reason") == "error":
        first = _REQUEST_ID.sub("", str(error or "")).strip().splitlines()
        text = first[0][:MAX_ERROR_CHARS] if first else "the session ended in an error"
        return f"The agent session failed: {text}"
    if code != 0:
        return f"omp exited with code {code} before the agent left a result"
    return None


def _infra_result(slug: str, problem: str) -> dict[str, Any]:
    return {
        "outlet": slug,
        "status": INFRA_ERROR,
        **dict.fromkeys(_RESULT_FIELDS),
        "explanation": problem,
        _EXCLUDED: None,
    }


def _git_in(root: Path, *args: str) -> bytes:
    git = shutil.which("git")
    if git is None:
        msg = "git is needed to collect the patch"
        raise SystemExit(msg)
    return subprocess.run(  # noqa: S603 (git, on the fix job's own checkout)
        [git, *args], cwd=root, capture_output=True, check=True
    ).stdout


def collect_patch(root: Path) -> str:
    """Return every change in the working tree of `root`, new files included."""
    _git_in(root, "add", "--intent-to-add", "--all")
    patch = _git_in(root, "diff", "--binary", "--no-renames", "HEAD")
    return patch.decode(_PATCH_ENCODING, _PATCH_ERRORS)


def redact(text: str, secrets: Sequence[str]) -> str:
    """Return `text` with every one of `secrets` in it masked."""
    for secret in secrets:
        if len(secret) >= MIN_SECRET_CHARS:
            text = text.replace(secret, "***")
    return text


#: Shorter "secrets" are not masked: they would mask ordinary text.
MIN_SECRET_CHARS = 8


# --- DeepSeek's off-peak hours ------------------------------------------------

#: DeepSeek's peak hours, in UTC, Monday to Friday: a request then costs twice
#: the off-peak rate. Every other hour is off-peak, weekends included, as
#: https://api-docs.deepseek.com/quick_start/pricing says (checked 2026-10-06).
#: Chinese public holidays are off-peak all day too; this treats them as
#: ordinary weekdays, which is only ever stricter.
PEAK_HOURS_UTC = ((time(1), time(4)), (time(6), time(10)))
PEAK_WEEKDAYS = range(5)


def _peaks(now: datetime) -> Iterator[tuple[datetime, datetime]]:
    """Yield the peak windows from the start of `now`'s day on, in order."""
    day = now.date()
    while True:
        if day.weekday() in PEAK_WEEKDAYS:
            for start, end in PEAK_HOURS_UTC:
                yield (
                    datetime.combine(day, start, tzinfo=timezone.utc),
                    datetime.combine(day, end, tzinfo=timezone.utc),
                )
        day += timedelta(days=1)


def off_peak_left(now: datetime) -> timedelta | None:
    """Return how long DeepSeek stays off-peak after `now`; `None` in peak hours."""
    if now.utcoffset() is None:
        msg = f"{now} has no timezone; give a UTC time"
        raise ValueError(msg)
    for start, end in _peaks(now.astimezone(timezone.utc)):
        if now < start:
            return start - now
        if now < end:
            return None
    msg = "unreachable: there is always a next peak"
    raise AssertionError(msg)


_DURATION = re.compile(r"(\d+)([smh]?)")


def duration(text: str) -> timedelta:
    """Return how long omp's `--max-time` reads `text` as: `600`, `45s`, `15m`..."""
    match = _DURATION.fullmatch(text.strip())
    if match is None:
        msg = f"{text!r} is not a duration like 600, 45s, 15m or 1h"
        raise ValueError(msg)
    unit = {"": "seconds", "s": "seconds", "m": "minutes", "h": "hours"}[match[2]]
    return timedelta(**{unit: int(match[1])})


def peak_problem(now: datetime, needed: timedelta) -> str | None:
    """Return why a run of length `needed` may not start at `now`, if it may not."""
    left = off_peak_left(now)
    if left is None:
        return f"{now:%a %H:%M} UTC is in DeepSeek's peak hours; run off-peak"
    if left < needed:
        return (
            f"DeepSeek's off-peak hours end in {left}, before a run of {needed} "
            "could finish; run after the peak"
        )
    return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def run_agent(  # noqa: PLR0913 (each is an input of the session)
    slug: str,
    *,
    root: Path,
    inputs: Path,
    out_dir: Path,
    omp: str,
    model: str,
    max_time: str,
    run: DirRunner = _run_in,
    redact_secrets: Sequence[str] = (),
    now: datetime | None = None,
    balance: Callable[[], str | None] = lambda: None,
) -> dict[str, Any]:
    """Run one fix session of `slug` in the checkout `root`, and collect it.

    Writes `patch.diff`, `result.json` and `session.jsonl` to `out_dir`, and
    returns the result. The agent can read its environment, so anything it
    leaves is kept with `redact_secrets` masked: artifacts of a public
    repository are public.

    A session that could reach DeepSeek's peak hours, given `max_time`, is not
    started: the result is `deferred`, and the patch empty.

    One that fails for want of DeepSeek or the harness, not the agent's own
    verdict, is `infra_error` with an empty patch: `balance` is asked before
    omp starts and returns why DeepSeek will refuse, or None; and a session
    that left no result after a provider error or a failing exit is one too.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = peak_problem(now or _utcnow(), duration(max_time))
    if problem is not None:
        result = {
            "outlet": slug,
            "status": "deferred",
            **dict.fromkeys(_RESULT_FIELDS),
            "explanation": f"Not started: {problem}.",
            _EXCLUDED: None,
        }
        write_patch(out_dir / "patch.diff", "")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    if (problem := balance()) is not None:
        result = _infra_result(slug, f"Not started: {redact(problem, redact_secrets)}")
        write_patch(out_dir / "patch.diff", "")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    task = json.loads((inputs / "task.json").read_text())
    fixture = root / task["fixture"]
    fixture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs / "today.html", fixture)

    prompt = build_prompt(task, inputs.relative_to(root))
    code, events = run(omp_command(omp, model, max_time, prompt), root)
    (out_dir / "session.jsonl").write_text(redact(events, redact_secrets))
    session = summarise_session(events)
    problem = (
        None
        if _written_result(inputs / "result.json")
        else infra_problem(session, code)
    )
    if problem is not None:
        result = {
            **_infra_result(slug, redact(problem, redact_secrets)),
            "omp_exit_code": code,
            "session": session,
        }
        write_patch(out_dir / "patch.diff", "")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    # whatever the agent left, the snapshot is the one the extractor gives
    snapshot = [sys.executable, "-m", "tests.selfheal_fix", "snapshot", slug]
    snapshot_code, _ = run(snapshot, root)

    write_patch(out_dir / "patch.diff", redact(collect_patch(root), redact_secrets))
    agent_result = read_result(inputs / "result.json")
    result = {
        "outlet": slug,
        **{
            k: None if v is None else redact(v, redact_secrets)
            for k, v in agent_result.items()
        },
        "omp_exit_code": code,
        "snapshot_exit_code": snapshot_code,
        "session": session,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# --- command line -------------------------------------------------------------


def _prepare(args: argparse.Namespace) -> int:
    today = args.date
    forced = bool(args.outlet)
    outlets = (
        [resolve_outlet(args.outlet)]
        if forced
        else broken_outlets(args.data_dir, today)
    )
    waiting: list[str] = []
    if args.repo:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        github = GitHub(args.repo, token=token)
        outlets, waiting = select(
            outlets,
            data_dir=args.data_dir,
            today=today,
            records=lambda newspaper: record_of(github, newspaper),
            forced=forced,
        )
    for reason in waiting:
        sys.stdout.write(f"waits: {reason}\n")
    slugs = []
    for newspaper in outlets:
        write_inputs(
            newspaper,
            raw_html_dir=args.raw_html_dir,
            data_dir=args.data_dir,
            today=today,
            out_dir=args.out_dir,
        )
        slugs.append(slug_of(newspaper))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "outlets.json").write_text(json.dumps(slugs) + "\n")
    (args.out_dir / "waiting.json").write_text(json.dumps(waiting) + "\n")
    sys.stdout.write(json.dumps(slugs) + "\n")
    return 0


def _agent(args: argparse.Namespace) -> int:
    slug = slug_of(resolve_outlet(args.outlet))
    root = Path.cwd().resolve()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    result = run_agent(
        slug,
        root=root,
        inputs=(root / args.inputs / slug).resolve(),
        out_dir=args.out_dir,
        omp=args.omp,
        model=args.model,
        max_time=args.max_time,
        redact_secrets=[key],
        balance=lambda: balance_problem(key),
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    if result["status"] == INFRA_ERROR:
        # a failed run, not an outlet the agent gave up on
        sys.stdout.write(
            f"::error::Self-fix infrastructure failure: {result['explanation']}\n"
        )
        return 1
    return 0


def _gate(args: argparse.Namespace) -> int:
    task = json.loads(args.task.read_text())
    problems = gate(task, read_patch(args.patch), Path(NEWSPAPERS_PY).read_text())
    for problem in problems:
        sys.stdout.write(f"refused: {problem}\n")
    if not problems:
        sys.stdout.write(f"{args.patch} stays within {task['slug']}\n")
    return 1 if problems else 0


def _off_peak(args: argparse.Namespace) -> int:
    problem = peak_problem(_utcnow(), duration(args.needs))
    if problem is not None:
        sys.stdout.write(f"refused: {problem}\n")
        return 1
    sys.stdout.write(f"off-peak for {off_peak_left(_utcnow())}\n")
    return 0


def _snapshot(args: argparse.Namespace) -> int:
    fixture = write_snapshot(resolve_outlet(args.outlet))
    sys.stdout.write(f"wrote {fixture.snapshot_path.relative_to(_ROOT)}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step of the fix job; see the module docstring."""
    parser = argparse.ArgumentParser(prog="poe selfheal-fix", description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)

    prepare = steps.add_parser("prepare", help="write the agent's inputs")
    prepare.add_argument("--raw-html-dir", type=Path, required=True)
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--date", type=date.fromisoformat, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument(
        "--outlet",
        help="fix this outlet, whatever the health report, the two-run rule "
        "and the weekly cap say",
    )
    prepare.add_argument(
        "--repo",
        help="owner/name: hold back outlets by their selfheal issues and PRs "
        "there (token from GITHUB_TOKEN or GH_TOKEN)",
    )
    prepare.set_defaults(func=_prepare)

    agent = steps.add_parser("agent", help="run one fix session")
    agent.add_argument("outlet")
    agent.add_argument(
        "--inputs", type=Path, default=Path("selfheal"), help="what prepare wrote"
    )
    agent.add_argument("--out-dir", type=Path, required=True)
    agent.add_argument("--omp", default="omp")
    agent.add_argument("--model", default="deepseek/deepseek-flash")
    agent.add_argument("--max-time", default="15m")
    agent.set_defaults(func=_agent)

    check = steps.add_parser("gate", help="refuse a patch beyond the task")
    check.add_argument("task", type=Path, help="the task.json prepare wrote")
    check.add_argument("patch", type=Path)
    check.set_defaults(func=_gate)

    off_peak = steps.add_parser(
        "off-peak", help="fail unless DeepSeek stays off-peak long enough"
    )
    off_peak.add_argument("--needs", default="0", help="how long, e.g. 15m")
    off_peak.set_defaults(func=_off_peak)

    snapshot = steps.add_parser("snapshot", help="write the newest page's snapshot")
    snapshot.add_argument("outlet")
    snapshot.set_defaults(func=_snapshot)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
