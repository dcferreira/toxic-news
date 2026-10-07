# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`poe selfheal-pr`: what the PR job of the self-fix loop writes.

The job holds write access, so it runs none of the patch's code: everything
here reads files as data. Its inputs, all under one `--evidence` directory:

- `task.json`, `last_good.json`: what `poe selfheal-fix prepare` wrote;
- `result.json`, `patch.diff`: what the fix session left;
- `check.json`: `poe selfheal-check --json` on the patched checkout, run as an
  unprivileged user; the patch's code ran there, so it is evidence, not proof;
- `outline.json`, `pixels.json`: the screenshot's counts, and how many of its
  outlines were found in the pixels.

`xpath`
    prints the headline XPath the patch dates from the task's `from_date`,
    read from the patched `newspapers.py` as text, for the screenshot.
`body`
    writes the PR's body.
`comment`
    writes the comment recording the attempt on the outlet's issue.
"""

import argparse
import ast
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from tests.selfheal_check import NEWSPAPERS_PY, _cell, _layout
from tests.selfheal_fix import _expected_from_date, attempt_marker

#: GitHub refuses a PR body longer than this many characters.
MAX_BODY = 65_536
#: Headlines shown before the fold.
FIRST_HEADLINES = 10


def title(task: dict[str, Any]) -> str:
    """Return the PR's title, a Conventional Commit like every PR here."""
    return f"fix: repair the {task['outlet']} headline XPath"


def branch(task: dict[str, Any]) -> str:
    """Return the PR's branch."""
    return f"selfheal/{task['slug']}-{task['today']}"


# --- the XPath to outline ----------------------------------------------------


def fixed_xpath(source: str, task: dict[str, Any]) -> str | None:
    """Return the `headline_xpath` of the `DatedXpath` from the task's `from_date`.

    `source` is the patched `newspapers.py`, only parsed. `None` when the fix
    added no such `DatedXpath`, or its XPath is not a plain string.
    """
    layout = _layout(ast.parse(source), task["slug"])
    own = [layout.entry, *layout.helpers] if layout.entry is not None else []
    wanted = _expected_from_date(task)
    for node in own:
        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "DatedXpath"
            ):
                continue
            keywords = {k.arg: k.value for k in call.keywords}
            if "from_date" not in keywords:
                continue
            if ast.unparse(keywords["from_date"]) != wanted:
                continue
            xpath = keywords.get("headline_xpath")
            if isinstance(xpath, ast.Constant) and isinstance(xpath.value, str):
                return xpath.value
            return None
    return None


# --- evidence -----------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """Everything the PR job knows of one attempt, read from `--evidence`."""

    task: dict[str, Any]
    last_good: dict[str, Any]
    result: dict[str, Any]
    patch: str
    check: dict[str, Any] | None
    outline: dict[str, Any] | None
    pixels: dict[str, Any] | None

    @classmethod
    def read(cls, directory: Path) -> "Evidence":
        """Read the evidence files; the optional ones may be missing or broken."""

        def load(name: str) -> dict[str, Any] | None:
            try:
                found = json.loads((directory / name).read_text())
            except (OSError, ValueError):
                return None
            return found if isinstance(found, dict) else None

        patch = directory / "patch.diff"
        return cls(
            task=json.loads((directory / "task.json").read_text()),
            last_good=json.loads((directory / "last_good.json").read_text()),
            result=load("result.json") or {},
            patch=patch.read_bytes().decode(errors="replace")
            if patch.is_file()
            else "",
            check=load("check.json"),
            outline=load("outline.json"),
            pixels=load("pixels.json"),
        )

    @property
    def passed(self) -> bool:
        """Return whether the check passed and the screenshot shows every match."""
        return bool(self.check and self.check.get("passed") is True)


_FILE = re.compile(r"^diff --git a/(\S+) b/", re.MULTILINE)


def file_diffs(patch: str) -> dict[str, str]:
    """Split a git patch into each file's part, by path."""
    starts = list(_FILE.finditer(patch))
    return {
        m[1]: patch[m.start() : starts[i + 1].start() if i + 1 < len(starts) else None]
        for i, m in enumerate(starts)
    }


def _fenced(text: str, language: str = "") -> str:
    """Put `text` in a code block no backtick run in it can close."""
    runs = [len(r) for r in re.findall(r"`+", text)]
    fence = "`" * max(3, max(runs, default=0) + 1)
    return f"{fence}{language}\n{text.rstrip()}\n{fence}"


def _quote(text: str | None) -> str:
    """Quote the agent's words, inert: it is model output, shown to a person."""
    if not text:
        return "> _(none given)_"
    return "\n".join(f"> {_cell(line)}" for line in text.strip().splitlines())


def _headline_rows(headlines: Sequence[Sequence[str]]) -> list[str]:
    rows = ["| # | Headline | Link |", "|---|---|---|"]
    rows += [
        f"| {i} | {_cell(text)} | {_cell(url)} |"
        for i, (text, url) in enumerate(headlines, start=1)
    ]
    return rows


def _details(summary: str, lines: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", "", *lines, "", "</details>"]


def _counts(evidence: Evidence) -> list[str]:
    task, good = evidence.task, evidence.last_good
    expected = task["expected_headlines"]
    today = task["today"]
    old_today = task.get("headlines_today")
    new = len(evidence.check["headlines"]) if evidence.check else None
    last = (
        f"last good ({good['date']}, old extractor)"
        if task.get("last_good")
        else f"newest recorded page ({good['date']}, old extractor)"
    )

    def cell(n: object) -> str:
        return "?" if n is None else str(n)

    return [
        "| Run | Headlines | Expected |",
        "|---|---|---|",
        f"| {last} | {len(good['headlines'])} | {expected} |",
        f"| today ({today}), old extractor | {cell(old_today)} | {expected} |",
        f"| today ({today}), new extractor | {cell(new)} | {expected} |",
    ]


def _screenshot(evidence: Evidence, screenshot_url: str | None) -> list[str]:
    outline, pixels = evidence.outline, evidence.pixels
    if screenshot_url is None or outline is None:
        return [
            (
                "No screenshot: the fix's extractor is not a plain XPath, or the "
                "browser could not load the page."
            )
        ]
    found = pixels["found"] if pixels else None
    line = (
        f"The browser matched {outline['matches']} nodes with the new XPath "
        f"and outlined {outline['outlined']}"
    )
    if outline.get("hidden"):
        line += f" ({outline['hidden']} take up no space on the page)"
    if found is None:
        line += "; the outlines could not be checked in the pixels."
    elif found == outline["outlined"]:
        line += "; every outline was found in the screenshot's pixels."
    else:
        line += (
            f"; only {found} of the outlines were found in the screenshot's "
            "pixels, so some matches may be cut off or covered."
        )
    return [
        line,
        "",
        f"![{_cell(evidence.task['outlet'])} front page]({screenshot_url})",
    ]


def _checks(evidence: Evidence) -> list[str]:
    if evidence.check is None:
        return ["`selfheal-check` left no report."]
    rows = ["| | Check | Result |", "|---|---|---|"]
    for r in evidence.check["results"]:
        mark = "✅" if r["passed"] else "❌"
        rows.append(f"| {mark} | {_cell(r['name'])} | {_cell(r['detail'])} |")
    return rows


def _session(result: dict[str, Any]) -> str:
    session = result.get("session") or {}
    cost = session.get("cost_usd")
    parts = ["deepseek-flash"]
    if session.get("turns") is not None:
        parts.append(f"{session['turns']} turns")
    if cost is not None:
        parts.append(f"${cost:.3f}")
    return " · ".join(parts)


def render_body(
    evidence: Evidence,
    *,
    issue: int | None,
    run_url: str,
    screenshot_url: str | None,
) -> str:
    """Return the PR's body: the case for the fix, so nobody has to run it."""
    task = evidence.task
    headlines = evidence.check["headlines"] if evidence.check else []
    lines = [
        f"## selfheal({task['slug']}): headline XPath drifted",
        "",
        f"**{_cell(task['outlet'])}** ({task['url']}) has been broken since "
        f"{task['first_bad']}"
        + (f"; last good on {task['last_good']}." if task.get("last_good") else "."),
    ]
    if issue is not None:
        lines += ["", f"Closes #{issue}."]
    lines += ["", *_counts(evidence), "", "### Today's headlines, new extractor", ""]
    lines += _headline_rows(headlines[:FIRST_HEADLINES])
    if len(headlines) > FIRST_HEADLINES:
        lines += [
            "",
            *_details(
                f"All {len(headlines)} headlines, in page order",
                _headline_rows(headlines),
            ),
        ]
    good = evidence.last_good
    lines += [
        "",
        *_details(
            f"The {len(good['headlines'])} headlines of {good['date']}, "
            f"from {_cell(good['source'])}",
            _headline_rows([(h["text"], h["url"]) for h in good["headlines"]]),
        ),
        "",
        "### Screenshot",
        "",
        *_screenshot(evidence, screenshot_url),
        "",
        "### The change",
        "",
        _fenced(file_diffs(evidence.patch).get(NEWSPAPERS_PY, "(none)"), "diff"),
        "",
        (
            "Also adds today's page and its parse snapshot: "
            f"`{task['fixture']}`, `{task['snapshot']}`."
        ),
        "",
        "### Why, in the agent's words",
        "",
        _quote(evidence.result.get("explanation")),
        "",
        "Left out on purpose:",
        "",
        _quote(evidence.result.get("excluded")),
        "",
        "### Checks",
        "",
        *_checks(evidence),
        "",
        "---",
        "",
        (
            f"Opened by the [self-fix loop]({run_url}) ({_session(evidence.result)}). "
            "Every count above comes from the workflow, not the agent; the checks "
            "ran the patch's code as an unprivileged user, so they are evidence for "
            "a person, not proof. Nothing here merges on its own. CI does not start "
            "on a PR the loop opens: marking it ready for review runs it."
        ),
    ]
    body = "\n".join(lines) + "\n"
    if len(body) > MAX_BODY:
        cut = "\n\n_(cut short: GitHub's limit; the run has the rest)_\n"
        body = body[: MAX_BODY - len(cut)] + cut
    return body


# --- the issue comment ----------------------------------------------------------

#: What became of an attempt that ran, for the comment on the outlet's issue.
OUTCOMES = {
    "pr": "opened a fix",
    "gave_up": "gave up",
    "refused": "left a patch that reaches beyond the outlet, so it was refused",
    "failed": "left a fix that fails `selfheal-check`",
    "no_patch": "left no patch",
}


def render_comment(
    evidence: Evidence, *, outcome: str, run_url: str, pr: int | None = None
) -> str:
    """Return the comment recording an attempt on the outlet's issue.

    It carries the attempt marker, which holds the outlet back for a week.
    """
    day = date.fromisoformat(evidence.task["today"])
    if outcome == "pr":
        head = f"The [self-fix loop]({run_url}) opened #{pr} to fix this."
        lines = [head]
    else:
        lines = [
            (
                f"The [self-fix loop]({run_url}) tried to fix this on {day} and "
                f"{OUTCOMES[outcome]}. It tries again in a week at the earliest."
            ),
            "",
            "The agent's account:",
            "",
            _quote(evidence.result.get("explanation")),
        ]
        if evidence.check is not None:
            lines += ["", *_details("selfheal-check", _checks(evidence))]
    return "\n".join([*lines, "", attempt_marker(day)]) + "\n"


# --- command line -------------------------------------------------------------


def _xpath(args: argparse.Namespace) -> int:
    task = json.loads(args.task.read_text())
    found = fixed_xpath(args.newspapers.read_text(), task)
    if found is None:
        return 1
    sys.stdout.write(found + "\n")
    return 0


def _body(args: argparse.Namespace) -> int:
    evidence = Evidence.read(args.evidence)
    body = render_body(
        evidence,
        issue=args.issue,
        run_url=args.run_url,
        screenshot_url=args.screenshot_url,
    )
    args.output.write_text(body)
    return 0


def _comment(args: argparse.Namespace) -> int:
    evidence = Evidence.read(args.evidence)
    comment = render_comment(
        evidence, outcome=args.outcome, run_url=args.run_url, pr=args.pr
    )
    args.output.write_text(comment)
    return 0


def _names(args: argparse.Namespace) -> int:
    task = json.loads(args.task.read_text())
    sys.stdout.write(f"title={title(task)}\nbranch={branch(task)}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step of the PR job; see the module docstring."""
    parser = argparse.ArgumentParser(prog="poe selfheal-pr", description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)

    xpath = steps.add_parser("xpath", help="print the XPath the fix dates")
    xpath.add_argument("task", type=Path)
    xpath.add_argument("newspapers", type=Path, help="the patched newspapers.py")
    xpath.set_defaults(func=_xpath)

    names = steps.add_parser("names", help="print the PR's title and branch")
    names.add_argument("task", type=Path)
    names.set_defaults(func=_names)

    body = steps.add_parser("body", help="write the PR's body")
    body.add_argument("--evidence", type=Path, required=True)
    body.add_argument("--issue", type=int)
    body.add_argument("--run-url", required=True)
    body.add_argument("--screenshot-url")
    body.add_argument("--output", type=Path, required=True)
    body.set_defaults(func=_body)

    comment = steps.add_parser("comment", help="write the attempt's issue comment")
    comment.add_argument("--evidence", type=Path, required=True)
    comment.add_argument("--outcome", choices=sorted(OUTCOMES), required=True)
    comment.add_argument("--run-url", required=True)
    comment.add_argument("--pr", type=int)
    comment.add_argument("--output", type=Path, required=True)
    comment.set_defaults(func=_comment)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
