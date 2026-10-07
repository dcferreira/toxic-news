# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Triage of broken outlets into GitHub issues, one issue per failure.

The first step of the self-fix loop only watches: after each run the health
reports say which outlets are `xpath`-broken, and this module keeps a GitHub
issue for each, so a person (and later the fixing agent) has one place to look.
An outlet has at most one open issue, found by its label. A closed issue is
never reopened: an outlet that breaks again, or one whose issue somebody closed
while it was still broken, is a new failure and gets a new issue. An issue is
closed only on evidence, after the outlet is `ok` in two reports in a row, so
an outlet sitting on its headline-count threshold does not open and close an
issue every other day. `other` verdicts (blocked, timing out) say nothing about
the extractor, so they neither open nor close anything.

`plan` decides, purely, from the reports and the open issues; `apply` carries
the plan out through a `Tracker`, so the planning is tested without a network
and re-running on the same reports writes nothing.
"""

import datetime
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import quote

import typer
from loguru import logger
from pydantic import BaseModel, ValidationError

from toxic_news.fetchers import clean_url
from toxic_news.health import OutletHealth, Verdict, read_health

LABEL = "selfheal"
LABEL_COLOR = "d93f0b"
API_URL = "https://api.github.com"
PAGE_SIZE = 100
HTTP_NOT_FOUND = 404
TIMEOUT_SECONDS = 30

MARKER = "<!-- selfheal-failure -->"
_BLOCK = re.compile(re.escape(MARKER) + r"\s*```json\s*(.*?)\s*```", re.DOTALL)


def label_of(url: str) -> str:
    """Return the label that ties issues to the outlet at `url`.

    It is the outlet's fixture slug, as `poe selfheal-check` knows it.
    """
    return f"{LABEL}:{clean_url(url)}"


class Report(BaseModel):
    """The health of every outlet on one day."""

    date: datetime.date
    outlets: list[OutletHealth]


class Failure(BaseModel):
    """An outlet's XPath failure, as the issue about it records it."""

    newspaper: str
    url: str
    first_seen: datetime.date
    last_seen: datetime.date
    headlines: int
    expected: int
    body_bytes: int | None

    @classmethod
    def of(
        cls,
        outlet: OutletHealth,
        *,
        first_seen: datetime.date,
        last_seen: datetime.date,
    ) -> "Failure":
        """Return the failure `outlet` shows, seen from `first_seen` to `last_seen`."""
        return cls(
            newspaper=outlet.newspaper,
            url=outlet.url,
            first_seen=first_seen,
            last_seen=last_seen,
            headlines=outlet.headlines,
            expected=outlet.expected,
            body_bytes=outlet.body_bytes,
        )

    @property
    def title(self) -> str:
        """Return the issue title."""
        return f"{LABEL}: {self.newspaper} XPath is broken"


class Issue(BaseModel):
    """An open issue of the repository."""

    number: int
    title: str
    body: str
    labels: list[str]


def render_body(failure: Failure) -> str:
    """Return the issue body: for people first, then the block `parse_failure` reads."""
    size = "no page" if failure.body_bytes is None else f"{failure.body_bytes:,} bytes"
    return (
        f"**{failure.newspaper}** ({failure.url}) is fetched fine, but its "
        "extractor no longer matches the page.\n\n"
        f"- headlines found: {failure.headlines} (expected {failure.expected})\n"
        f"- page size: {size}\n"
        f"- first seen: {failure.first_seen.isoformat()}\n"
        f"- last seen: {failure.last_seen.isoformat()}\n\n"
        "This is a triage-only issue, opened and kept up to date by the "
        "self-fix loop; it is closed once the outlet is healthy in two runs "
        "in a row. A fix attempt is recorded later as a comment carrying the "
        "marker `<!-- selfheal-attempt YYYY-MM-DD -->`.\n\n"
        "Machine-readable state, rewritten on every run:\n\n"
        f"{MARKER}\n```json\n{failure.json(indent=2)}\n```\n"
    )


def parse_failure(body: str) -> Failure | None:
    """Return the failure recorded in `body`, or None if it is missing or invalid."""
    match = _BLOCK.search(body)
    if match is None:
        return None
    try:
        return Failure.parse_raw(match.group(1))
    except ValidationError:
        return None


@dataclass(frozen=True)
class Open:
    """Open an issue for a new failure."""

    failure: Failure


@dataclass(frozen=True)
class Refresh:
    """Bring the issue of a failure that goes on up to date."""

    number: int
    failure: Failure


@dataclass(frozen=True)
class Close:
    """Close the issue of an outlet that recovered."""

    number: int
    outlet: OutletHealth
    previous_date: datetime.date
    latest_date: datetime.date


Action = Open | Refresh | Close


def describe(action: Action) -> str:
    """Return a one-line description of `action`."""
    if isinstance(action, Open):
        return f"open an issue for {action.failure.newspaper}"
    if isinstance(action, Refresh):
        return f"refresh #{action.number} for {action.failure.newspaper}"
    return f"close #{action.number}: {action.outlet.newspaper} recovered"


def _open_issue_by_label(open_issues: Sequence[Issue]) -> dict[str, Issue]:
    """Map each label to its lowest-numbered open issue."""
    by_label: dict[str, Issue] = {}
    for issue in sorted(open_issues, key=lambda i: i.number, reverse=True):
        for label in issue.labels:
            by_label[label] = issue
    return by_label


def plan(
    latest: Report,
    previous: Report | None,
    open_issues: Sequence[Issue],
) -> list[Action]:
    """Return what to do about the outlets of `latest`, in their order."""
    issues = _open_issue_by_label(open_issues)
    before = {} if previous is None else {o.url: o for o in previous.outlets}
    actions: list[Action] = []
    for outlet in latest.outlets:
        issue = issues.get(label_of(outlet.url))
        if outlet.verdict == Verdict.XPATH:
            if issue is None:
                failure = Failure.of(
                    outlet, first_seen=latest.date, last_seen=latest.date
                )
                actions.append(Open(failure))
                continue
            recorded = parse_failure(issue.body)
            first_seen = latest.date if recorded is None else recorded.first_seen
            failure = Failure.of(outlet, first_seen=first_seen, last_seen=latest.date)
            actions.append(Refresh(issue.number, failure))
        elif outlet.verdict == Verdict.OK and issue is not None:
            earlier = before.get(outlet.url)
            if previous is not None and earlier and earlier.verdict == Verdict.OK:
                actions.append(Close(issue.number, outlet, previous.date, latest.date))
    return actions


class Tracker(Protocol):
    """The issue tracker the triage writes to."""

    def open_issues(self) -> list[Issue]:
        """Return the open issues carrying the `selfheal` label."""

    def ensure_label(self, name: str) -> None:
        """Create the label `name` unless it exists."""

    def create_issue(self, title: str, body: str, labels: list[str]) -> int:
        """Open an issue and return its number."""

    def update_body(self, number: int, body: str) -> None:
        """Replace the body of an issue."""

    def comment(self, number: int, body: str) -> None:
        """Add a comment to an issue."""

    def close(self, number: int) -> None:
        """Close an issue as completed."""


def _recovery_comment(action: Close) -> str:
    outlet = action.outlet
    return (
        f"Recovered: ok on {action.previous_date.isoformat()} and "
        f"{action.latest_date.isoformat()} "
        f"({outlet.headlines}/{outlet.expected} headlines). Closing."
    )


def apply(actions: Sequence[Action], tracker: Tracker) -> None:
    """Carry out `actions`, writing to an issue only when it would change."""
    current: dict[int, str] | None = None
    for action in actions:
        if isinstance(action, Open):
            label = label_of(action.failure.url)
            tracker.ensure_label(LABEL)
            tracker.ensure_label(label)
            tracker.create_issue(
                action.failure.title, render_body(action.failure), [LABEL, label]
            )
        elif isinstance(action, Refresh):
            if current is None:
                current = {i.number: i.body for i in tracker.open_issues()}
            body = render_body(action.failure)
            if current.get(action.number) != body:
                tracker.update_body(action.number, body)
        else:
            tracker.comment(action.number, _recovery_comment(action))
            tracker.close(action.number)


def latest_reports(data_dir: Path) -> tuple[Report, Report | None]:
    """Return the newest health report and the one before it, if any."""
    dates = sorted(
        datetime.date(int(path.parts[-3]), int(path.parts[-2]), int(path.stem))
        for path in (data_dir / "health").glob("*/*/*.json")
    )
    if not dates:
        msg = f"No health reports under {data_dir / 'health'}"
        raise FileNotFoundError(msg)
    reports = [Report(date=d, outlets=read_health(data_dir, d)) for d in dates[-2:]]
    previous = reports[0] if len(reports) == 2 else None  # noqa: PLR2004
    return reports[-1], previous


class GitHub:
    """The few calls of the GitHub REST API the triage needs."""

    def __init__(self, repo: str, *, token: str | None, api_url: str = API_URL):
        """Talk to `repo` ("owner/name"), authenticating with `token` if given."""
        self.repo = repo
        self.token = token
        self.api_url = api_url.rstrip("/")
        self._labels: set[str] = set()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(  # noqa: S310 (api_url is http(s) by construction)
            f"{self.api_url}/repos/{self.repo}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            return json.loads(response.read() or "null")

    def open_issues(self) -> list[Issue]:
        """Return the open issues carrying the `selfheal` label, not pull requests."""
        issues: list[Issue] = []
        page = 1
        while True:
            items = self._request(
                "GET",
                f"/issues?state=open&labels={LABEL}&per_page={PAGE_SIZE}&page={page}",
            )
            issues += [
                Issue(
                    number=item["number"],
                    title=item["title"],
                    body=item["body"] or "",
                    labels=[label["name"] for label in item["labels"]],
                )
                for item in items
                if "pull_request" not in item
            ]
            if len(items) < PAGE_SIZE:
                return issues
            page += 1

    def _pages(self, path: str) -> list[dict[str, Any]]:
        """Return every item of a paginated list, `path` given with its query."""
        found: list[dict[str, Any]] = []
        page = 1
        while True:
            items = self._request("GET", f"{path}&per_page={PAGE_SIZE}&page={page}")
            found += items
            if len(items) < PAGE_SIZE:
                return found
            page += 1

    def labelled(self, label: str) -> list[dict[str, Any]]:
        """Return the issues and pull requests carrying `label`, in any state."""
        return self._pages(f"/issues?state=all&labels={quote(label, safe='')}")

    def comments(self, number: int) -> list[dict[str, Any]]:
        """Return the comments of an issue, oldest first."""
        return self._pages(f"/issues/{number}/comments?")

    def ensure_label(self, name: str) -> None:
        """Create the label `name` unless it exists."""
        if name in self._labels:
            return
        try:
            self._request("GET", f"/labels/{quote(name, safe='')}")
        except urllib.error.HTTPError as e:
            if e.code != HTTP_NOT_FOUND:
                raise
            self._request(
                "POST",
                "/labels",
                {
                    "name": name,
                    "color": LABEL_COLOR,
                    "description": "Tracked by the toxic-news self-fix loop",
                },
            )
        self._labels.add(name)

    def create_issue(self, title: str, body: str, labels: list[str]) -> int:
        """Open an issue and return its number."""
        created = self._request(
            "POST", "/issues", {"title": title, "body": body, "labels": labels}
        )
        return int(created["number"])

    def update_body(self, number: int, body: str) -> None:
        """Replace the body of an issue."""
        self._request("PATCH", f"/issues/{number}", {"body": body})

    def comment(self, number: int, body: str) -> None:
        """Add a comment to an issue."""
        self._request("POST", f"/issues/{number}/comments", {"body": body})

    def close(self, number: int) -> None:
        """Close an issue as completed."""
        state = {"state": "closed", "state_reason": "completed"}
        self._request("PATCH", f"/issues/{number}", state)


def triage(data_dir: Path, tracker: Tracker, *, dry_run: bool = False) -> list[Action]:
    """Triage the latest health reports into `tracker`, and return what was done."""
    latest, previous = latest_reports(data_dir)
    actions = plan(latest, previous, tracker.open_issues())
    if not dry_run:
        apply(actions, tracker)
    return actions


app = typer.Typer()


@app.command()
def main(
    data_dir: Path = Path("data"),
    repo: str = typer.Option(
        os.environ.get("GITHUB_REPOSITORY", ""),
        help="The repository, as owner/name.",
    ),
    # typer 0.9.0 cannot build a command from a PEP 604 union annotation
    report: Optional[Path] = typer.Option(  # noqa: UP045
        None,
        help="Also append the actions to this file, e.g. $GITHUB_STEP_SUMMARY.",
    ),
    # typer reads the option from the default; the parameter is keyword-only in use
    dry_run: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        help="Only print what would be done.",
    ),
) -> None:
    """Open, refresh and close an issue per broken outlet, from the health reports.

    The token is read from GITHUB_TOKEN, or GH_TOKEN.
    """
    if not repo:
        logger.error("No repository: pass --repo or set GITHUB_REPOSITORY")
        raise typer.Exit(code=1)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    actions = triage(data_dir, GitHub(repo, token=token), dry_run=dry_run)
    lines = [describe(action) for action in actions]
    for line in lines or ["Nothing to do"]:
        typer.echo(line)
    if report is not None:
        with report.open("a") as f:
            f.write("### Triage\n\n")
            f.write("".join(f"- {line}\n" for line in lines) or "Nothing to do.\n")
            f.write("\n")


if __name__ == "__main__":
    app()
