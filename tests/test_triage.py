# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the self-fix triage in ``toxic_news.triage``."""

import datetime
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from toxic_news.health import OutletHealth, write_health
from toxic_news.triage import (
    Close,
    Failure,
    GitHub,
    Issue,
    Open,
    Refresh,
    Report,
    apply,
    label_of,
    latest_reports,
    parse_failure,
    plan,
    render_body,
    triage,
)

DAY = datetime.date(2026, 10, 6)
ONE_DAY = datetime.timedelta(days=1)


def _outlet(
    newspaper: str = "BBC",
    url: str = "https://bbc.com",
    *,
    verdict: str = "xpath",
    headlines: int = 0,
    expected: int = 47,
) -> OutletHealth:
    return OutletHealth(
        newspaper=newspaper,
        url=url,
        status=200,
        fetch_error=None,
        parse_error=None,
        score_error=None,
        body_bytes=679_259,
        headlines=headlines,
        expected=expected,
        verdict=verdict,
    )


def _report(date: datetime.date, *outlets: OutletHealth) -> Report:
    return Report(date=date, outlets=list(outlets))


def _issue(number: int, outlet: OutletHealth, first_seen: datetime.date) -> Issue:
    failure = Failure.of(outlet, first_seen=first_seen, last_seen=first_seen)
    return Issue(
        number=number,
        title=failure.title,
        body=render_body(failure),
        labels=["selfheal", label_of(outlet.url)],
    )


class FakeTracker:
    """The issues of a repository, kept in memory."""

    def __init__(self) -> None:
        self.issues: dict[int, Issue] = {}
        self.closed: set[int] = set()
        self.comments: dict[int, list[str]] = {}
        self.labels: set[str] = set()
        self.writes = 0

    def open_issues(self) -> list[Issue]:
        return [
            issue for number, issue in self.issues.items() if number not in self.closed
        ]

    def ensure_label(self, name: str) -> None:
        if name not in self.labels:
            self.labels.add(name)
            self.writes += 1

    def create_issue(self, title: str, body: str, labels: list[str]) -> int:
        number = len(self.issues) + 1
        self.issues[number] = Issue(
            number=number, title=title, body=body, labels=labels
        )
        self.writes += 1
        return number

    def update_body(self, number: int, body: str) -> None:
        self.issues[number] = self.issues[number].copy(update={"body": body})
        self.writes += 1

    def comment(self, number: int, body: str) -> None:
        self.comments.setdefault(number, []).append(body)
        self.writes += 1

    def close(self, number: int) -> None:
        self.closed.add(number)
        self.writes += 1


# --- labels and the issue body ------------------------------------------------


def test_the_label_is_the_outlets_fixture_slug() -> None:
    assert label_of("https://bbc.com") == "selfheal:bbc.com"
    guardian = label_of("https://www.theguardian.com/us")
    assert guardian == "selfheal:www.theguardian.com__us"


def test_every_outlets_label_fits_githubs_limit() -> None:
    from toxic_news.newspapers import newspapers  # noqa: PLC0415

    assert all(len(label_of(str(n.url))) <= 50 for n in newspapers)


def test_the_body_carries_the_failure_in_a_machine_readable_block() -> None:
    failure = Failure.of(_outlet(), first_seen=DAY - ONE_DAY, last_seen=DAY)
    body = render_body(failure)

    assert "BBC" in body
    assert parse_failure(body) == failure


def test_a_body_without_the_block_has_no_failure() -> None:
    assert parse_failure("someone rewrote this by hand") is None


# --- planning -------------------------------------------------------------------


def test_an_outlet_newly_broken_gets_an_issue() -> None:
    bbc = _outlet()

    actions = plan(_report(DAY, bbc), _report(DAY - ONE_DAY), [])

    assert actions == [Open(Failure.of(bbc, first_seen=DAY, last_seen=DAY))]


def test_a_still_broken_outlet_has_its_issue_refreshed_not_refiled() -> None:
    first_seen = DAY - 3 * ONE_DAY
    issue = _issue(7, _outlet(headlines=0), first_seen)
    bbc = _outlet(headlines=2)

    actions = plan(_report(DAY, bbc), _report(DAY - ONE_DAY, bbc), [issue])

    failure = Failure.of(bbc, first_seen=first_seen, last_seen=DAY)
    assert actions == [Refresh(7, failure)]


def test_an_issue_whose_block_was_lost_keeps_today_as_its_first_day() -> None:
    issue = _issue(7, _outlet(), DAY).copy(update={"body": "edited by hand"})

    actions = plan(_report(DAY, _outlet()), None, [issue])

    assert actions == [Refresh(7, Failure.of(_outlet(), first_seen=DAY, last_seen=DAY))]


def test_an_outlet_ok_for_two_runs_has_its_issue_closed() -> None:
    issue = _issue(7, _outlet(), DAY - 5 * ONE_DAY)
    ok = _outlet(verdict="ok", headlines=45)

    actions = plan(_report(DAY, ok), _report(DAY - ONE_DAY, ok), [issue])

    assert actions == [Close(7, ok, DAY - ONE_DAY, DAY)]


def test_one_ok_run_is_not_a_recovery() -> None:
    issue = _issue(7, _outlet(), DAY - 5 * ONE_DAY)
    ok = _outlet(verdict="ok", headlines=45)

    assert plan(_report(DAY, ok), _report(DAY - ONE_DAY, _outlet()), [issue]) == []
    assert plan(_report(DAY, ok), _report(DAY - ONE_DAY), [issue]) == []
    assert plan(_report(DAY, ok), None, [issue]) == []


@pytest.mark.parametrize("previous", ["ok", "xpath"])
def test_an_outlet_failing_for_other_reasons_is_left_alone(previous: str) -> None:
    issue = _issue(7, _outlet(), DAY - 5 * ONE_DAY)
    other = _outlet(verdict="other")

    report = _report(DAY, other)
    before = _report(DAY - ONE_DAY, _outlet(verdict=previous))
    assert plan(report, before, [issue]) == []
    assert plan(report, before, []) == []


def test_a_healthy_outlet_without_an_issue_needs_nothing() -> None:
    ok = _outlet(verdict="ok", headlines=45)

    assert plan(_report(DAY, ok), _report(DAY - ONE_DAY, ok), []) == []


def test_an_issue_for_an_outlet_missing_from_the_report_is_left_alone() -> None:
    issue = _issue(7, _outlet(), DAY - 5 * ONE_DAY)

    assert plan(_report(DAY), _report(DAY - ONE_DAY), [issue]) == []


def test_issues_of_other_outlets_do_not_count() -> None:
    fox = _outlet("Fox News", "https://foxnews.com")
    bbc_issue = _issue(7, _outlet(), DAY - ONE_DAY)

    actions = plan(_report(DAY, fox), None, [bbc_issue])

    assert actions == [Open(Failure.of(fox, first_seen=DAY, last_seen=DAY))]


# --- applying, over several days ------------------------------------------------


def _run(tracker: FakeTracker, days: list[Report]) -> None:
    previous = None
    for report in days:
        apply(plan(report, previous, tracker.open_issues()), tracker)
        previous = report


def test_applying_opens_a_labelled_issue_with_the_evidence() -> None:
    tracker = FakeTracker()

    _run(tracker, [_report(DAY, _outlet())])

    (issue,) = tracker.open_issues()
    assert issue.labels == ["selfheal", "selfheal:bbc.com"]
    assert tracker.labels == {"selfheal", "selfheal:bbc.com"}
    assert parse_failure(issue.body) == Failure.of(
        _outlet(), first_seen=DAY, last_seen=DAY
    )


def test_triaging_the_same_reports_twice_changes_nothing() -> None:
    tracker = FakeTracker()
    days = [_report(DAY - ONE_DAY, _outlet()), _report(DAY, _outlet())]
    _run(tracker, days)
    writes = tracker.writes

    apply(plan(days[1], days[0], tracker.open_issues()), tracker)

    assert tracker.writes == writes
    assert len(tracker.issues) == 1


def test_a_recovered_outlet_gets_a_comment_and_its_issue_closed() -> None:
    tracker = FakeTracker()
    ok = _outlet(verdict="ok", headlines=45)

    _run(
        tracker,
        [
            _report(DAY - 2 * ONE_DAY, _outlet()),
            _report(DAY - ONE_DAY, ok),
            _report(DAY, ok),
        ],
    )

    assert tracker.open_issues() == []
    assert tracker.closed == {1}
    (comment,) = tracker.comments[1]
    assert "45" in comment
    assert DAY.isoformat() in comment


def test_a_new_failure_after_a_recovery_gets_a_new_issue() -> None:
    tracker = FakeTracker()
    ok = _outlet(verdict="ok", headlines=45)

    _run(
        tracker,
        [
            _report(DAY - 3 * ONE_DAY, _outlet()),
            _report(DAY - 2 * ONE_DAY, ok),
            _report(DAY - ONE_DAY, ok),
            _report(DAY, _outlet()),
        ],
    )

    assert tracker.closed == {1}
    (issue,) = tracker.open_issues()
    assert issue.number == 2
    failure = parse_failure(issue.body)
    assert failure is not None
    assert failure.first_seen == DAY


def test_an_issue_closed_by_hand_while_still_broken_starts_a_new_failure() -> None:
    tracker = FakeTracker()
    _run(tracker, [_report(DAY - ONE_DAY, _outlet())])
    tracker.close(1)

    _run(tracker, [_report(DAY, _outlet())])

    assert [issue.number for issue in tracker.open_issues()] == [2]


def test_a_flapping_outlet_keeps_one_issue_until_it_settles() -> None:
    """Fox News sat on its over-count threshold for a week."""
    tracker = FakeTracker()
    verdicts = ["xpath", "ok", "xpath", "ok", "xpath", "ok", "ok"]

    _run(
        tracker,
        [
            _report(
                DAY + i * ONE_DAY,
                _outlet("Fox News", "https://foxnews.com", verdict=v, expected=125),
            )
            for i, v in enumerate(verdicts)
        ],
    )

    assert len(tracker.issues) == 1
    assert tracker.closed == {1}


# --- reading the reports --------------------------------------------------------


def test_the_latest_two_reports_are_read_whatever_the_gap(tmp_path: Path) -> None:
    write_health(tmp_path, DAY - 9 * ONE_DAY, [_outlet()])
    write_health(tmp_path, DAY - 4 * ONE_DAY, [_outlet(headlines=1)])
    write_health(tmp_path, DAY, [_outlet(headlines=2)])

    latest, previous = latest_reports(tmp_path)

    assert latest == _report(DAY, _outlet(headlines=2))
    assert previous == _report(DAY - 4 * ONE_DAY, _outlet(headlines=1))


def test_a_single_report_has_no_previous(tmp_path: Path) -> None:
    write_health(tmp_path, DAY, [_outlet()])

    assert latest_reports(tmp_path) == (_report(DAY, _outlet()), None)


def test_no_reports_at_all_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        latest_reports(tmp_path)


# --- the GitHub API ---------------------------------------------------------------


class FakeGitHub(BaseHTTPRequestHandler):
    """Just enough of the GitHub REST API for the triage, kept in class state."""

    issues: ClassVar[list[dict[str, Any]]] = []
    labels: ClassVar[set[str]] = set()
    comments: ClassVar[dict[int, list[dict[str, Any]]]] = {}
    requests: ClassVar[list[tuple[str, str, Any]]] = []

    def _send(self, status: int, payload: object = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length)) if length else None
        self.requests.append((self.command, self.path, payload))
        assert self.headers["Authorization"] == "Bearer t0ken"
        return payload

    def do_GET(self) -> None:
        self._record()
        url = urlsplit(self.path)
        if url.path == "/repos/o/r/issues":
            query = parse_qs(url.query)
            page = int(query["page"][0])
            per_page = int(query["per_page"][0])
            items = [
                i
                for i in self.issues
                if query["state"][0] in ("all", i["state"])
                and query["labels"][0] in [lb["name"] for lb in i["labels"]]
            ]
            self._send(200, items[(page - 1) * per_page : page * per_page])
        elif url.path.endswith("/comments"):
            query = parse_qs(url.query)
            page = int(query["page"][0])
            per_page = int(query["per_page"][0])
            number = int(url.path.split("/")[-2])
            items = self.comments.get(number, [])
            self._send(200, items[(page - 1) * per_page : page * per_page])
        elif url.path.startswith("/repos/o/r/labels/"):
            name = unquote(url.path.removeprefix("/repos/o/r/labels/"))
            if name in self.labels:
                self._send(200, {"name": name})
            else:
                self._send(404, {"message": "Not Found"})
        else:
            self._send(404, {"message": "Not Found"})

    def do_POST(self) -> None:
        payload = self._record()
        url = urlsplit(self.path)
        if url.path == "/repos/o/r/labels":
            self.labels.add(payload["name"])
            self._send(201, payload)
        elif url.path == "/repos/o/r/issues":
            number = len(self.issues) + 1
            self.issues.append(
                {
                    "number": number,
                    "state": "open",
                    "title": payload["title"],
                    "body": payload["body"],
                    "labels": [{"name": n} for n in payload["labels"]],
                }
            )
            self._send(201, {"number": number})
        elif url.path.endswith("/comments"):
            self._send(201, {})
        else:
            self._send(404, {"message": "Not Found"})

    def do_PATCH(self) -> None:
        payload = self._record()
        number = int(urlsplit(self.path).path.rsplit("/", 1)[1])
        issue = self.issues[number - 1]
        issue.update(payload)
        self._send(200, issue)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (the base's name)
        """Keep the test output quiet."""


@pytest.fixture
def github() -> Iterator[GitHub]:
    FakeGitHub.issues = []
    FakeGitHub.labels = {"selfheal"}
    FakeGitHub.comments = {}
    FakeGitHub.requests = []
    server = HTTPServer(("127.0.0.1", 0), FakeGitHub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield GitHub(
        "o/r",
        token="t0ken",  # noqa: S106 (the fake server's)
        api_url=f"http://127.0.0.1:{server.server_port}",
    )
    server.shutdown()


def _api_issue(number: int, labels: list[str], **extra: object) -> dict[str, Any]:
    return {
        "number": number,
        "state": "open",
        "title": f"issue {number}",
        "body": None,
        "labels": [{"name": n} for n in labels],
        **extra,
    }


def test_open_issues_leaves_out_pull_requests(github: GitHub) -> None:
    FakeGitHub.issues = [
        _api_issue(1, ["selfheal", "selfheal:bbc.com"]),
        _api_issue(2, ["selfheal"], pull_request={"url": "..."}),
        _api_issue(3, ["bug"]),
    ]

    (issue,) = github.open_issues()

    assert issue == Issue(
        number=1,
        title="issue 1",
        body="",
        labels=["selfheal", "selfheal:bbc.com"],
    )


def test_open_issues_reads_every_page(github: GitHub) -> None:
    FakeGitHub.issues = [_api_issue(n, ["selfheal"]) for n in range(1, 151)]

    assert len(github.open_issues()) == 150


def test_labelled_reads_every_page_in_any_state(github: GitHub) -> None:
    FakeGitHub.issues = [
        _api_issue(n, ["selfheal", "selfheal:bbc.com"], state=state)
        for n, state in enumerate(["open", "closed"] * 75, start=1)
    ] + [_api_issue(151, ["selfheal", "selfheal:cnn.com"])]

    found = github.labelled("selfheal:bbc.com")

    assert [item["number"] for item in found] == list(range(1, 151))
    paths = [path for method, path, _ in FakeGitHub.requests if method == "GET"]
    assert paths == [
        f"/repos/o/r/issues?state=all&labels=selfheal%3Abbc.com&per_page=100&page={p}"
        for p in (1, 2)
    ]


def test_comments_reads_every_page(github: GitHub) -> None:
    FakeGitHub.comments = {7: [{"body": f"c{n}"} for n in range(200)]}

    found = github.comments(7)

    assert [c["body"] for c in found] == [f"c{n}" for n in range(200)]
    paths = [path for method, path, _ in FakeGitHub.requests if method == "GET"]
    assert paths == [
        f"/repos/o/r/issues/7/comments?&per_page=100&page={p}" for p in (1, 2, 3)
    ]


def test_a_missing_label_is_created_once(github: GitHub) -> None:
    github.ensure_label("selfheal")
    github.ensure_label("selfheal:bbc.com")
    github.ensure_label("selfheal:bbc.com")

    posts = [r for r in FakeGitHub.requests if r[0] == "POST"]
    assert [(path, payload["name"]) for _, path, payload in posts] == [
        ("/repos/o/r/labels", "selfheal:bbc.com")
    ]


def test_the_writes_reach_the_issue(github: GitHub) -> None:
    number = github.create_issue("t", "b", ["selfheal"])
    github.update_body(number, "b2")
    github.comment(number, "recovered")
    github.close(number)

    assert FakeGitHub.issues[0]["body"] == "b2"
    assert FakeGitHub.issues[0]["state"] == "closed"
    assert FakeGitHub.issues[0]["state_reason"] == "completed"
    assert ("POST", "/repos/o/r/issues/1/comments", {"body": "recovered"}) in (
        FakeGitHub.requests
    )


def test_triage_end_to_end(tmp_path: Path, github: GitHub) -> None:
    write_health(tmp_path, DAY - ONE_DAY, [_outlet()])
    write_health(tmp_path, DAY, [_outlet()])

    first = triage(tmp_path, github)
    second = triage(tmp_path, github)

    assert [type(a) for a in first] == [Open]
    assert [type(a) for a in second] == [Refresh]
    assert len(FakeGitHub.issues) == 1
    assert "selfheal:bbc.com" in FakeGitHub.labels
    # the refresh had nothing new to say, so it wrote nothing
    assert not [r for r in FakeGitHub.requests if r[0] == "PATCH"]


def test_a_dry_run_writes_nothing(tmp_path: Path, github: GitHub) -> None:
    write_health(tmp_path, DAY, [_outlet()])

    actions = triage(tmp_path, github, dry_run=True)

    assert [type(a) for a in actions] == [Open]
    assert {method for method, _, _ in FakeGitHub.requests} == {"GET"}
