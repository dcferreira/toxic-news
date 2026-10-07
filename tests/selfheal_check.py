# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""`poe selfheal-check <outlet>`: the gate a self-fix of one outlet has to pass.

A self-fix edits one outlet's extractor in `toxic_news/newspapers.py` and adds
a fixture of today's front page with its parse snapshot. This checks, against
the working tree, that:

0. the fix adds exactly one page, newer than the outlet's others, and (given
   `--page`) byte for byte the page the run fetched;
1. on that page, the headline count is within 0.6-1.4x of the outlet's
   `expected_headlines`;
2. those headlines look real: none empty or over 300 characters, nearly all
   unique, almost none navigation text, and most linking to the outlet;
3. every fixture of every outlet still parses exactly to its snapshot, so
   historical and Wayback parsing are unchanged;
4. the diff since `--base` (by default `HEAD`, so commit the fix only after
   checking it, or pass the commit it branched from) stays within that outlet:
   its `Newspaper(...)` entry, the helpers nothing else uses, no suppression
   comments, and its own new pages and snapshots;
5. its `expected_headlines` is unchanged;
6. lint and type checks pass.

This runs the working tree's code, fix included, so the run that decides
whether a fix is merged has to come from a trusted checkout of the checker.

It prints a Markdown report fit for a PR body, writes it to `--output` if
asked (outside the checkout, e.g. `$RUNNER_TEMP`), and exits non-zero if any
check failed. Nothing here touches the network or needs CI to run on the PR:
the self-fix workflow runs it itself.
"""

import argparse
import ast
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from tests.fixtures import (
    HTML_ROOT,
    PARSE_SNAPSHOTS_ROOT,
    Fixture,
    fixtures_for,
    parse_fixture_date,
    slug_of,
)
from toxic_news.fetchers import clean_url
from toxic_news.newspapers import Newspaper, newspapers

#: Where the outlets are configured, relative to the repository root.
NEWSPAPERS_PY = "toxic_news/newspapers.py"

#: Where an outlet's fixtures and their parse snapshots live.
HTML_DIR = PurePosixPath("tests/assets/html")
SNAPSHOT_DIR = PurePosixPath("tests/snapshots/test_parse")

#: The band the headline count has to fall in, as multiples of the expected one.
COUNT_BAND = (0.6, 1.4)

#: The longest a headline can be.
MAX_HEADLINE_CHARS = 300

#: The least share of headlines that have to be distinct. Front pages repeat a
#: story in a second slot now and then.
MIN_UNIQUE_SHARE = 0.9

#: The least share of headlines that have to link to the outlet itself. Front
#: pages link sister sites (pagesix.com, barrons.com, theathletic.com...) and
#: ads, which reach a fifth of the headlines on some.
MIN_ON_SITE_SHARE = 0.7

#: The most headlines that can look like navigation, as a share of them all.
MAX_NAVIGATION_SHARE = 0.1

#: Section and menu labels, which a too-broad XPath picks up instead of stories.
NAVIGATION_TEXT = frozenset(
    {
        "business",
        "culture",
        "entertainment",
        "health",
        "home",
        "latest",
        "live",
        "log in",
        "login",
        "menu",
        "more",
        "most popular",
        "news",
        "newsletters",
        "opinion",
        "politics",
        "read more",
        "science",
        "search",
        "sign in",
        "sign up",
        "sport",
        "sports",
        "subscribe",
        "tech",
        "technology",
        "travel",
        "us",
        "video",
        "videos",
        "watch",
        "weather",
        "world",
    }
)

#: The most headlines that can be promos for the outlet itself (newsletters,
#: apps, accounts, subscriptions) rather than stories, as a share of them all.
#: They link to the outlet and read like sentences, so nothing else catches
#: them; the first BBC dry run passed with 9 in 49.
MAX_UTILITY_SHARE = 0.05

#: Path segments of pages that sign readers up or in, rather than tell a story.
UTILITY_PATH_SEGMENTS = frozenset(
    {
        "account",
        "accounts",
        "log-in",
        "login",
        "myaccount",
        "newsletter",
        "newsletters",
        "register",
        "registration",
        "session",
        "sign-in",
        "signin",
        "subscribe",
        "subscription",
        "subscriptions",
    }
)

#: Subdomains that serve the same kind of page.
UTILITY_HOSTS = ("account.", "accounts.", "login.", "session.", "subscribe.")

#: How a call to action starts, as against a headline.
CALLS_TO_ACTION = re.compile(
    r"(sign (up|in)|log in|subscribe|download (the|our)|register (for|now)|"
    r"get the|stream the|join us|follow us|install)\b",
    re.IGNORECASE,
)

Headlines = Sequence[tuple[str, str]]

#: Runs a command, returning its exit code and combined output.
Runner = Callable[[list[str]], tuple[int, str]]


@dataclass(frozen=True, order=True)
class Change:
    """One file changed since the base: `A`dded, `M`odified or `D`eleted."""

    path: str
    status: str


@dataclass
class CheckResult:
    """The outcome of one check, with a one-line reason a reviewer can read."""

    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    """Every check run on one outlet, and the headlines its newest page gives."""

    newspaper: Newspaper
    fixture: Fixture
    results: list[CheckResult]
    headlines: list[tuple[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether every check passed."""
        return all(r.passed for r in self.results)

    def json(self) -> dict[str, object]:
        """Return the report as data, for the PR job to render."""
        return {
            "outlet": self.newspaper.name,
            "date": self.fixture.date.isoformat(),
            "passed": self.passed,
            "results": [
                {"name": r.name, "passed": r.passed, "detail": r.detail}
                for r in self.results
            ],
            "headlines": [list(h) for h in self.headlines],
        }

    def markdown(self) -> str:
        """Render the report as Markdown, for a job summary or a PR body."""
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            (
                f"### selfheal-check: {self.newspaper.name} "
                f"(`{self.fixture.slug}`): {verdict}"
            ),
            "",
            "| | Check | Result |",
            "|---|---|---|",
        ]
        lines += [
            f"| {'✅' if r.passed else '❌'} | {r.name} | {_cell(r.detail)} |"
            for r in self.results
        ]
        lines += [
            "",
            "<details>",
            (
                f"<summary>{len(self.headlines)} headlines on "
                f"{self.fixture.date.isoformat()}</summary>"
            ),
            "",
            "| # | Headline | Link |",
            "|---|---|---|",
        ]
        lines += [
            f"| {i} | {_cell(text)} | {_cell(url)} |"
            for i, (text, url) in enumerate(self.headlines, start=1)
        ]
        lines += ["", "</details>", ""]
        return "\n".join(lines)


#: Characters scraped text could use to break out of a table cell, or to
#: mention people, link issues, or embed HTML and links, in a PR body.
_ENTITIES = str.maketrans({c: f"&#{ord(c)};" for c in "&<>|@#[]`\\"})


def _cell(text: str) -> str:
    """Make `text` inert inside a Markdown table cell, scraped text included."""
    return " ".join(text.translate(_ENTITIES).split())


def _parse(newspaper: Newspaper, fixture: Fixture) -> list[tuple[str, str]]:
    """Return what `fixture` parses to, as of its own date, like `test_parse`."""
    return newspaper.get_headlines(
        fixture.path.read_text(), request_date=fixture.request_time
    )


# --- rule 1 -------------------------------------------------------------------


def check_count(headlines: Headlines, expected: int) -> CheckResult:
    """Check the headline count is within `COUNT_BAND` of `expected`."""
    low, high = (expected * k for k in COUNT_BAND)
    passed = low <= len(headlines) <= high
    return CheckResult(
        "Headline count",
        passed,
        f"{len(headlines)} headlines, expected {expected} "
        f"(allowed {low:g} to {high:g})",
    )


# --- rule 2 -------------------------------------------------------------------


def _site(url: str) -> str:
    """Return the host of `url`, without a leading `www.`."""
    return (urlsplit(url).hostname or "").removeprefix("www.")


def _on_site(link: str, site: str) -> bool:
    host = _site(link)
    return host == site or host.endswith(f".{site}")


def _looks_like_navigation(text: str) -> bool:
    words = text.split()
    return len(words) <= 1 or " ".join(words).lower() in NAVIGATION_TEXT


def is_utility(text: str, link: str) -> bool:
    """Return whether a headline is a promo: a newsletter, app, account or
    subscription link, or a call to action, rather than a story."""
    parts = urlsplit(link)
    host = (parts.hostname or "").removeprefix("www.")
    segments = {s.lower() for s in parts.path.split("/")}
    return (
        bool(segments & UTILITY_PATH_SEGMENTS)
        or host.startswith(UTILITY_HOSTS)
        or CALLS_TO_ACTION.match(text.strip()) is not None
    )


def check_headlines(headlines: Headlines, url: str) -> CheckResult:
    """Check the headlines look like real stories from the outlet at `url`."""
    name = "Headlines look real"
    if not headlines:
        return CheckResult(name, passed=False, detail="no headlines at all")

    site = _site(url)
    texts = [text for text, _ in headlines]
    problems = []
    if empty := sum(not t.strip() for t in texts):
        problems.append(f"{empty} empty")
    if long := sum(len(t) > MAX_HEADLINE_CHARS for t in texts):
        problems.append(f"{long} over {MAX_HEADLINE_CHARS} characters")
    unique = len(set(texts)) / len(texts)
    if unique < MIN_UNIQUE_SHARE:
        problems.append(f"only {unique:.0%} unique (need {MIN_UNIQUE_SHARE:.0%})")
    on_site = sum(_on_site(link, site) for _, link in headlines) / len(headlines)
    if on_site < MIN_ON_SITE_SHARE:
        problems.append(
            f"only {on_site:.0%} link to {site} (need {MIN_ON_SITE_SHARE:.0%})"
        )
    navigation = [t for t in texts if t.strip() and _looks_like_navigation(t)]
    if len(navigation) / len(texts) > MAX_NAVIGATION_SHARE:
        shown = ", ".join(repr(t) for t in navigation[:5])
        problems.append(f"{len(navigation)} look like navigation text: {shown}")

    promos = [t for t, link in headlines if is_utility(t, link)]
    if len(promos) > max(1, MAX_UTILITY_SHARE * len(texts)):
        shown = ", ".join(repr(t) for t in promos[:10])
        problems.append(
            f"{len(promos)} look like promos, not stories "
            f"(at most {MAX_UTILITY_SHARE:.0%}): {shown}"
        )

    if problems:
        return CheckResult(name, passed=False, detail="; ".join(problems))
    return CheckResult(
        name,
        passed=True,
        detail=f"{unique:.0%} unique, {on_site:.0%} link to {site}, "
        f"{len(navigation)} navigation-like, {len(promos)} promo-like",
    )


# --- rule 3 -------------------------------------------------------------------


def check_fixtures_reproduce(
    fixtures: Sequence[tuple[Newspaper, Fixture]],
) -> CheckResult:
    """Check every fixture still parses exactly to its recorded snapshot.

    Every outlet's fixtures are checked, not only the fixed one's: a fix must
    not change how anything else parses.
    """
    problems = []
    for newspaper, fixture in fixtures:
        day = f"{fixture.slug} {fixture.date.isoformat()}"
        if not fixture.snapshot_path.is_file():
            problems.append(f"{day} has no snapshot")
            continue
        parsed = json.dumps(_parse(newspaper, fixture), indent=2)
        if parsed != fixture.snapshot_path.read_text():
            problems.append(f"{day} no longer matches its snapshot")
    if problems:
        return CheckResult(
            "Fixtures reproduce", passed=False, detail="; ".join(problems)
        )
    outlets = len({f.slug for _, f in fixtures})
    return CheckResult(
        "Fixtures reproduce",
        passed=True,
        detail=f"all {len(fixtures)} fixtures of {outlets} outlet(s) unchanged",
    )


# --- rules 4 and 5 ------------------------------------------------------------


def _url_of(call: ast.Call) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "url" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return None


def _entries(tree: ast.Module) -> list[ast.Call]:
    """Return the `Newspaper(...)` calls in the module's `newspapers` list."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "newspapers" for t in node.targets
            )
            and isinstance(node.value, ast.List)
        ):
            return [e for e in node.value.elts if isinstance(e, ast.Call)]
    return []


def _entry(tree: ast.Module, slug: str) -> ast.Call | None:
    for call in _entries(tree):
        url = _url_of(call)
        if url is not None and clean_url(url) == slug:
            return call
    return None


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _span(node: ast.Call | Definition) -> range:
    """Return the 0-based line numbers `node` covers, with any decorators."""
    decorators = [] if isinstance(node, ast.Call) else node.decorator_list
    start = min([node.lineno, *(d.lineno for d in decorators)])
    return range(start - 1, node.end_lineno or node.lineno)


@dataclass
class _Layout:
    """How `newspapers.py` splits into one outlet's own code and the rest."""

    entry: ast.Call | None
    helpers: list[Definition]
    #: `ast.dump` of every other entry, by its URL, in order.
    other_entries: list[tuple[str | None, str]]
    #: `ast.dump` of every top-level statement that is not the outlet's.
    rest: list[str]

    @property
    def own_lines(self) -> set[int]:
        """Return the 0-based lines of the entry and of its own helpers."""
        if self.entry is None:
            return set()
        lines = set(_span(self.entry))
        for helper in self.helpers:
            lines |= set(_span(helper))
        return lines


def _is_newspapers(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "newspapers" for t in node.targets)
        and isinstance(node.value, ast.List)
    )


def _layout(tree: ast.Module, slug: str) -> _Layout:
    """Split `tree` into `slug`'s entry, the helpers only it uses, and the rest.

    A helper is the outlet's own when the entry names it and nothing else in the
    module does: not another entry, not another function, not module code.
    """
    entry = _entry(tree, slug)
    entries = _entries(tree)
    definitions = {
        node.name: node for node in tree.body if isinstance(node, Definition)
    }
    candidates = (_names(entry) if entry is not None else set()) & set(definitions)
    used_elsewhere = set().union(
        *(_names(e) for e in entries if e is not entry),
        *(
            _names(node)
            for node in tree.body
            if not _is_newspapers(node)
            and not (isinstance(node, Definition) and node.name in candidates)
        ),
    )
    own = candidates - used_elsewhere
    return _Layout(
        entry=entry,
        helpers=[definitions[name] for name in sorted(own)],
        other_entries=[(_url_of(e), ast.dump(e)) for e in entries if e is not entry],
        rest=[
            # the list itself is compared entry by entry, so only its target here
            ast.dump(
                node.targets[0]
                if isinstance(node, ast.Assign) and _is_newspapers(node)
                else node
            )
            for node in tree.body
            if not (isinstance(node, Definition) and node.name in own)
        ],
    )


#: Comments that would switch lint, formatting or type checks off.
_SUPPRESSION = re.compile(r"#.*\b(noqa|type:\s*ignore|ty:\s*ignore|fmt:|pragma)")


def _newspapers_problems(base: str, new: str, slug: str) -> list[str]:
    """Return how `new` changes `newspapers.py` beyond `slug`'s own code."""
    try:
        base_layout, new_layout = (_layout(ast.parse(t), slug) for t in (base, new))
    except SyntaxError as e:
        return [f"{NEWSPAPERS_PY} does not parse: {e}"]
    if new_layout.entry is None:
        return [f"{NEWSPAPERS_PY} no longer has an entry for {slug}"]
    problems = []
    if new_layout.other_entries != base_layout.other_entries:
        before = {url for url, _ in base_layout.other_entries}
        after = {url for url, _ in new_layout.other_entries}
        changed = sorted(
            str(url)
            for url in before | after
            if dict(base_layout.other_entries).get(url)
            != dict(new_layout.other_entries).get(url)
        )
        what = ", ".join(changed) if changed else "their order"
        problems.append(f"{NEWSPAPERS_PY} changes other outlets: {what}")
    if new_layout.rest != base_layout.rest:
        problems.append(f"{NEWSPAPERS_PY} changes code shared by other outlets")

    base_lines, new_lines = base.splitlines(), new.splitlines()
    base_own, new_own = base_layout.own_lines, new_layout.own_lines
    stray, suppressions = [], []
    matcher = difflib.SequenceMatcher(a=base_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed = [i for i in range(i1, i2) if base_lines[i].strip()]
        added = [j for j in range(j1, j2) if new_lines[j].strip()]
        if any(i not in base_own for i in removed) or any(
            j not in new_own for j in added
        ):
            stray.append(j1 + 1)
        suppressions += [j + 1 for j in added if _SUPPRESSION.search(new_lines[j])]
    if stray:
        where = ", ".join(map(str, stray))
        problems.append(f"{NEWSPAPERS_PY} changed outside {slug} (line {where})")
    if suppressions:
        where = ", ".join(map(str, suppressions))
        problems.append(f"{NEWSPAPERS_PY} adds a suppression comment (line {where})")
    return problems


def _new_fixture_date(change: Change, slug: str) -> tuple[str, date] | None:
    """Return the kind and date of the fixture or snapshot `change` adds for `slug`."""
    path = PurePosixPath(change.path)
    if change.status != "A" or path.parent.name != slug:
        return None
    for kind, root, suffix in (
        ("page", HTML_DIR, ".html"),
        ("snapshot", SNAPSHOT_DIR, ".txt"),
    ):
        if path.parent.parent == root and path.suffix == suffix:
            day = parse_fixture_date(path.stem)
            return None if day is None else (kind, day)
    return None


def check_scope(
    slug: str, changes: Sequence[Change], base_source: str, new_source: str
) -> CheckResult:
    """Check the change since the base is a fix of `slug` and nothing else.

    It has to edit `newspapers.py` only within the outlet's entry and the
    helpers only it uses, and may add its own dated pages and snapshots.
    """
    problems = []
    if Change(NEWSPAPERS_PY, "M") not in changes:
        problems.append(f"{NEWSPAPERS_PY} is not changed")
    for change in changes:
        if change == Change(NEWSPAPERS_PY, "M"):
            problems += _newspapers_problems(base_source, new_source, slug)
        elif _new_fixture_date(change, slug) is None:
            problems.append(f"{change.path} ({change.status})")
    if problems:
        return CheckResult(
            "Diff stays within the outlet",
            passed=False,
            detail="out of scope: " + "; ".join(problems),
        )
    return CheckResult(
        "Diff stays within the outlet",
        passed=True,
        detail=f"{len(changes)} changed file(s), all within {slug}",
    )


def added_pages(
    slug: str,
    changes: Sequence[Change],
    html_root: Path = HTML_ROOT,
    snapshot_root: Path = PARSE_SNAPSHOTS_ROOT,
) -> list[Fixture]:
    """Return the pages of `slug` that `changes` adds, oldest first."""
    days = sorted(
        found[1]
        for change in changes
        if (found := _new_fixture_date(change, slug)) and found[0] == "page"
    )
    return [Fixture(slug, day, html_root, snapshot_root) for day in days]


def check_new_page(
    added: Sequence[Fixture], fixtures: Sequence[Fixture], page: Path | None
) -> CheckResult:
    """Check the fix adds exactly one page, the newest, and the one fetched.

    `page` is the front page the run saved; when given, the new fixture has to
    be it byte for byte, so the fix is judged against what the site served.
    """
    name = "Today's page"
    if len(added) != 1:
        return CheckResult(
            name,
            passed=False,
            detail=f"expected exactly one new page, found {len(added)}",
        )
    (new,) = added
    if new.path.is_symlink() or not new.path.is_file():
        return CheckResult(
            name, passed=False, detail=f"{new.path.name} is a symlink or missing"
        )
    if fixtures and new != max(fixtures):
        return CheckResult(
            name,
            passed=False,
            detail=f"{new.date.isoformat()} is not the newest page "
            f"({max(fixtures).date.isoformat()} is)",
        )
    digest = hashlib.sha256(new.path.read_bytes()).hexdigest()[:12]
    if page is not None and page.read_bytes() != new.path.read_bytes():
        return CheckResult(
            name, passed=False, detail=f"{new.path.name} differs from {page.name}"
        )
    matched = f", identical to {page.name}" if page is not None else ""
    return CheckResult(
        name,
        passed=True,
        detail=f"{new.date.isoformat()} (sha256 {digest}){matched}",
    )


def _expected(source: str, slug: str) -> object:
    entry = _entry(ast.parse(source), slug)
    if entry is None:
        return None
    for keyword in entry.keywords:
        if keyword.arg == "expected_headlines":
            return ast.unparse(keyword.value)
    return None


def check_expected_unchanged(
    slug: str, base_source: str, new_source: str
) -> CheckResult:
    """Check the outlet's `expected_headlines` is what it was at the base."""
    before, after = _expected(base_source, slug), _expected(new_source, slug)
    if before is None or before != after:
        return CheckResult(
            "expected_headlines unchanged",
            passed=False,
            detail=f"was {before}, now {after}",
        )
    return CheckResult("expected_headlines unchanged", passed=True, detail=str(after))


# --- rule 6 -------------------------------------------------------------------

#: The lint and type checks `poe lint` and `poe types` run.
QUALITY_COMMANDS = (
    [sys.executable, "-m", "ruff", "check", "."],
    [sys.executable, "-m", "ruff", "format", "--check", "."],
    # against this interpreter's environment, wherever the checkout's `.venv` is
    [sys.executable, "-m", "ty", "check", "--python", sys.prefix],
)


def _run(cmd: list[str]) -> tuple[int, str]:
    done = subprocess.run(  # noqa: S603 (the fixed QUALITY_COMMANDS)
        cmd, capture_output=True, text=True, check=False
    )
    return done.returncode, done.stdout + done.stderr


def check_quality(run: Runner = _run) -> CheckResult:
    """Check lint, formatting and types all pass."""
    failures = []
    for cmd in QUALITY_COMMANDS:
        code, output = run(list(cmd))
        if code != 0:
            tail = " ".join(output.strip().splitlines()[-3:])
            failures.append(f"`{' '.join(cmd[2:])}` failed: {tail}")
    if failures:
        return CheckResult("Lint and types", passed=False, detail="; ".join(failures))
    return CheckResult("Lint and types", passed=True, detail="ruff and ty pass")


# --- the whole check ----------------------------------------------------------


def resolve_outlet(outlet: str) -> Newspaper:
    """Return the newspaper `outlet` names: its slug, its name, or a short name.

    The short name is the slug's first label without `www.`, e.g. `bbc` or
    `theguardian`.
    """
    wanted = outlet.strip().lower()
    for match in (
        lambda n: slug_of(n) == wanted,
        lambda n: n.name.lower() == wanted,
        lambda n: slug_of(n).removeprefix("www.").split(".")[0] == wanted,
    ):
        found = [n for n in newspapers if match(n)]
        if len(found) == 1:
            return found[0]
    known = ", ".join(slug_of(n) for n in newspapers)
    msg = f"Unknown outlet {outlet!r}; pick one of: {known}"
    raise SystemExit(msg)


def run_checks(  # noqa: PLR0913 (each is an input the gate is run against)
    newspaper: Newspaper,
    *,
    changes: Sequence[Change],
    base_source: str,
    new_source: str,
    quality: Runner | None = _run,
    page: Path | None = None,
    html_root: Path = HTML_ROOT,
    snapshot_root: Path = PARSE_SNAPSHOTS_ROOT,
) -> Report:
    """Run every check on a fix of `newspaper`; `quality=None` skips lint and types.

    The headline checks run on the page the fix adds, or on the outlet's newest
    page when it adds none.
    """
    slug = slug_of(newspaper)
    fixtures = fixtures_for(slug, html_root, snapshot_root)
    added = added_pages(slug, changes, html_root, snapshot_root)
    if len(added) == 1 and added[0].path.is_file():
        today = added[0]
    elif fixtures:
        today = fixtures[-1]
    else:
        msg = f"No recorded front page for {slug!r}"
        raise SystemExit(msg)
    headlines = _parse(newspaper, today)
    every_fixture = [
        (n, f)
        for n in newspapers
        for f in fixtures_for(slug_of(n), html_root, snapshot_root)
    ]
    results = [
        check_new_page(added, fixtures, page),
        check_count(headlines, newspaper.expected_headlines),
        check_headlines(headlines, str(newspaper.url)),
        check_fixtures_reproduce(every_fixture),
        check_scope(slug, changes, base_source, new_source),
        check_expected_unchanged(slug, base_source, new_source),
    ]
    if quality is not None:
        results.append(check_quality(quality))
    for result in results[1:3]:
        result.name = f"{result.name} ({today.date.isoformat()})"
    return Report(newspaper, today, results, headlines)


def _git(*args: str) -> str:
    """Run git in the working directory and return what it printed."""
    git = shutil.which("git")
    if git is None:
        msg = "git is needed to find what changed since the base"
        raise SystemExit(msg)
    return subprocess.run(  # noqa: S603 (git, with the revision the caller names)
        [git, *args], capture_output=True, text=True, check=True
    ).stdout


def changes_since(base: str) -> list[Change]:
    """Return the files the working tree changed since `base`, untracked included."""
    diff = _git("diff", "--name-status", "--no-renames", base, "--")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    changes = [
        Change(path, status[0])
        for status, path in (line.split("\t", 1) for line in diff.splitlines())
    ]
    changes += [Change(path, "A") for path in untracked.splitlines()]
    return changes


def source_at(base: str, path: str) -> str:
    """Return the contents of `path` at the revision `base`."""
    return _git("show", f"{base}:{path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Check one outlet's self-fix, print the report and return the exit code."""
    parser = argparse.ArgumentParser(prog="poe selfheal-check", description=__doc__)
    parser.add_argument("outlet", help="slug (bbc.com), name (BBC) or short name (bbc)")
    parser.add_argument(
        "--base", default="HEAD", help="the revision the fix is checked against"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the report here; keep it outside the checkout, or it "
        "counts as a change out of scope",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="also write the report here as JSON; outside the checkout too",
    )
    parser.add_argument(
        "--page",
        type=Path,
        help="the front page the run fetched; the new fixture has to be it",
    )
    parser.add_argument(
        "--skip-quality", action="store_true", help="skip lint and type checks"
    )
    args = parser.parse_args(argv)

    newspaper = resolve_outlet(args.outlet)
    report = run_checks(
        newspaper,
        changes=changes_since(args.base),
        base_source=source_at(args.base, NEWSPAPERS_PY),
        new_source=Path(NEWSPAPERS_PY).read_text(),
        quality=None if args.skip_quality else _run,
        page=args.page,
    )
    markdown = report.markdown()
    sys.stdout.write(markdown)
    if args.output is not None:
        args.output.write_text(markdown)
    if args.json is not None:
        args.json.write_text(json.dumps(report.json(), indent=2) + "\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
