# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for `poe selfheal-check`, the gate a self-fix has to pass."""

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from tests import selfheal_check
from tests.fixtures import Fixture, all_fixtures
from tests.selfheal_check import (
    MAX_UTILITY_SHARE,
    NEWSPAPERS_PY,
    Change,
    changes_since,
    check_count,
    check_expected_unchanged,
    check_fixtures_reproduce,
    check_headlines,
    check_new_page,
    check_quality,
    check_scope,
    is_utility,
    main,
    resolve_outlet,
    run_checks,
    source_at,
)
from toxic_news.newspapers import Newspaper, get_xpath_fn

SOURCE = """\
import re


def shared_fn(content, base_url, request_date):
    return []


def own_fn(content, base_url, request_date):
    return []


newspapers = [
    Newspaper(
        name="A",
        language="en",
        url="https://a.com",
        get_headlines_fn=get_xpath_fn("//h1", href_xpath="a"),
        expected_headlines=10,
        tags=shared_fn,
    ),
    Newspaper(
        name="B",
        language="en",
        url="https://www.b.com/",
        get_headlines_fn=own_fn,
        expected_headlines=20,
        tags=shared_fn,
    ),
]
"""


def _edit(source: str, old: str, new: str) -> str:
    assert old in source
    return source.replace(old, new, 1)


def _newspaper(url: str = "https://a.com", expected: int = 10) -> Newspaper:
    return Newspaper(
        name="A",
        language="en",
        url=url,
        get_headlines_fn=get_xpath_fn("//h1[a]", href_xpath="a"),
        expected_headlines=expected,
    )


def _headlines(n: int, domain: str = "a.com") -> list[tuple[str, str]]:
    return [(f"Headline number {i} is here", f"https://{domain}/{i}") for i in range(n)]


# --- rule 1: count ------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "passed"), [(5, False), (6, True), (14, True), (15, False)]
)
def test_count_must_be_within_the_expected_band(count, passed):
    result = check_count(_headlines(count), expected=10)
    assert result.passed is passed
    assert str(count) in result.detail


# --- rule 2: the headlines look real ----------------------------------------


def test_real_looking_headlines_pass():
    result = check_headlines(_headlines(20), url="https://a.com")
    assert result.passed, result.detail


def test_a_subdomain_of_the_outlet_counts_as_its_own():
    headlines = _headlines(20, domain="news.a.com")
    assert check_headlines(headlines, url="https://www.a.com").passed


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (("   ", "https://a.com/x"), "empty"),
        (("x" * 301, "https://a.com/x"), "300"),
    ],
)
def test_any_empty_or_overlong_headline_fails(bad, reason):
    result = check_headlines([*_headlines(20), bad], url="https://a.com")
    assert not result.passed
    assert reason in result.detail


def test_mostly_repeated_headlines_fail():
    headlines = [("Same headline every time", f"https://a.com/{i}") for i in range(10)]
    result = check_headlines(headlines, url="https://a.com")
    assert not result.passed
    assert "unique" in result.detail


def test_a_few_repeats_and_off_site_links_are_tolerated():
    """Real front pages repeat a headline or link to sister sites now and then."""
    headlines = [
        *_headlines(17),
        ("Headline number 0 is here", "https://a.com/again"),
        ("A sister site story", "https://sister.com/1"),
        ("Another sister site story", "https://sister.com/2"),
    ]
    assert check_headlines(headlines, url="https://a.com").passed


def test_mostly_off_site_links_fail():
    result = check_headlines(_headlines(20, domain="ads.example"), url="https://a.com")
    assert not result.passed
    assert "a.com" in result.detail


def test_navigation_text_fails():
    nav = [(text, f"https://a.com/{text}") for text in ("Home", "Sport", "Menu")]
    result = check_headlines([*_headlines(10), *nav], url="https://a.com")
    assert not result.passed
    assert "navigation" in result.detail


def test_no_headlines_at_all_fails():
    assert not check_headlines([], url="https://a.com").passed


#: Promos the first BBC dry run's XPath picked up alongside its stories.
BBC_PROMOS = [
    ("The best of the BBC, delivered to you", "https://www.bbc.com/newsletters?x=1"),
    ("US Politics Unspun", "http://www.bbc.com/newsletters?USElectionUnspun"),
    ("Sign up to World of Business", "http://www.bbc.com/newsletters?WorldOfBusiness"),
    ("Get The Essential List", "http://www.bbc.com/newsletters?TheEssentialList"),
    ("Stream the best of British TV", "https://www.britbox.com/?utm_source=bbc.com"),
    ("Download the BBC app", "https://www.bbc.com/pages/download-the-bbc-app"),
    ("Register for a BBC account", "https://session.bbc.com/session?userOrigin=x"),
]


@pytest.mark.parametrize(
    ("text", "link"),
    [
        *BBC_PROMOS,
        ("Subscribe for $1 a week", "https://a.com/offers/digital"),
        ("Log in to your account", "https://a.com/x"),
        ("Our morning briefing", "https://a.com/account/newsletters"),
        ("Breaking news alerts", "https://login.a.com/start"),
    ],
)
def test_utility_links_are_recognised(text, link):
    assert is_utility(text, link)


@pytest.mark.parametrize(
    ("text", "link"),
    [
        ("Watch: At the scene of student protests in Lille", "https://a.com/videos/1"),
        ("Apple's app store faces EU fine", "https://a.com/news/apple-app-store-fine"),
        ("Registered voters surge ahead of midterms", "https://a.com/news/voters"),
        ("Streaming wars: who is winning?", "https://a.com/business/streaming"),
        ("Download speeds lag in rural areas", "https://a.com/tech/broadband"),
        ("How to Sleep Better, according to science", "https://a.com/health/sleep"),
    ],
)
def test_stories_are_not_utility_links(text, link):
    assert not is_utility(text, link)


def test_promos_among_the_headlines_fail():
    """The first BBC dry run passed with 9 promos in 49; this rule catches it."""
    stories = _headlines(40, domain="bbc.com")
    result = check_headlines([*stories, *BBC_PROMOS], url="https://bbc.com")
    assert not result.passed
    assert "promos" in result.detail
    assert "Download the BBC app" in result.detail


def test_a_stray_promo_is_tolerated():
    result = check_headlines(
        [*_headlines(30), ("Sign up for our newsletter", "https://a.com/newsletters")],
        url="https://a.com",
    )
    assert result.passed, result.detail


@pytest.mark.parametrize("fixture", all_fixtures(), ids=lambda f: f.slug)
def test_no_recorded_snapshot_reads_as_promos(fixture):
    """Every outlet's own recorded headlines stay under the promo share."""
    headlines = json.loads(fixture.snapshot_path.read_text())
    if not headlines:
        pytest.skip("an outlet broken on its recorded day")
    promos = [h for h in headlines if is_utility(*h)]
    assert len(promos) / len(headlines) <= MAX_UTILITY_SHARE, promos


# --- rule 3: every fixture still reproduces its snapshot ----------------------


def _fixture(tmp_path: Path, day: date, html: str, snapshot: object) -> Fixture:
    fixture = Fixture(
        slug="a.com",
        date=day,
        html_root=tmp_path / "html",
        snapshot_root=tmp_path / "snapshots",
    )
    fixture.path.parent.mkdir(parents=True, exist_ok=True)
    fixture.path.write_text(html)
    if snapshot is not None:
        fixture.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        fixture.snapshot_path.write_text(json.dumps(snapshot, indent=2))
    return fixture


PAGE = (
    "<html><body>"
    "<h1><a href='/1'>One</a></h1><h1><a href='/2'>Two</a></h1>"
    "</body></html>"
)
PARSED = [["One", "https://a.com/1"], ["Two", "https://a.com/2"]]


def test_fixtures_matching_their_snapshots_pass(tmp_path):
    fixtures = [_fixture(tmp_path, date(2023, 5, 20), PAGE, PARSED)]
    result = check_fixtures_reproduce([(_newspaper(), f) for f in fixtures])
    assert result.passed, result.detail


def test_a_fixture_parsing_differently_fails_and_is_named(tmp_path):
    fixtures = [
        _fixture(tmp_path, date(2023, 5, 20), PAGE, PARSED),
        _fixture(tmp_path, date(2026, 9, 20), PAGE, PARSED[:1]),
    ]
    result = check_fixtures_reproduce([(_newspaper(), f) for f in fixtures])
    assert not result.passed
    assert "2026-09-20" in result.detail
    assert "2023-05-20" not in result.detail


def test_a_broken_fixture_of_another_outlet_fails_too(tmp_path):
    """A fix must not change how any outlet parses, not only its own."""
    mine = _fixture(tmp_path, date(2023, 5, 20), PAGE, PARSED)
    other = _fixture(tmp_path, date(2023, 5, 21), PAGE, PARSED[:1])
    result = check_fixtures_reproduce(
        [(_newspaper(), mine), (_newspaper("https://b.com"), other)]
    )
    assert not result.passed
    assert "2023-05-21" in result.detail


def test_a_fixture_without_a_snapshot_fails(tmp_path):
    fixtures = [_fixture(tmp_path, date(2026, 9, 20), PAGE, None)]
    result = check_fixtures_reproduce([(_newspaper(), f) for f in fixtures])
    assert not result.passed
    assert "snapshot" in result.detail


# --- rule 4: the diff stays within the outlet ---------------------------------


def _fix(slug: str, day: str = "2026-09-20") -> list[Change]:
    """The files a fix of `slug` touches: its entry, a new page and its snapshot."""
    return [
        Change(NEWSPAPERS_PY, "M"),
        Change(f"tests/assets/html/{slug}/{day}.html", "A"),
        Change(f"tests/snapshots/test_parse/{slug}/{day}.txt", "A"),
    ]


def _scope(slug: str, new: str, changes: list[Change] | None = None):
    changes = _fix(slug) if changes is None else changes
    return check_scope(slug, changes, base_source=SOURCE, new_source=new)


def test_an_edit_inside_the_outlets_entry_is_in_scope():
    new = _edit(SOURCE, '"//h1"', '"//h2"')
    assert _scope("a.com", new).passed
    result = _scope("www.b.com", new)
    assert not result.passed
    assert "newspapers.py" in result.detail


def test_an_edit_to_the_outlets_own_helper_is_in_scope():
    new = _edit(
        SOURCE,
        "def own_fn(content, base_url, request_date):\n    return []",
        "def own_fn(content, base_url, request_date):\n    return [1]",
    )
    assert _scope("www.b.com", new).passed
    assert not _scope("a.com", new).passed


def test_a_helper_shared_with_another_outlet_is_out_of_scope():
    new = _edit(
        SOURCE,
        "def shared_fn(content, base_url, request_date):\n    return []",
        "def shared_fn(content, base_url, request_date):\n    return [1]",
    )
    assert not _scope("a.com", new).passed


def test_a_new_helper_the_outlet_uses_is_in_scope():
    new = _edit(
        SOURCE,
        "\n\nnewspapers = [",
        "\n\ndef a_fn(content, base_url, request_date):\n    return []\n"
        "\n\nnewspapers = [",
    )
    new = _edit(
        new,
        'get_headlines_fn=get_xpath_fn("//h1", href_xpath="a")',
        "get_headlines_fn=a_fn",
    )
    assert _scope("a.com", new).passed


def test_an_edit_outside_any_entry_is_out_of_scope():
    new = _edit(SOURCE, "import re\n", "import os\nimport re\n")
    assert not _scope("a.com", new).passed


def test_stretching_the_entry_over_its_neighbour_is_out_of_scope():
    """Swallowing the next entry into a string keeps every changed line "inside"."""
    new = _edit(
        SOURCE,
        '        tags=shared_fn,\n    ),\n    Newspaper(\n        name="B"',
        '        tags=shared_fn,\n        notes="""\n    Newspaper(\n        name="B"',
    )
    new = _edit(
        new,
        "        tags=shared_fn,\n    ),\n]",
        '        tags=shared_fn,\n""",\n    ),\n]',
    )
    result = _scope("a.com", new)
    assert not result.passed
    assert "www.b.com" in result.detail


def test_reordering_the_outlets_is_out_of_scope():
    tree_a = SOURCE.index('    Newspaper(\n        name="A"')
    tree_b = SOURCE.index('    Newspaper(\n        name="B"')
    end = SOURCE.index("]\n", tree_b)
    new = SOURCE[:tree_a] + SOURCE[tree_b:end] + SOURCE[tree_a:tree_b] + SOURCE[end:]
    assert not _scope("a.com", new).passed


@pytest.mark.parametrize(
    "comment",
    ["# noqa: E501", "# type: ignore", "# ty: ignore[call-arg]", "# fmt: skip"],
)
def test_a_new_suppression_comment_is_out_of_scope(comment):
    new = _edit(
        SOURCE,
        'get_xpath_fn("//h1", href_xpath="a"),',
        f'get_xpath_fn("//h2", href_xpath="a"),  {comment}',
    )
    result = _scope("a.com", new)
    assert not result.passed
    assert "suppression" in result.detail


def test_a_helper_another_helper_calls_is_not_the_outlets_own():
    """`get_xpath_fn`-style helpers are shared even if one entry names them."""
    source = _edit(
        SOURCE,
        "\n\nnewspapers = [",
        "\n\ndef wrapper(content, base_url, request_date):\n"
        "    return own_fn(content, base_url, request_date)\n\n\nnewspapers = [",
    )
    new = _edit(
        source,
        "def own_fn(content, base_url, request_date):\n    return []",
        "def own_fn(content, base_url, request_date):\n    return [1]",
    )
    result = check_scope("www.b.com", _fix("www.b.com"), source, new)
    assert not result.passed


def test_a_fix_with_its_page_and_snapshot_is_in_scope():
    assert _scope("a.com", _edit(SOURCE, '"//h1"', '"//h2"')).passed


def test_a_fix_that_leaves_newspapers_py_alone_fails():
    result = _scope("a.com", SOURCE, _fix("a.com")[1:])
    assert not result.passed
    assert NEWSPAPERS_PY in result.detail


@pytest.mark.parametrize(
    "change",
    [
        Change("tests/assets/html/a.com/2023-05-20.html", "M"),
        Change("tests/snapshots/test_parse/a.com/2023-05-20.txt", "D"),
        Change("tests/assets/html/www.b.com/2026-09-20.html", "A"),
        Change("tests/assets/html/a.com/notes.html", "A"),
        Change("tests/snapshots/test_mock_classify/a.com.txt", "M"),
        Change("toxic_news/fetchers.py", "M"),
    ],
    ids=lambda c: f"{c.status}:{c.path}",
)
def test_any_other_file_change_is_out_of_scope(change):
    new = _edit(SOURCE, '"//h1"', '"//h2"')
    result = _scope("a.com", new, [*_fix("a.com"), change])
    assert not result.passed
    assert change.path in result.detail


# --- the new page -------------------------------------------------------------


def test_one_new_newest_page_passes(tmp_path):
    old = _fixture(tmp_path, date(2023, 5, 20), PAGE, PARSED)
    new = _fixture(tmp_path, date(2026, 9, 20), PAGE, PARSED)
    result = check_new_page([new], [old, new], page=None)
    assert result.passed, result.detail
    assert "2026-09-20" in result.detail


def test_no_new_page_fails(tmp_path):
    old = _fixture(tmp_path, date(2023, 5, 20), PAGE, PARSED)
    result = check_new_page([], [old], page=None)
    assert not result.passed
    assert "exactly one" in result.detail


def test_two_new_pages_fail(tmp_path):
    one = _fixture(tmp_path, date(2026, 9, 19), PAGE, PARSED)
    two = _fixture(tmp_path, date(2026, 9, 20), PAGE, PARSED)
    assert not check_new_page([one, two], [one, two], page=None).passed


def test_a_new_page_older_than_an_existing_one_fails(tmp_path):
    new = _fixture(tmp_path, date(2023, 5, 19), PAGE, PARSED)
    old = _fixture(tmp_path, date(2023, 5, 20), PAGE, PARSED)
    result = check_new_page([new], [new, old], page=None)
    assert not result.passed
    assert "newest" in result.detail


def test_a_symlinked_new_page_fails(tmp_path):
    old = _fixture(tmp_path, date(2023, 5, 20), PAGE, PARSED)
    new = Fixture("a.com", date(2026, 9, 20), tmp_path / "html", tmp_path / "snapshots")
    new.path.symlink_to(old.path)
    result = check_new_page([new], [old, new], page=None)
    assert not result.passed
    assert "symlink" in result.detail


def test_the_new_page_has_to_be_the_page_that_was_fetched(tmp_path):
    new = _fixture(tmp_path, date(2026, 9, 20), PAGE, PARSED)
    fetched = tmp_path / "today.html"
    fetched.write_text(PAGE)
    assert check_new_page([new], [new], page=fetched).passed
    fetched.write_text(PAGE + "<!-- changed -->")
    result = check_new_page([new], [new], page=fetched)
    assert not result.passed
    assert "today.html" in result.detail


# --- rule 5: expected_headlines is unchanged ----------------------------------


def test_an_unchanged_expected_count_passes():
    new = _edit(SOURCE, '"//h1"', '"//h2"')
    assert check_expected_unchanged("a.com", SOURCE, new).passed


def test_a_changed_expected_count_fails():
    new = _edit(SOURCE, "expected_headlines=10", "expected_headlines=3")
    result = check_expected_unchanged("a.com", SOURCE, new)
    assert not result.passed
    assert "10" in result.detail
    assert "3" in result.detail


# --- rule 6: lint and types ---------------------------------------------------


def test_quality_passes_when_every_command_succeeds():
    ran = []
    result = check_quality(lambda cmd: ran.append(cmd) or (0, ""))
    assert result.passed
    assert len(ran) == 3


def test_quality_fails_naming_the_failing_command():
    def run(cmd: list[str]) -> tuple[int, str]:
        return (1, "E501 line too long") if "format" in cmd else (0, "")

    result = check_quality(run)
    assert not result.passed
    assert "format" in result.detail
    assert "E501" in result.detail


# --- the whole check ----------------------------------------------------------


def test_resolve_outlet_takes_a_slug_or_a_name():
    assert resolve_outlet("bbc.com").name == "BBC"
    assert resolve_outlet("bbc").url == "https://bbc.com"
    assert resolve_outlet("the guardian (us)").url == "https://www.theguardian.com/us"
    with pytest.raises(SystemExit, match="no-such-outlet"):
        resolve_outlet("no-such-outlet")


BBC_FIX = (
    "get_headlines_fn=get_dated_xpath_fn(\n"
    "            DatedXpath(\n"
    "                from_date=None,\n"
    "                headline_xpath=\"//h3[@class='media__title' and a]\",\n"
    '                href_xpath="a",\n'
    "            ),\n"
    "        ),"
)


def _bbc_fix(tmp_path: Path) -> dict:
    """A fix of BBC, with today's page a copy of its recorded one."""
    html, snapshots = tmp_path / "html", tmp_path / "snapshots"
    recorded = Fixture("bbc.com", date(2023, 5, 20))
    today = Fixture("bbc.com", date(2026, 10, 5), html, snapshots)
    for fixture in (Fixture("bbc.com", date(2023, 5, 20), html, snapshots), today):
        fixture.path.parent.mkdir(parents=True, exist_ok=True)
        fixture.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        fixture.path.write_bytes(recorded.path.read_bytes())
        fixture.snapshot_path.write_bytes(recorded.snapshot_path.read_bytes())
    source = (Path(__file__).parents[1] / NEWSPAPERS_PY).read_text()
    new = _edit(
        source,
        "get_headlines_fn=get_xpath_fn(\n"
        '            "//h3[@class=\'media__title\' and a]", href_xpath="a"\n'
        "        ),",
        BBC_FIX,
    )
    return {
        "changes": _fix("bbc.com", "2026-10-05"),
        "base_source": source,
        "new_source": new,
        "quality": None,
        "html_root": html,
        "snapshot_root": snapshots,
    }


def test_a_fix_in_scope_passes_every_check(tmp_path):
    report = run_checks(resolve_outlet("bbc.com"), **_bbc_fix(tmp_path))
    assert report.passed, report.markdown()
    markdown = report.markdown()
    assert "BBC" in markdown
    assert "PASS" in markdown
    assert "2026-10-05" in markdown


def test_an_untouched_outlet_fails_for_want_of_a_fix():
    source = (Path(__file__).parents[1] / NEWSPAPERS_PY).read_text()
    report = run_checks(
        resolve_outlet("bbc.com"),
        changes=[],
        base_source=source,
        new_source=source,
        quality=None,
    )
    assert not report.passed
    assert "FAIL" in report.markdown()


@pytest.mark.parametrize(
    ("text", "unwanted"),
    [
        ("A | B", "A | B"),
        ("<img src=x>", "<img"),
        ("</details>", "</details>"),
        ("cc @dcferreira", "@dcferreira"),
        ("see #12", "#12"),
        ("[click](https://evil.example)", "](https"),
    ],
)
def test_the_report_neutralises_scraped_text(tmp_path, text, unwanted):
    report = run_checks(resolve_outlet("bbc.com"), **_bbc_fix(tmp_path))
    report.headlines = [(text, "https://bbc.com/1")]
    (row,) = [
        line for line in report.markdown().splitlines() if line.startswith("| 1 |")
    ]
    assert unwanted not in row


def _no_git(monkeypatch, *, passed: bool):
    """Stub out git, and have every check come back `passed`."""

    def run_checks(*_args, **_kwargs):
        return selfheal_check.Report(
            resolve_outlet("bbc.com"),
            Fixture("bbc.com", date(2023, 5, 20)),
            [selfheal_check.CheckResult("x", passed=passed, detail="")],
        )

    monkeypatch.setattr(selfheal_check, "changes_since", lambda _base: [])
    monkeypatch.setattr(selfheal_check, "source_at", lambda _base, _path: "")
    monkeypatch.setattr(selfheal_check, "run_checks", run_checks)


def test_main_writes_the_report_and_exits_zero_on_a_pass(monkeypatch, tmp_path, capsys):
    _no_git(monkeypatch, passed=True)
    out = tmp_path / "report.md"
    code = main(["bbc.com", "--skip-quality", "--output", str(out)])
    assert code == 0
    assert "PASS" in out.read_text()
    assert out.read_text() in capsys.readouterr().out


def test_main_exits_nonzero_on_a_failure(monkeypatch):
    _no_git(monkeypatch, passed=False)
    assert main(["bbc.com", "--skip-quality"]) == 1


# --- git plumbing -------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603


def test_changes_since_lists_edits_and_untracked_files(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "kept.txt").write_text("a")
    (tmp_path / "edited.txt").write_text("a")
    (tmp_path / "gone.txt").write_text("a")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "edited.txt").write_text("b")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / "new.txt").write_text("c")
    monkeypatch.chdir(tmp_path)

    assert sorted(changes_since("HEAD")) == [
        Change("edited.txt", "M"),
        Change("gone.txt", "D"),
        Change("new.txt", "A"),
    ]
    assert source_at("HEAD", "edited.txt") == "a"
