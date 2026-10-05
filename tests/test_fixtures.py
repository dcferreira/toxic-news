# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the layout of the dated front-page fixtures."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from tests.fixtures import (
    HTML_ROOT,
    PARSE_SNAPSHOTS_ROOT,
    Fixture,
    all_fixtures,
    fixtures_for,
    latest_fixture,
    slug_of,
)
from toxic_news.fetchers import clean_url
from toxic_news.newspapers import newspapers


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html></html>")
    return path


def test_fixtures_are_read_at_noon_utc_of_their_date():
    """A fixture is parsed as of the daily Update run, at 12:00 UTC.

    A `DatedXpath` goes live at 00:00 UTC on its first day, and only applies to
    pages requested strictly after that, so a fixture dated that same day has to
    be read later than midnight to exercise it.
    """
    fixture = Fixture(slug="bbc.com", date=date(2026, 9, 20), html_root=HTML_ROOT)
    assert fixture.request_time == datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def test_fixture_paths_are_per_outlet_and_per_date(tmp_path):
    fixture = Fixture(slug="bbc.com", date=date(2026, 9, 20), html_root=tmp_path)
    assert fixture.path == tmp_path / "bbc.com" / "2026-09-20.html"
    assert fixture.snapshot_path == (
        PARSE_SNAPSHOTS_ROOT / "bbc.com" / "2026-09-20.txt"
    )


def test_fixtures_for_lists_an_outlets_dates_oldest_first(tmp_path):
    _touch(tmp_path / "bbc.com" / "2026-09-20.html")
    _touch(tmp_path / "bbc.com" / "2023-05-20.html")
    _touch(tmp_path / "foxnews.com" / "2024-01-01.html")

    found = fixtures_for("bbc.com", html_root=tmp_path)

    assert [f.date for f in found] == [date(2023, 5, 20), date(2026, 9, 20)]
    assert {f.slug for f in found} == {"bbc.com"}
    assert latest_fixture("bbc.com", html_root=tmp_path).date == date(2026, 9, 20)


def test_fixtures_for_ignores_files_not_named_by_date(tmp_path):
    _touch(tmp_path / "bbc.com" / "2026-09-20.html")
    _touch(tmp_path / "bbc.com" / "notes.html")
    _touch(tmp_path / "bbc.com" / "2026-09-21.txt")
    _touch(tmp_path / "bbc.com" / "20260922.html")
    _touch(tmp_path / "bbc.com" / "2026-W39-3.html")

    assert [f.date for f in fixtures_for("bbc.com", html_root=tmp_path)] == [
        date(2026, 9, 20)
    ]


def test_latest_fixture_fails_for_an_outlet_without_fixtures(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"bbc\.com"):
        latest_fixture("bbc.com", html_root=tmp_path)


def test_every_newspaper_has_a_fixture_with_a_snapshot():
    """The recorded fixtures cover every outlet, each with its parse snapshot."""
    fixtures = all_fixtures()
    assert {f.slug for f in fixtures} == {slug_of(n) for n in newspapers}
    for fixture in fixtures:
        assert fixture.path.is_file()
        assert fixture.snapshot_path.is_file(), fixture.snapshot_path


def test_slug_of_is_the_cleaned_url():
    for newspaper in newspapers:
        assert slug_of(newspaper) == clean_url(str(newspaper.url))
