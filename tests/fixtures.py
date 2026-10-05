# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The recorded front pages, one per outlet per date, and their parse snapshots.

Each outlet keeps its pages under `tests/assets/html/<slug>/<YYYY-MM-DD>.html`,
where `<slug>` is its cleaned URL, and the headlines each one parses to under
`tests/snapshots/test_parse/<slug>/<YYYY-MM-DD>.txt`. A page is parsed as of
its own date, so a `DatedXpath` added for a new page is exercised by that page,
while the older ones keep proving that historical parsing is unchanged.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path

from toxic_news.fetchers import clean_url
from toxic_news.newspapers import Newspaper

_TESTS = Path(__file__).resolve().parent

#: The recorded front pages, one directory per outlet.
HTML_ROOT = _TESTS / "assets" / "html"

#: The headlines each recorded front page parses to, laid out like `HTML_ROOT`.
PARSE_SNAPSHOTS_ROOT = _TESTS / "snapshots" / "test_parse"

#: When the daily Update run fetches, which is when a fixture is read as of.
REQUEST_TIME_OF_DAY = time(12, tzinfo=timezone.utc)


def slug_of(newspaper: Newspaper) -> str:
    """Return the slug `newspaper`'s fixtures are filed under."""
    return clean_url(str(newspaper.url))


@dataclass(frozen=True, order=True)
class Fixture:
    """One outlet's front page, as recorded on one date."""

    slug: str
    date: date
    html_root: Path = field(default=HTML_ROOT, compare=False)
    snapshot_root: Path = field(default=PARSE_SNAPSHOTS_ROOT, compare=False)

    @property
    def path(self) -> Path:
        """Return the recorded HTML."""
        return self.html_root / self.slug / f"{self.date.isoformat()}.html"

    @property
    def snapshot_path(self) -> Path:
        """Return the headlines this page is recorded to parse to."""
        return self.snapshot_root / self.slug / f"{self.date.isoformat()}.txt"

    @property
    def request_time(self) -> datetime:
        """Return the time this page is parsed as of: noon UTC on its date.

        A `DatedXpath` applies to pages requested strictly after its
        `from_date`, which is 00:00 UTC on the first day it went live, so a
        page from that same day has to be read later than midnight.
        """
        return datetime.combine(self.date, REQUEST_TIME_OF_DAY)


#: A fixture is named after its date, as `YYYY-MM-DD` and nothing else: Python
#: also reads `YYYYMMDD` and week dates, which `Fixture.path` would not find.
_DATE_NAME = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_fixture_date(stem: str) -> date | None:
    """Return the date a fixture named `stem` was recorded on, if it is one."""
    if not _DATE_NAME.fullmatch(stem):
        return None
    try:
        return date.fromisoformat(stem)
    except ValueError:
        return None


def fixtures_for(
    slug: str,
    html_root: Path = HTML_ROOT,
    snapshot_root: Path = PARSE_SNAPSHOTS_ROOT,
) -> list[Fixture]:
    """Return every recorded page of the outlet `slug`, oldest first."""
    fixtures = [
        Fixture(slug=slug, date=day, html_root=html_root, snapshot_root=snapshot_root)
        for path in (html_root / slug).glob("*.html")
        if (day := parse_fixture_date(path.stem)) is not None
    ]
    return sorted(fixtures)


def all_fixtures(html_root: Path = HTML_ROOT) -> list[Fixture]:
    """Return every recorded page of every outlet, by outlet then date."""
    slugs = sorted(p.name for p in html_root.iterdir() if p.is_dir())
    return [f for slug in slugs for f in fixtures_for(slug, html_root)]


def _only(fixtures: list[Fixture], slug: str) -> list[Fixture]:
    if not fixtures:
        msg = f"No recorded front page for {slug!r}"
        raise FileNotFoundError(msg)
    return fixtures


def latest_fixture(slug: str, html_root: Path = HTML_ROOT) -> Fixture:
    """Return the most recent recorded page of the outlet `slug`."""
    return _only(fixtures_for(slug, html_root), slug)[-1]


def earliest_fixture(slug: str, html_root: Path = HTML_ROOT) -> Fixture:
    """Return the oldest recorded page of the outlet `slug`.

    The classification snapshots are kept for this page only: new pages are
    added for parsing, and scoring them would need the models.
    """
    return _only(fixtures_for(slug, html_root), slug)[0]
