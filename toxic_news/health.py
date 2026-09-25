# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The per-outlet health of a run, and whether a broken outlet is an XPath error.

A run publishes whatever it scraped, so a broken outlet never fails it. The
health report is what tells the outlets apart: each one is `ok`, an `xpath`
error — the page was fetched fine but the extractor no longer matches it,
which is what the self-fix loop repairs — or `other`, for everything no code
change here can fix (the site blocking us, timing out, or serving a bot
challenge). Reports are kept one JSON file per day, under
``<data_dir>/health/YYYY/MM/DD.json`` on the ``data`` branch.
"""

import datetime
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, parse_obj_as

from toxic_news.newspapers import Newspaper

# An extractor finding a count outside this fraction of `expected_headlines`,
# either way, is broken; `test_parse_live` holds live pages to the same band.
ALLOWED_DIFFERENCE_HEADLINES = 0.4

# A fetched page smaller than this fraction of the outlet's last good page is
# taken for a stub (a bot wall, an error page), not the front page.
MIN_REFERENCE_SIZE_RATIO = 0.3

# Nor is a page under this size, whatever came before: an outlet broken since
# before its last good day has no reference, and real front pages run to
# hundreds of kilobytes.
MIN_PAGE_BYTES = 20_000

# Markers of the bot challenges outlets serve with a 200 status. They are kept
# specific on purpose: a plain "captcha" appears on ordinary front pages, in
# newsletter sign-up forms.
CHALLENGE_MARKERS = (
    b"<title>just a moment",  # Cloudflare
    b"__cf_chl_",  # Cloudflare
    b"<title>access denied",  # Akamai
    b"px-captcha",  # PerimeterX
    b"captcha-delivery.com",  # DataDome
)

HTTP_OK = 200


class Verdict(str, Enum):
    """What a run found for one outlet."""

    OK = "ok"
    XPATH = "xpath"
    OTHER = "other"


class OutletHealth(BaseModel):
    """What fetching and parsing one outlet's front page produced."""

    newspaper: str
    url: str
    status: int | None
    fetch_error: str | None
    parse_error: str | None
    body_bytes: int | None
    headlines: int
    expected: int
    verdict: Verdict

    class Config:
        """Pydantic options for `OutletHealth`."""

        use_enum_values = True


def headline_count_is_ok(
    found: int,
    expected: int,
    allowed_difference_headlines: float = ALLOWED_DIFFERENCE_HEADLINES,
) -> bool:
    """Return whether `found` headlines are within the tolerated band."""
    return (
        expected * (1 - allowed_difference_headlines)
        <= found
        <= expected * (1 + allowed_difference_headlines)
    )


def looks_like_challenge(body: bytes) -> bool:
    """Return whether `body` is a bot-challenge page rather than a front page."""
    lowered = body.lower()
    return any(marker in lowered for marker in CHALLENGE_MARKERS)


def _verdict(  # noqa: PLR0913 (every observation of the run bears on the verdict)
    *,
    status: int | None,
    fetch_error: str | None,
    body: bytes | None,
    headlines: int,
    parse_error: str | None,
    expected: int,
    reference_bytes: int | None,
) -> Verdict:
    if fetch_error is not None or body is None:
        return Verdict.OTHER
    if parse_error is None and headline_count_is_ok(headlines, expected):
        return Verdict.OK
    if status != HTTP_OK or looks_like_challenge(body) or len(body) < MIN_PAGE_BYTES:
        return Verdict.OTHER
    if (
        reference_bytes is not None
        and len(body) < reference_bytes * MIN_REFERENCE_SIZE_RATIO
    ):
        return Verdict.OTHER
    return Verdict.XPATH


def outlet_health(  # noqa: PLR0913 (every observation of the run bears on the verdict)
    newspaper: Newspaper,
    *,
    status: int | None,
    fetch_error: str | None,
    body: bytes | None,
    headlines: int,
    parse_error: str | None,
    reference_bytes: int | None,
) -> OutletHealth:
    """Return the health of `newspaper` given what fetching and parsing it gave.

    `reference_bytes` is the size of the outlet's last good page, if one is
    known; a page much smaller than it is not treated as the front page.
    """
    return OutletHealth(
        newspaper=newspaper.name,
        url=str(newspaper.url),
        status=status,
        fetch_error=fetch_error,
        parse_error=parse_error,
        body_bytes=None if body is None else len(body),
        headlines=headlines,
        expected=newspaper.expected_headlines,
        verdict=_verdict(
            status=status,
            fetch_error=fetch_error,
            body=body,
            headlines=headlines,
            parse_error=parse_error,
            expected=newspaper.expected_headlines,
            reference_bytes=reference_bytes,
        ),
    )


def health_path(data_dir: Path, date: datetime.date) -> Path:
    """Return the health report for `date`."""
    return (
        data_dir
        / "health"
        / f"{date.year:04d}"
        / f"{date.month:02d}"
        / f"{date.day:02d}.json"
    )


def write_health(
    data_dir: Path, date: datetime.date, report: Sequence[OutletHealth]
) -> None:
    """Write `date`'s health report, replacing whatever was there."""
    path = health_path(data_dir, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([outlet.dict() for outlet in report], indent=2) + "\n")


def read_health(data_dir: Path, date: datetime.date) -> list[OutletHealth]:
    """Return the health report for `date`, or nothing if the day is missing."""
    path = health_path(data_dir, date)
    if not path.exists():
        return []
    return parse_obj_as(list[OutletHealth], json.loads(path.read_text()))


def reference_sizes(
    data_dir: Path, date: datetime.date, lookback_days: int = 30
) -> dict[str, int]:
    """Return each outlet's latest good page size in the days before `date`."""
    sizes: dict[str, int] = {}
    for back in range(lookback_days, 0, -1):
        for outlet in read_health(data_dir, date - datetime.timedelta(days=back)):
            if outlet.verdict == Verdict.OK and outlet.body_bytes is not None:
                sizes[outlet.newspaper] = outlet.body_bytes
    return sizes
