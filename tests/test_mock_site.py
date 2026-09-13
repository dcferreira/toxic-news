# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""End-to-end test of the pipeline against the recorded front pages.

Nothing here touches the network: `tests.mock_site` serves the fixtures the
outlet scrapers expect, and the scoring models are replaced with a constant, so
a whole `update` run — scrape, aggregate, rebuild, render — happens locally.
"""

import csv
import datetime
from collections import Counter
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

import pytest

from tests.mock_site import serve
from toxic_news.fetchers import clean_url
from toxic_news.main import newspapers_from, update
from toxic_news.models import AllModels, Scores
from toxic_news.newspapers import newspapers

# The published column order, pinned here rather than read from the pipeline so
# that a reordered column is a test failure rather than a moved assertion.
SCORE_FIELDS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
    "positive",
    "neutral",
    "negative",
)
RAW_HEADER = ("newspaper", "language", "date", "url", "text", *SCORE_FIELDS)
DAILY_HEADER = ("name", *SCORE_FIELDS, "count", "date")

RENDERED = ("index.html", "daily.html", "about.html")


class Response(NamedTuple):
    """One response from the mock site."""

    status: int
    content_type: str
    allow: str
    body: bytes


@pytest.fixture(scope="module")
def mock_site_url(assets) -> Iterator[str]:
    """The origin of a mock site serving the recorded front pages."""
    server, origin = serve(assets / "html")
    yield origin
    server.shutdown()
    server.server_close()


def _request(origin: str, path: str, method: str = "GET") -> Response:
    """Send one request to the mock site and return what it answered."""
    parts = urlsplit(origin)
    assert parts.hostname is not None  # the origin always comes from `serve`
    connection = HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return Response(
            status=response.status,
            content_type=response.headers.get("Content-Type", "") or "",
            allow=response.headers.get("Allow", "") or "",
            body=response.read(),
        )
    finally:
        connection.close()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return the header and the rows of one of the pipeline's CSVs."""
    with path.open(newline="") as fd:
        reader = csv.reader(fd)
        header = next(reader)
        return header, [dict(zip(header, row, strict=True)) for row in reader]


def _score_half(_self: AllModels, texts: list[str]) -> list[Scores]:
    """Score every text identically, so that no model has to be loaded.

    This replaces `AllModels.predict` — a plain function on the class, so it is
    called as a bound method and takes `self` first, exactly like the real one.
    """
    return [Scores(**dict.fromkeys(Scores.__fields__, 0.5)) for _ in texts]


def test_mock_site_serves_the_recorded_front_pages(assets, mock_site_url):
    """The mock returns a fixture byte for byte, and refuses anything else."""
    recorded = (assets / "html" / "bbc.com.html").read_bytes()

    served = _request(mock_site_url, "/bbc.com")

    assert served.status == 200
    assert served.body == recorded
    assert served.content_type == "text/html; charset=utf-8"

    missing = _request(mock_site_url, "/no-such-outlet")
    assert missing.status == 404
    assert "no-such-outlet" in missing.body.decode()

    head = _request(mock_site_url, "/bbc.com", method="HEAD")
    assert head.status == 200
    assert head.body == b""

    refused = _request(mock_site_url, "/bbc.com", method="POST")
    assert refused.status == 405
    assert refused.allow == "GET, HEAD"


def test_update_offline_scrapes_fixtures_and_publishes_the_day(
    mock_site_url, monkeypatch, tmp_path
):
    """A whole `update` run scrapes the fixtures, and writes only where asked."""
    # A stray write to a relative path then lands in `tmp_path`, where the last
    # assertion below sees it.
    monkeypatch.chdir(tmp_path)
    # Both models are replaced on the class, so every importer sees it.
    monkeypatch.setattr(AllModels, "__init__", lambda _self: None)
    monkeypatch.setattr(AllModels, "predict", _score_half)

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "public"
    today = datetime.datetime.now(datetime.timezone.utc).date()

    update(data_dir=data_dir, out_dir=out_dir, base_url=mock_site_url)

    # The raw store holds today's headlines, scraped from the recorded pages.
    raw_header, raw_rows = _read_csv(
        data_dir / "headlines" / f"{today:%Y}" / f"{today:%m}" / f"{today:%d}.csv"
    )
    assert tuple(raw_header) == RAW_HEADER
    assert len(raw_rows) > 1
    assert all(row["text"].strip() for row in raw_rows)
    assert {row["date"][:10] for row in raw_rows} == {today.isoformat()}
    outlets = {row["newspaper"] for row in raw_rows}
    assert len(outlets) > 1
    assert outlets <= {newspaper.name for newspaper in newspapers}

    # The published day is that raw day averaged per outlet, and nothing else.
    daily_header, daily_rows = _read_csv(
        out_dir / "daily" / f"{today:%Y}" / f"{today:%m}" / f"{today:%d}.csv"
    )
    assert tuple(daily_header) == DAILY_HEADER
    assert {row["name"] for row in daily_rows} == outlets
    per_outlet = Counter(row["newspaper"] for row in raw_rows)
    assert {row["name"]: int(row["count"]) for row in daily_rows} == dict(per_outlet)

    # And the site was rendered.
    assert set(RENDERED) <= {path.name for path in out_dir.glob("*.html")}

    # Nothing was written outside the two directories the run was given.
    written = {
        path.relative_to(tmp_path).parts[0]
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert written <= {"data", "public"}


def test_newspapers_from_accepts_a_mock_origin_without_a_tld():
    """A mock origin like `http://mock:8000` has no TLD, which `HttpUrl` rejects.

    The Docker runner reaches the mock by service name, so a single-label host
    has to survive the rewrite.
    """
    base = "http://mock:8000"
    rewritten = newspapers_from(base)
    assert len(rewritten) == len(newspapers)
    for original, mock in zip(newspapers, rewritten, strict=True):
        assert str(mock.url) == f"{base}/{clean_url(str(original.url))}"
        assert mock.name == original.name
        assert mock.expected_headlines == original.expected_headlines
