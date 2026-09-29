# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the health a scrape records, outlet by outlet.

Every test here fetches over real HTTP from a local server — the mock site of
recorded front pages, or one that never answers — so what is recorded is what
`aiohttp` really reported, with no network beyond the loopback.
"""

import datetime
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import toxic_news.main
from tests.mock_site import serve
from toxic_news.fetchers import clean_url
from toxic_news.health import Verdict, read_health
from toxic_news.main import newspapers_from, scrape_newspapers, summary, update
from toxic_news.models import AllModels, Scores
from toxic_news.newspapers import Newspaper, newspapers


def _score_half(_self: AllModels, texts: list[str]) -> list[Scores]:
    """Score every text identically, so that no model has to be loaded."""
    return [Scores(**dict.fromkeys(Scores.__fields__, 0.5)) for _ in texts]


@pytest.fixture(autouse=True)
def _no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AllModels, "__init__", lambda _self: None)
    monkeypatch.setattr(AllModels, "predict", _score_half)


@pytest.fixture
def empty_site_url(tmp_path: Path) -> Iterator[str]:
    """The origin of a mock site with no recorded pages: every outlet 404s."""
    root = tmp_path / "empty"
    root.mkdir()
    server, origin = serve(root)
    yield origin
    server.shutdown()
    server.server_close()


@pytest.fixture
def silent_site_url() -> Iterator[str]:
    """The origin of a server that accepts connections and never answers."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    accepted: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        listener.settimeout(0.1)
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            accepted.append(connection)

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    stop.set()
    thread.join()
    for connection in accepted:
        connection.close()
    listener.close()


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def test_update_records_the_health_of_every_outlet(
    assets: Path, mock_site_url: str, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    update(data_dir=data_dir, out_dir=tmp_path / "public", base_url=mock_site_url)

    health = read_health(data_dir, _today())
    assert [outlet.newspaper for outlet in health] == [n.name for n in newspapers]
    for outlet, newspaper in zip(health, newspapers, strict=True):
        fixture = assets / "html" / f"{clean_url(str(newspaper.url))}.html"
        assert outlet.status == 200
        assert outlet.fetch_error is None
        assert outlet.body_bytes == fixture.stat().st_size
        assert outlet.expected == newspaper.expected_headlines
    # the recorded pages all still parse
    assert {outlet.verdict for outlet in health} == {Verdict.OK}


def test_update_records_health_even_when_nothing_was_scraped(
    empty_site_url: str, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    update(data_dir=data_dir, out_dir=tmp_path / "public", base_url=empty_site_url)

    health = read_health(data_dir, _today())
    assert len(health) == len(newspapers)
    assert {outlet.status for outlet in health} == {404}
    assert {outlet.verdict for outlet in health} == {Verdict.OTHER}
    assert not (data_dir / "headlines").exists()


def test_update_saves_every_fetched_page_when_asked(
    assets: Path, mock_site_url: str, tmp_path: Path
) -> None:
    raw_html_dir = tmp_path / "raw-html"
    update(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "public",
        base_url=mock_site_url,
        raw_html_dir=raw_html_dir,
    )

    for newspaper in newspapers:
        name = f"{clean_url(str(newspaper.url))}.html"
        recorded = (assets / "html" / name).read_bytes()
        assert (raw_html_dir / name).read_bytes() == recorded


def _raising(*_args: object, **_kwargs: object) -> list[tuple[str, str]]:
    """An extractor whose XPath misses the link, as `res[0]` does in the wild."""
    raise IndexError


def test_an_extractor_that_raises_breaks_only_its_own_outlet(
    mock_site_url: str,
) -> None:
    good, other = newspapers_from(mock_site_url)[:2]
    broken = other.copy(update={"get_headlines_fn": _raising})

    scrape = scrape_newspapers(outlets=[good, broken])

    assert {headline.newspaper for headline in scrape.headlines} == {good.name}
    good_health, broken_health = scrape.health
    assert good_health.parse_error is None
    assert broken_health.parse_error == "IndexError()"
    assert broken_health.headlines == 0
    assert broken_health.verdict == Verdict.XPATH


def test_an_outlet_that_never_answers_times_out(
    monkeypatch: pytest.MonkeyPatch, silent_site_url: str
) -> None:
    monkeypatch.setattr(toxic_news.main, "FETCH_TIMEOUT_SECONDS", 0.5)
    silent: Newspaper = newspapers_from(silent_site_url)[0]

    scrape = scrape_newspapers(outlets=[silent])

    (health,) = scrape.health
    assert health.status is None
    assert health.fetch_error is not None
    assert "TimeoutError" in health.fetch_error
    assert health.verdict == Verdict.OTHER


def test_summary_reports_each_outlets_status_and_verdict(
    empty_site_url: str, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    update(data_dir=data_dir, out_dir=tmp_path / "public", base_url=empty_site_url)
    report = tmp_path / "summary.md"

    summary(data_dir=data_dir, out_dir=tmp_path / "public", report=report)

    text = report.read_text()
    assert "| outlet | status | headlines | expected | verdict |" in text
    for newspaper in newspapers:
        expected = newspaper.expected_headlines
        assert f"| {newspaper.name} | 404 | 0 | {expected} | other |" in text


def test_summary_says_so_when_the_day_has_no_health_report(tmp_path: Path) -> None:
    report = tmp_path / "summary.md"

    summary(data_dir=tmp_path / "data", out_dir=tmp_path / "public", report=report)

    text = report.read_text()
    assert "No health report" in text
    assert "| outlet |" not in text
