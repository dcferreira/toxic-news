# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for healing missing days from the Wayback Machine, with no network.

The Wayback Machine is stood in for at its two calls: the availability API
that names a snapshot, and the GET that fetches it.
"""

import asyncio
import datetime
from pathlib import Path

import aiohttp
import pytest
from tenacity import RetryError, wait_none

from tests.fixtures import latest_fixture, slug_of
from toxic_news.fetchers import Fetcher, WaybackFetcher
from toxic_news.main import _fetch_wayback, _noon, heal
from toxic_news.models import AllModels, Scores
from toxic_news.newspapers import Newspaper, newspapers

FOX = next(n for n in newspapers if n.name == "Fox News")
DAYS = [datetime.date(2026, 9, 5), datetime.date(2026, 9, 6)]
BOT_BLOCK = (
    b"<html><head><title>Access Denied</title></head>"
    b"<body>Your request is being blocked.</body></html>"
)


def _score_half(_self: AllModels, texts: list[str]) -> list[Scores]:
    """Score every text identically, so that no model has to be loaded."""
    return [Scores(**dict.fromkeys(Scores.__fields__, 0.5)) for _ in texts]


@pytest.fixture(autouse=True)
def _no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AllModels, "__init__", lambda _self: None)
    monkeypatch.setattr(AllModels, "predict", _score_half)


class _Archive:
    """What the availability API answers: a snapshot's URL and its time."""

    def __init__(self, when: datetime.datetime) -> None:
        self.when = when
        self.archive_url = (
            f"https://web.archive.org/web/{when:%Y%m%d%H%M%S}/https://foxnews.com"
        )

    def timestamp(self) -> datetime.datetime:
        # waybackpy hands back a naive datetime, in UTC
        return self.when


def _near(**kwargs: int) -> _Archive:
    # naive on purpose: that is what waybackpy returns
    naive = datetime.datetime(kwargs["year"], kwargs["month"], kwargs["day"])  # noqa: DTZ001
    return _Archive(naive)


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self) -> bytes:
        return self.body


class _Response:
    def __init__(self, body: bytes) -> None:
        self.status = 200
        self.charset = None
        self.content = _Content(body)


def _serve(pages: dict[str, bytes]):
    """Return a `ClientSession.get` serving `pages` by snapshot date."""

    async def get(_session: aiohttp.ClientSession, url: str, **_kwargs: object):
        # any day not given is archived as a bot block
        return _Response(
            next((b for day, b in pages.items() if f"/{day}" in url), BOT_BLOCK)
        )

    return get


def _fox_page(_assets: Path) -> bytes:
    return latest_fixture(slug_of(FOX)).path.read_bytes()


@pytest.fixture
def _wayback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "waybackpy.WaybackMachineAvailabilityAPI.near",
        lambda _self, **kwargs: _near(**kwargs),
    )
    # nothing to wait for between availability calls
    monkeypatch.setattr("toxic_news.fetchers.time.sleep", lambda _s: None)


@pytest.mark.usefixtures("_wayback")
@pytest.mark.asyncio
async def test_a_wayback_page_is_dated_in_utc() -> None:
    async with aiohttp.ClientSession() as session:
        fetcher = WaybackFetcher(date=_noon(DAYS[0]), newspaper=FOX, session=session)
        fetcher.get_wayback_url()

    assert fetcher.request_time == datetime.datetime(
        2026, 9, 5, tzinfo=datetime.timezone.utc
    )


@pytest.mark.usefixtures("_wayback")
def test_a_dated_outlet_heals_from_its_archived_page(
    assets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _fox_page(assets)
    monkeypatch.setattr(
        aiohttp.ClientSession, "get", _serve({"20260905": page, "20260906": page})
    )

    per_day = asyncio.run(_fetch_wayback(FOX, [_noon(d) for d in DAYS], None))

    assert [len(headlines) for headlines in per_day] == [125, 125]


def _failing_on(day: str):
    real = WaybackFetcher._request_coroutine

    async def request(self: WaybackFetcher):
        if self.date.strftime("%Y%m%d") == day:
            raise RetryError(last_attempt=None)  # ty: ignore[invalid-argument-type]
        return await real(self)

    return request


@pytest.mark.usefixtures("_wayback")
def test_a_day_the_wayback_machine_fails_on_is_skipped_not_fatal(
    assets: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        aiohttp.ClientSession, "get", _serve({"20260906": _fox_page(assets)})
    )
    monkeypatch.setattr(WaybackFetcher, "_request_coroutine", _failing_on("20260905"))

    # with a cache, which a failed fetch must not try to save into
    per_day = asyncio.run(
        _fetch_wayback(FOX, [_noon(d) for d in DAYS], tmp_path / "cache")
    )

    assert [len(headlines) for headlines in per_day] == [0, 125]


@pytest.mark.usefixtures("_wayback")
def test_an_archived_bot_block_page_heals_nothing(
    assets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        aiohttp.ClientSession,
        "get",
        _serve({"20260905": BOT_BLOCK, "20260906": _fox_page(assets)}),
    )

    per_day = asyncio.run(_fetch_wayback(FOX, [_noon(d) for d in DAYS], None))

    assert [len(headlines) for headlines in per_day] == [0, 125]


def _raising(*_args: object, **_kwargs: object) -> list[tuple[str, str]]:
    raise IndexError


@pytest.mark.usefixtures("_wayback")
def test_an_extractor_that_raises_on_an_archived_page_skips_that_day(
    assets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _fox_page(assets)
    monkeypatch.setattr(
        aiohttp.ClientSession, "get", _serve({"20260905": page, "20260906": page})
    )
    broken: Newspaper = FOX.copy(update={"get_headlines_fn": _raising})

    per_day = asyncio.run(_fetch_wayback(broken, [_noon(d) for d in DAYS], None))

    assert per_day == [[], []]


@pytest.mark.usefixtures("_wayback")
def test_heal_fills_the_days_it_could_fetch(
    assets: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page = _fox_page(assets)
    monkeypatch.setattr(
        aiohttp.ClientSession, "get", _serve({"20260905": page, "20260906": page})
    )
    monkeypatch.setattr(WaybackFetcher, "_request_coroutine", _failing_on("20260905"))
    monkeypatch.setattr("toxic_news.main.newspapers", [FOX])
    today = datetime.datetime.now(datetime.timezone.utc).date()
    out_dir = tmp_path / "public"

    heal(
        data_dir=tmp_path / "data",
        out_dir=out_dir,
        days=(today - DAYS[0]).days,
        allowed_difference_headlines=0.4,
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )

    assert (out_dir / "daily" / "2026" / "09" / "06.csv").exists()
    assert not (out_dir / "daily" / "2026" / "09" / "05.csv").exists()


def test_a_page_cached_with_a_naive_time_is_read_as_utc(
    tmp_path: Path,
) -> None:
    cached = Fetcher(newspaper=FOX, cache_dir=tmp_path)
    cached._content = b"<html></html>"
    cached._request_time = datetime.datetime(2026, 9, 5, 12)  # noqa: DTZ001 (as old caches hold)
    cached.save()

    fetcher = Fetcher(newspaper=FOX, cache_dir=tmp_path)
    assert fetcher.load(datetime.datetime(2026, 9, 5, tzinfo=datetime.timezone.utc))
    assert fetcher.request_time.tzinfo == datetime.timezone.utc


@pytest.mark.usefixtures("_wayback")
def test_a_model_failure_in_heal_is_not_taken_for_a_bad_snapshot(
    assets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _fox_page(assets)
    monkeypatch.setattr(
        aiohttp.ClientSession, "get", _serve({"20260905": page, "20260906": page})
    )

    def predict(_self: AllModels, _texts: list[str]) -> list[Scores]:
        msg = "the model fell over"
        raise RuntimeError(msg)

    monkeypatch.setattr(AllModels, "predict", predict)

    with pytest.raises(RuntimeError, match="the model fell over"):
        asyncio.run(_fetch_wayback(FOX, [_noon(d) for d in DAYS], None))


@pytest.mark.usefixtures("_wayback")
def test_a_get_that_keeps_failing_is_retried_then_skipped(
    assets: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the real tenacity retry, without its waits
    retrying = WaybackFetcher._request_coroutine.retry  # ty: ignore[unresolved-attribute]
    monkeypatch.setattr(retrying, "wait", wait_none())
    page = _fox_page(assets)
    served = _serve({"20260906": page})
    calls: list[str] = []

    async def get(session: aiohttp.ClientSession, url: str, **kwargs: object):
        calls.append(url)
        if "/20260905" in url:
            raise aiohttp.ClientConnectionError
        return await served(session, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "get", get)

    per_day = asyncio.run(
        _fetch_wayback(FOX, [_noon(d) for d in DAYS], tmp_path / "cache")
    )

    assert [len(headlines) for headlines in per_day] == [0, 125]
    assert sum("/20260905" in url for url in calls) == 8
