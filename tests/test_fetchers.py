# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the newspaper fetchers, driven by recorded HTML fixtures."""

import asyncio
import json
from datetime import datetime, timezone

import aiohttp
import pytest
from aiohttp import ClientResponse

from tests.fixtures import Fixture, all_fixtures, earliest_fixture, slug_of
from toxic_news.fetchers import (
    Fetcher,
    Headline,
    Newspaper,
    WaybackFetcher,
    decode_page,
)
from toxic_news.models import AllModels, Scores
from toxic_news.newspapers import newspapers

DATE = datetime(2023, 5, 20, tzinfo=timezone.utc)

#: Every newspaper, keyed by the slug its fixtures are filed under.
_by_slug = {slug_of(n): n for n in newspapers}


def _clean_newspaper(n: Newspaper) -> str:
    return slug_of(n)


def _fixture_id(fixture: Fixture) -> str:
    return f"{fixture.slug}@{fixture.date.isoformat()}"


def make_mock_fetcher(monkeypatch, fixture: Fixture):
    """Serve `fixture` as the fetched page, requested at its own date."""
    monkeypatch.setattr(Fetcher, "content", fixture.path.read_text())
    monkeypatch.setattr(Fetcher, "request_time", fixture.request_time)


def test_fetcher_save_load(tmp_path):
    newspaper = newspapers[0]
    fake_time = DATE

    fetcher = Fetcher(newspaper=newspaper, cache_dir=tmp_path)
    fetcher._content = earliest_fixture(slug_of(newspaper)).path.read_bytes()
    fetcher._request_time = fake_time
    fetcher.save()

    fetcher2 = Fetcher(newspaper=newspaper, cache_dir=tmp_path)
    fetcher2.load(fake_time)

    assert fetcher2.request_time == fake_time
    assert fetcher2._content == fetcher._content


@pytest.mark.parametrize("fixture", all_fixtures(), ids=_fixture_id)
def test_parse(snapshot, monkeypatch, fixture):
    """Every fixture parses, at its own date, to the headlines recorded for it."""
    make_mock_fetcher(monkeypatch, fixture)

    fetcher = Fetcher(_by_slug[fixture.slug])
    snapshot.snapshot_dir = fixture.snapshot_path.parent
    content = fetcher.fetch()
    snapshot.assert_match(json.dumps(content, indent=2), fixture.snapshot_path.name)


@pytest.mark.integration
@pytest.mark.parametrize("newspaper", newspapers, ids=_clean_newspaper)
def test_parse_live(assets, snapshot, newspaper):
    # check live version of the website, and see if the xpath we have configured
    # still outputs around the same number of headlines
    fetcher = Fetcher(newspaper)
    live_headlines = fetcher.fetch()

    assert (
        newspaper.expected_headlines * 0.6
        <= len(live_headlines)
        <= newspaper.expected_headlines * 1.4
    )


def _remove_dates(
    headlines: list[Headline],
) -> list[dict[str, str | float | dict[str, float]]]:
    return [
        {
            k: v
            for k, v in headline.dict().items()
            if k != "date"  # ignore date so it generalizes for the future
        }
        for headline in headlines
    ]


@pytest.mark.parametrize("newspaper", newspapers, ids=_clean_newspaper)
def test_mock_classify(assets, snapshot, monkeypatch, newspaper):
    make_mock_fetcher(monkeypatch, earliest_fixture(slug_of(newspaper)))

    def mock_predict(self, texts) -> list[Scores]:
        return [Scores(**dict.fromkeys(Scores.__fields__, 0.5)) for _ in texts]

    # avoid initializing models
    monkeypatch.setattr(AllModels, "__init__", lambda _: None)
    monkeypatch.setattr(AllModels, "predict", mock_predict)

    fetcher = Fetcher(newspaper)
    snapshot.snapshot_dir = assets / "../snapshots/test_mock_classify"
    snapshot.assert_match(
        json.dumps(
            _remove_dates(headlines=fetcher.classify()),
            indent=2,
        ),
        f"{slug_of(newspaper)}.txt",
    )


class _StubContent:
    """The `read()`-able body of a stubbed `aiohttp` response."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self) -> bytes:
        """Return the stubbed body."""
        return self._content


class _StubResponse:
    """The `status`, `charset` and `content` a fetch reads off a response."""

    def __init__(self, content: bytes) -> None:
        self.status = 200
        self.charset = None
        self.content = _StubContent(content)


def _stub_session_get(content: bytes):
    """Return a `ClientSession.get` stand-in serving `content`, with no network."""

    async def get(_session, _url, **_kwargs):
        return _StubResponse(content)

    return get


@pytest.mark.asyncio
async def test_async_fetch_then_classify_and_cache_round_trip(tmp_path, monkeypatch):
    """`fetch_with` records one front page; classify reads it and a cache keeps it."""
    newspaper = newspapers[0]
    html = earliest_fixture(slug_of(newspaper)).path.read_bytes()
    monkeypatch.setattr(aiohttp.ClientSession, "get", _stub_session_get(html))

    def mock_predict(self, texts) -> list[Scores]:
        return [Scores(**dict.fromkeys(Scores.__fields__, 0.5)) for _ in texts]

    # avoid initializing models
    monkeypatch.setattr(AllModels, "__init__", lambda _: None)
    monkeypatch.setattr(AllModels, "predict", mock_predict)

    fetcher = Fetcher(newspaper=newspaper, cache_dir=tmp_path)
    assert fetcher.fetched is False

    async with aiohttp.ClientSession() as session:
        await fetcher.fetch_with(session)

    assert fetcher.fetched is True
    assert fetcher.content == html.decode()

    parsed = fetcher.parse(fetcher.content)
    assert parsed, "the recorded front page should still yield headlines"
    classified = fetcher.classify()
    assert [headline.text for headline in classified] == [text for text, _ in parsed]
    assert {headline.newspaper for headline in classified} == {newspaper.name}
    assert {headline.date for headline in classified} == {fetcher.request_time}
    assert {headline.scores.toxicity for headline in classified} == {0.5}

    fetcher.save()
    reloaded = Fetcher(newspaper=newspaper, cache_dir=tmp_path)
    assert reloaded.fetched is False
    assert reloaded.load(fetcher.request_time) is True
    assert reloaded.fetched is True
    assert reloaded.content == fetcher.content
    assert reloaded.request_time == fetcher.request_time


@pytest.mark.slow
@pytest.mark.parametrize("newspaper", newspapers, ids=_clean_newspaper)
def test_classify(assets, snapshot, monkeypatch, newspaper):
    make_mock_fetcher(monkeypatch, earliest_fixture(slug_of(newspaper)))

    # force rounding scores to 5 decimal places
    class RoundingFloat(float):
        __repr__ = staticmethod(lambda x: format(x, ".5f"))

    # typeshed doesn't declare these CPython implementation hooks, which
    # `RoundingFloat` monkeypatches in.
    json.encoder.c_make_encoder = None  # ty: ignore[unresolved-attribute]
    json.encoder.float = RoundingFloat  # ty: ignore[unresolved-attribute]

    fetcher = Fetcher(newspaper)
    snapshot.snapshot_dir = assets / "../snapshots/test_classify"
    snapshot.assert_match(
        json.dumps(_remove_dates(fetcher.classify()), indent=2),
        f"{slug_of(newspaper)}.txt",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wayback_integration():
    newspaper = newspapers[0]
    async with aiohttp.ClientSession() as session:
        fetchers = [
            WaybackFetcher(date=date, newspaper=newspaper, session=session)
            for date in [
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 1, 2, tzinfo=timezone.utc),
            ]
        ]
        tasks = [asyncio.create_task(f.run_request_coroutine()) for f in fetchers]
        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        for r in results:
            assert r is not None
            assert isinstance(r, bytes)
            assert len(r) > 0
        for f in fetchers:
            assert isinstance(f._response, ClientResponse)
            assert f._response.status == 200
            assert isinstance(await f.get_async_content(), bytes)


@pytest.mark.integration
def test_wayback_sync():
    newspaper = newspapers[0]
    fetchers = [
        WaybackFetcher(date=date, newspaper=newspaper)
        for date in [
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 2, tzinfo=timezone.utc),
        ]
    ]

    for f in fetchers:
        asyncio.run(f.run_request_coroutine())

    for f in fetchers:
        assert f._response is not None
        assert f._response.status == 200
        assert isinstance(f.content, str)


@pytest.mark.parametrize(
    ("body", "charset", "text"),
    [
        ("café".encode(), "utf-8", "café"),
        ("café".encode(), None, "café"),
        ("café".encode("latin-1"), "ISO-8859-1", "café"),
        # mislabelled the other way: UTF-8 served as Latin-1 is still read right
        ("café".encode(), "ISO-8859-1", "café"),
        # mislabelled: the bad byte is replaced rather than failing the page
        ("café".encode("latin-1"), "utf-8", "caf�"),
        ("café".encode("latin-1"), None, "caf�"),
        # an unknown charset falls back to utf-8
        ("café".encode(), "no-such-charset", "café"),
    ],
)
def test_decode_page(body: bytes, charset: str | None, text: str) -> None:
    assert decode_page(body, charset) == text
