import asyncio
import json
from datetime import datetime

import aiohttp
import pytest
from aiohttp import ClientResponse
from fastapi.encoders import jsonable_encoder

from toxic_news.fetchers import (
    DetoxifyResults,
    Fetcher,
    Newspaper,
    Scores,
    WaybackFetcher,
    clean_url,
    newspapers,
    parse_detoxify_scores,
)


def _clean_newspaper(n: Newspaper) -> str:
    return clean_url(n.url)


def test_parse_detoxify_scores():
    scores = DetoxifyResults(
        toxicity=[1, 2],
        severe_toxicity=[3, 4],
        obscene=[5, 6],
        identity_attack=[7, 8],
        insult=[9, 10],
        threat=[11, 12],
        sexual_explicit=[13, 14],
    )
    result = parse_detoxify_scores(scores)

    expected = [
        Scores(
            toxicity=1,
            severe_toxicity=3,
            obscene=5,
            identity_attack=7,
            insult=9,
            threat=11,
            sexual_explicit=13,
        ),
        Scores(
            toxicity=2,
            severe_toxicity=4,
            obscene=6,
            identity_attack=8,
            insult=10,
            threat=12,
            sexual_explicit=14,
        ),
    ]

    assert result == expected


def make_mock_fetcher(monkeypatch, url, assets):
    # mock content to use the HTML in the assets
    with open(assets / "html" / f"{clean_url(url)}.html", "r") as fd:
        monkeypatch.setattr(Fetcher, "content", fd.read())

    # set request time to a fixed value
    monkeypatch.setattr(Fetcher, "request_time", datetime(2000, 1, 1))


def test_fetcher_save_load(tmp_path, monkeypatch, assets):
    newspaper = newspapers[0]
    fake_time = datetime(2000, 1, 1)

    fetcher = Fetcher(newspaper=newspaper, cache_dir=tmp_path)
    with open(assets / "html" / f"{clean_url(newspaper.url)}.html", "rb") as fd:
        fetcher._content = fd.read()
    fetcher._request_time = fake_time
    fetcher.save()

    fetcher2 = Fetcher(newspaper=newspaper, cache_dir=tmp_path)
    fetcher2.load(fake_time)

    assert fetcher2.request_time == fake_time
    assert fetcher2._content == fetcher._content


@pytest.mark.parametrize("newspaper", newspapers, ids=_clean_newspaper)
def test_parse(assets, snapshot, monkeypatch, newspaper):
    make_mock_fetcher(monkeypatch, newspaper.url, assets)

    fetcher = Fetcher(newspaper)
    snapshot.snapshot_dir = assets / "../snapshots/test_parse"
    snapshot.assert_match(
        json.dumps(fetcher.fetch(), indent=2), f"{clean_url(newspaper.url)}.txt"
    )


@pytest.mark.integration
@pytest.mark.parametrize("newspaper", newspapers, ids=_clean_newspaper)
def test_parse_live(assets, snapshot, newspaper):
    # check live version of the website, and see if the xpath we have configured
    # still outputs around the same number of headlines
    fetcher = Fetcher(newspaper)
    live_headlines = fetcher.fetch()

    print(list(map(lambda x: x[0], live_headlines)))
    assert (
        newspaper.expected_headlines * 0.6
        <= len(live_headlines)
        <= newspaper.expected_headlines * 1.4
    )


@pytest.mark.parametrize("newspaper", newspapers, ids=_clean_newspaper)
def test_mock_classify(assets, snapshot, monkeypatch, newspaper):
    make_mock_fetcher(monkeypatch, newspaper.url, assets)

    class MockModel:
        def predict(self, texts):
            return {
                k: [0.5] * len(texts)
                for k in [
                    "toxicity",
                    "severe_toxicity",
                    "obscene",
                    "identity_attack",
                    "insult",
                    "threat",
                    "sexual_explicit",
                ]
            }

    monkeypatch.setattr(Fetcher, "model", MockModel())

    fetcher = Fetcher(newspaper)
    snapshot.snapshot_dir = assets / "../snapshots/test_mock_classify"
    snapshot.assert_match(
        json.dumps(jsonable_encoder(fetcher.classify()), indent=2),
        f"{clean_url(newspaper.url)}.txt",
    )


@pytest.mark.slow
@pytest.mark.parametrize("newspaper", newspapers, ids=_clean_newspaper)
def test_classify(assets, snapshot, monkeypatch, newspaper):
    make_mock_fetcher(monkeypatch, newspaper.url, assets)

    # force rounding scores to 5 decimal places
    class RoundingFloat(float):
        __repr__ = staticmethod(lambda x: format(x, ".5f"))  # type: ignore

    json.encoder.c_make_encoder = None  # type: ignore
    json.encoder.float = RoundingFloat  # type: ignore

    fetcher = Fetcher(newspaper)
    snapshot.snapshot_dir = assets / "../snapshots/test_classify"
    snapshot.assert_match(
        json.dumps(jsonable_encoder(fetcher.classify()), indent=2),
        f"{clean_url(newspaper.url)}.txt",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wayback_integration():
    newspaper = Newspaper.parse_obj(
        {
            "name": "BBC",
            "language": "en",
            "url": "https://bbc.com",
            "title_xpath": "//h3[@class='media__title']",
            "relative_href_xpath": "/a",
            "expected_headlines": 47,
        }
    )
    async with aiohttp.ClientSession() as session:
        fetchers = [
            WaybackFetcher(date=date, newspaper=newspaper, session=session)
            for date in [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        ]
        tasks = [asyncio.create_task(f.run_request_coroutine()) for f in fetchers]
        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        for r in results:
            assert r is not None
            assert isinstance(r, bytes)
            assert len(r) > 0  # type: ignore
        for f in fetchers:
            assert isinstance(f._response, ClientResponse)
            assert f._response.status == 200
            assert isinstance(await f.get_async_content(), bytes)


@pytest.mark.integration
def test_wayback_sync():
    newspaper = Newspaper.parse_obj(
        {
            "name": "BBC",
            "language": "en",
            "url": "https://bbc.com",
            "title_xpath": "//h3[@class='media__title']",
            "relative_href_xpath": "/a",
            "expected_headlines": 47,
        }
    )
    fetchers = [
        WaybackFetcher(date=date, newspaper=newspaper)
        for date in [datetime(2023, 1, 1), datetime(2023, 1, 2)]
    ]

    for f in fetchers:
        asyncio.run(f.run_request_coroutine())

    for f in fetchers:
        assert f._response is not None
        assert f._response.status == 200
        assert isinstance(f.content, str)
