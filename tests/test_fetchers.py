import asyncio
import json
import re
from datetime import datetime

import aiohttp
import pytest
from fastapi.encoders import jsonable_encoder

from toxic_news.fetchers import Fetcher, Newspaper, WaybackFetcher, newspapers


def clean_url(url: str) -> str:
    """Removes the slashes from a URL, to make sure it's safe to use as a filename."""
    assert "http" in url
    out = re.sub("(http|https)://", "", url).strip("/")
    assert "/" not in out
    return out


def make_mock_fetcher(monkeypatch, url, assets):
    # mock content to use the HTML in the assets
    with open(assets / "html" / f"{clean_url(url)}.html", "r") as fd:
        monkeypatch.setattr(Fetcher, "content", fd.read())

    # set request time to a fixed value
    monkeypatch.setattr(Fetcher, "request_time", datetime(2000, 1, 1))


@pytest.mark.parametrize("newspaper", newspapers)
def test_parse(assets, snapshot, monkeypatch, newspaper):
    make_mock_fetcher(monkeypatch, newspaper.url, assets)

    fetcher = Fetcher(newspaper)
    snapshot.snapshot_dir = assets / "../snapshots/test_parse"
    snapshot.assert_match(
        json.dumps(fetcher.fetch(), indent=2), f"{clean_url(newspaper.url)}.txt"
    )


@pytest.mark.parametrize("newspaper", newspapers)
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
@pytest.mark.parametrize("newspaper", newspapers)
def test_classify(assets, snapshot, monkeypatch, newspaper):
    make_mock_fetcher(monkeypatch, newspaper.url, assets)

    fetcher = Fetcher(newspaper)
    snapshot.snapshot_dir = assets / "../snapshots/test_classify"
    snapshot.assert_match(
        json.dumps(jsonable_encoder(fetcher.classify()), indent=2),
        f"{clean_url(newspaper.url)}.txt",
    )


@pytest.mark.asyncio
async def test_wayback_integration():
    newspaper = Newspaper.parse_obj(
        {
            "name": "BBC",
            "language": "en",
            "url": "https://bbc.com",
            "xpath": "//h3[@class='media__title']/a",
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
            assert r.status == 200
        for f in fetchers:
            assert f._response is not None
            assert isinstance(await f.get_async_content(), bytes)


def test_wayback_sync():
    newspaper = Newspaper.parse_obj(
        {
            "name": "BBC",
            "language": "en",
            "url": "https://bbc.com",
            "xpath": "//h3[@class='media__title']/a",
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
        assert isinstance(f.content, bytes)
