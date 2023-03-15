import json
import re
from datetime import datetime

import pytest
from fastapi.encoders import jsonable_encoder

from toxic_news.fetchers import Fetcher, newspapers


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
