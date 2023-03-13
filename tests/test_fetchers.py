import json
import re

import pytest

from toxic_news.fetchers import Fetcher, newspapers


def clean_url(url: str) -> str:
    """Removes the slashes from a URL, to make sure it's safe to use as a filename."""
    assert "http" in url
    out = re.sub("(http|https)://", "", url).strip("/")
    assert "/" not in out
    return out


@pytest.mark.parametrize("newspaper", newspapers)
def test_parse(assets, snapshot, monkeypatch, newspaper):
    def mockfetch() -> str:
        with open(assets / "html" / f"{clean_url(newspaper.url)}.html", "r") as fd:
            return fd.read()

    monkeypatch.setattr(Fetcher, "content", mockfetch())
    snapshot.snapshot_dir = assets / "../snapshots/test_parse"
    snapshot.assert_match(
        json.dumps(newspaper.fetch(), indent=2), f"{clean_url(newspaper.url)}.txt"
    )
