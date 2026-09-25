# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Scrape newspaper front pages and classify the headlines they publish."""

import asyncio
import atexit
import pickle
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import aiohttp
import nest_asyncio
from aiohttp import ClientResponse, ClientSession
from loguru import logger
from pydantic import BaseModel, HttpUrl, ValidationError
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
from waybackpy import WaybackMachineAvailabilityAPI

from toxic_news.models import AllModels, Scores
from toxic_news.newspapers import Newspaper

# allow nesting of loops
# removing this fails the test `test_wayback_sync`
# with `RuntimeError: Timeout context manager should be used inside a task`
nest_asyncio.apply()

user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": user_agent,
}


class Headline(BaseModel):
    """A scraped headline, with the model scores and the source it came from."""

    newspaper: str
    language: str
    text: str
    date: datetime
    scores: Scores
    url: HttpUrl | None = None


def validate_url(url: str) -> bool:
    """Return whether `url` is a valid HTTP(S) URL."""

    class UrlModel(BaseModel):
        url: HttpUrl

    try:
        UrlModel(url=url)
    except ValidationError:
        return False
    else:
        return True


class Fetcher:
    """Fetch, cache and classify the headlines of one newspaper front page."""

    def __init__(
        self,
        newspaper: Newspaper,
        cache_dir: Path | None = None,
        model: AllModels | None = None,
    ) -> None:
        """Set up a fetcher for `newspaper`, caching responses under `cache_dir`."""
        self.newspaper = newspaper
        self.cache_dir = cache_dir

        self._model: AllModels | None = model
        self._response: ClientResponse | None = None
        self._content: bytes | None = None
        self._request_time: datetime | None = None

    def _get_cache_filename(self, date: datetime) -> Path:
        if self.cache_dir is None:
            msg = "Cache path was not set!"
            raise RuntimeError(msg)
        return (
            self.cache_dir
            / clean_url(self.newspaper.url)
            / f"{date.strftime('%Y%m%d')}.pickle"
        )

    def save(self) -> None:
        """Write the fetched response and its request time to the cache."""
        if self.cache_dir is None:
            msg = f"Trying to save a website without setting a cache path! {self=}"
            raise ValueError(msg)
        if self._request_time is None:
            msg = f"Trying to save a website that wasn't yet fetched! {self=}"
            raise RuntimeError(msg)
        save_path = self._get_cache_filename(self._request_time)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving {self!r} to {save_path!r}")
        with save_path.open("wb+") as fd:
            pickle.dump(
                {"content": self._content, "request_time": self._request_time}, fd
            )

    def load(self, date: datetime) -> bool:
        """Load the cached response for `date`, reporting whether one was found."""
        if self.cache_dir is None:
            return False
        load_path = self._get_cache_filename(date)
        if load_path.exists():
            logger.debug(f"Loading {self.newspaper.url} from cache: {load_path}")
            with load_path.open("rb") as fd:
                d = pickle.load(fd)  # noqa: S301 (project's own local scrape cache)
                self._content = d["content"]
                self._request_time = d["request_time"]
            return True
        return False

    @property
    def model(self) -> AllModels:
        """Return the classification model, creating it on first use."""
        if self._model is None:
            self._model = AllModels()
        return self._model

    async def fetch_with(self, session: ClientSession) -> None:
        """Fetch this front page over `session`, recording the response and time.

        Sharing one session is what makes a whole day's scrape concurrent: the
        outlet front pages are fetched side by side rather than one after the
        other.
        """
        logger.debug(f"Fetching {self.newspaper.url!r}...")
        if self.newspaper.url == "https://newsmax.com":
            # newsmax requires http/2 when using our header,
            # but aiohttp doesn't support http/2
            result = await session.get(self.newspaper.url)
        else:
            result = await session.get(self.newspaper.url, headers=HEADERS)
        self._response = result
        self._content = await result.content.read()
        self._request_time = datetime.now(timezone.utc)
        logger.debug(f"{self.newspaper.url} fetched with code: {self._response.status}")

    @property
    def status(self) -> int | None:
        """Return the HTTP status the front page was served with, if it was fetched."""
        return None if self._response is None else self._response.status

    @property
    def body(self) -> bytes | None:
        """Return the raw bytes of the front page, if it was fetched."""
        return self._content

    @property
    def fetched(self) -> bool:
        """Return whether this front page has been fetched or loaded from cache."""
        return self._content is not None

    async def _request_coroutine(self) -> tuple[ClientResponse, bytes]:
        async with aiohttp.ClientSession() as session:
            await self.fetch_with(session)
        return cast("ClientResponse", self._response), cast("bytes", self._content)

    def _request(self) -> None:
        logger.debug(f"Fetching {self.newspaper.url!r}...")
        self._response, self._content = asyncio.run(self._request_coroutine())
        self._request_time = datetime.now(timezone.utc)
        if self.cache_dir is not None:
            self.save()
        logger.debug(f"{self.newspaper.url} fetched with code: {self._response.status}")

    @property
    def content(self) -> str:
        """Return the page content, fetching it on first access."""
        if self._content is None:
            self._request()
        return cast("bytes", self._content).decode()

    @property
    def request_time(self) -> datetime:
        """Return when the page was fetched, fetching it on first access."""
        if self._request_time is None:
            self._request()
        return cast("datetime", self._request_time)

    def parse(self, content: str) -> list[tuple[str, str]]:
        """Return the `(headline, url)` pairs found in `content`."""
        return self.newspaper.get_headlines(
            content=content, request_date=self.request_time
        )

    def fetch(self) -> list[tuple[str, str]]:
        """Fetch the front page and return its `(headline, url)` pairs."""
        return self.parse(self.content)

    def classify(self) -> list[Headline]:
        """Return the fetched headlines with their model scores."""
        content = self.fetch()
        if len(content) == 0:  # no content was found
            return []
        scores_list = self.model.predict([x[0] for x in content])
        return [
            Headline(
                newspaper=self.newspaper.name,
                language=self.newspaper.language,
                text=t,
                date=self.request_time,
                scores=s,
                url=cast("HttpUrl", u) if validate_url(u) else None,
            )
            for s, (t, u) in zip(scores_list, content, strict=False)
        ]


class WaybackFetcher(Fetcher):
    """Fetch a front page from the Wayback Machine snapshot nearest `date`."""

    def __init__(
        self,
        date: datetime,
        newspaper: Newspaper,
        session: ClientSession | None = None,
        cache_dir: Path | None = None,
        model: AllModels | None = None,
    ) -> None:
        """Fetch `newspaper` as archived by the Wayback Machine around `date`."""
        self.date = date
        self.availability_api = WaybackMachineAvailabilityAPI(newspaper.url)
        self.wayback_url: str | None = None

        self.session = session if session is not None else aiohttp.ClientSession()
        # close session when object ends
        atexit.register(self._close_session)

        super().__init__(newspaper, cache_dir=cache_dir, model=model)

    def _close_session(self) -> None:
        asyncio.run(self.session.close())

    @retry(
        wait=wait_exponential(multiplier=1, min=10, max=100), stop=stop_after_attempt(8)
    )
    def get_wayback_url(self) -> str:
        """Return the archived URL nearest the requested date, retrying on failure."""
        if self.wayback_url is None:
            archive = self.availability_api.near(
                year=self.date.year,
                month=self.date.month,
                day=self.date.day,
                hour=self.date.hour,
                minute=self.date.minute,
            )
            self._request_time = archive.timestamp()
            time.sleep(1)  # blocking sleep, to not spam the API

            # use the `id_` flag to get the original copy
            # see https://webapps.stackexchange.com/a/155393
            self.wayback_url = archive.archive_url.replace("/http", "id_/http")
        return self.wayback_url

    @retry(
        wait=wait_exponential(multiplier=1, min=10, max=100), stop=stop_after_attempt(8)
    )
    async def _request_coroutine(self) -> tuple[ClientResponse, bytes]:
        url = self.get_wayback_url()
        logger.debug(f"Fetching (async) {url!r}...")
        result = await self.session.get(url, headers=HEADERS)
        content = await result.content.read()
        return result, content

    def _request(self) -> None:
        logger.debug(f"Fetching {self.newspaper.url}...")
        self._response, self._content = asyncio.run(self._request_coroutine())
        if self.cache_dir is not None:
            self.save()
        logger.debug(f"{self.newspaper.url} fetched with code: {self._response.status}")

    async def run_request_coroutine(self, *, ignore_raise: bool = False) -> bytes:
        """Fetch the archived page and return its content as bytes.

        If `ignore_raise` is true, a failed fetch logs a warning and is treated as
        empty content instead of propagating the retry error.
        """
        try:
            self._response, self._content = await self._request_coroutine()
        except RetryError:
            if ignore_raise:
                logger.warning(
                    f"Failed to fetch for {self.newspaper} @ "
                    f"{self.request_time.strftime('%Y/%m/%d')}. "
                    f"Url used was {self.wayback_url=}"
                )
                self._content = b""
            else:
                raise
        if self.cache_dir is not None:
            self.save()
        if self._content is None:
            msg = (
                f"Something failed in the request to {self.newspaper.url}, "
                f"the fetched content is empty!"
            )
            raise RuntimeError(msg)
        return self._content

    async def get_async_content(self) -> bytes:
        """Return the page content, fetching it if nothing is cached yet."""
        if self._content is None:
            await self.run_request_coroutine()
        response = cast("ClientResponse", self._response)
        return await response.content.read()


def clean_url(url: str) -> str:
    """Strip the scheme and slashes from a URL so it is safe as a filename."""
    if "http" not in url:
        msg = f"Not a URL: {url!r}"
        raise ValueError(msg)
    return re.sub("(http|https)://", "", url).strip("/").replace("/", "__")
