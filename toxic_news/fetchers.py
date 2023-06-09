import asyncio
import atexit
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

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
    newspaper: str
    language: str
    text: str
    date: datetime
    scores: Scores
    url: Optional[HttpUrl]


def validate_url(url: str) -> bool:
    class UrlModel(BaseModel):
        url: HttpUrl

    try:
        UrlModel(url=url)  # type: ignore
        return True
    except ValidationError:
        return False


class Fetcher:
    def __init__(
        self,
        newspaper: Newspaper,
        cache_dir: Optional[Path] = None,
        model: Optional[AllModels] = None,
    ):
        self.newspaper = newspaper
        self.cache_dir = cache_dir

        self._model: Optional[AllModels] = model
        self._response: Optional[ClientResponse] = None
        self._content: Optional[bytes] = None
        self._request_time: Optional[datetime] = None

    def _get_cache_filename(self, date: datetime) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("Cache path was not set!")
        return (
            self.cache_dir
            / clean_url(self.newspaper.url)
            / f"{date.strftime('%Y%m%d')}.pickle"
        )

    def save(self):
        if self.cache_dir is None:
            raise ValueError(
                f"Trying to save a website without setting a cache path! {self=}"
            )
        if self._request_time is None:
            raise RuntimeError(
                f"Trying to save a website that wasn't yet fetched! {self=}"
            )
        save_path = self._get_cache_filename(self._request_time)
        os.makedirs(save_path.parent, exist_ok=True)

        logger.debug(f"Saving {self!r} to {save_path!r}")
        with open(save_path, "wb+") as fd:
            pickle.dump(
                {"content": self._content, "request_time": self._request_time}, fd
            )

    def load(self, date: datetime) -> bool:
        if self.cache_dir is None:
            return False
        load_path = self._get_cache_filename(date)
        if load_path.exists():
            logger.debug(f"Loading {self.newspaper.url} from cache: {load_path}")
            with open(load_path, "rb") as fd:
                d = pickle.load(fd)
                self._content = d["content"]
                self._request_time = d["request_time"]
            return True
        return False

    @property
    def model(self) -> AllModels:
        if self._model is None:
            self._model = AllModels()
        return self._model

    async def _request_coroutine(self) -> tuple[ClientResponse, bytes]:
        async with aiohttp.ClientSession() as session:
            if self.newspaper.url == "https://newsmax.com":
                # newsmax requires http/2 when using our header,
                # but aiohttp doesn't support http/2
                result = await session.get(self.newspaper.url)
            else:
                result = await session.get(self.newspaper.url, headers=HEADERS)
            content = await result.content.read()
            return result, content

    def _request(self):
        logger.debug(f"Fetching {self.newspaper.url!r}...")
        self._response, self._content = asyncio.run(self._request_coroutine())
        self._request_time = datetime.utcnow()
        if self.cache_dir is not None:
            self.save()
        logger.debug(f"{self.newspaper.url} fetched with code: {self._response.status}")

    @property
    def content(self) -> str:
        if self._content is None:
            self._request()
        return cast(bytes, self._content).decode()

    @property
    def request_time(self) -> datetime:
        if self._request_time is None:
            self._request()
        return cast(datetime, self._request_time)

    def parse(self, content: str) -> list[tuple[str, str]]:
        return self.newspaper.get_headlines(
            content=content, request_date=self.request_time
        )

    def fetch(self) -> list[tuple[str, str]]:
        return self.parse(self.content)

    def classify(self) -> list[Headline]:
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
                url=cast(HttpUrl, u) if validate_url(u) else None,
            )
            for s, (t, u) in zip(scores_list, content)
        ]


class WaybackFetcher(Fetcher):
    def __init__(
        self,
        date: datetime,
        newspaper: Newspaper,
        session: Optional[ClientSession] = None,
        **kwargs,
    ):
        self.date = date
        self.availability_api = WaybackMachineAvailabilityAPI(newspaper.url)
        self.wayback_url: Optional[str] = None

        self.session = session if session is not None else aiohttp.ClientSession()
        # close session when object ends
        atexit.register(self._close_session)

        super().__init__(newspaper, **kwargs)

    def _close_session(self):
        asyncio.run(self.session.close())

    @retry(
        wait=wait_exponential(multiplier=1, min=10, max=100), stop=stop_after_attempt(8)
    )
    def get_wayback_url(self) -> str:
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
        return cast(str, self.wayback_url)

    @retry(
        wait=wait_exponential(multiplier=1, min=10, max=100), stop=stop_after_attempt(8)
    )
    async def _request_coroutine(self) -> tuple[ClientResponse, bytes]:
        url = self.get_wayback_url()
        logger.debug(f"Fetching (async) {url!r}...")
        result = await self.session.get(url, headers=HEADERS)
        content = await result.content.read()
        return result, content

    def _request(self):
        logger.debug(f"Fetching {self.newspaper.url}...")
        self._response, self._content = asyncio.run(self._request_coroutine())
        if self.cache_dir is not None:
            self.save()
        logger.debug(f"{self.newspaper.url} fetched with code: {self._response.status}")

    async def run_request_coroutine(self, ignore_raise: bool = False) -> bytes:
        try:
            self._response, self._content = await self._request_coroutine()
        except RetryError as e:
            if ignore_raise:
                logger.warning(
                    f"Failed to fetch for {self.newspaper} @ "
                    f"{self.request_time.strftime('%Y/%m/%d')}. "
                    f"Url used was {self.wayback_url=}"
                )
                self._content = b""
            else:
                raise e
        if self.cache_dir is not None:
            self.save()
        if self._content is None:
            raise RuntimeError(
                f"Something failed in the request to {self.newspaper.url}, "
                f"the fetched content is empty!"
            )
        return self._content

    async def get_async_content(self) -> bytes:
        if self._content is None:
            await self.run_request_coroutine()
        response = cast(ClientResponse, self._response)
        result = await response.content.read()
        return result


def clean_url(url: str) -> str:
    """Removes the slashes from a URL, to make sure it's safe to use as a filename."""
    assert "http" in url
    out = re.sub("(http|https)://", "", url).strip("/").replace("/", "__")
    return out
