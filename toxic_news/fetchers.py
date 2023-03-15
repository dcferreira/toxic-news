import asyncio
import atexit
import csv
import importlib.resources as pkg_resources
import urllib.parse
from datetime import datetime
from typing import Optional, cast

import aiohttp
import lxml.html
import nest_asyncio
from aiohttp import ClientResponse, ClientSession
from detoxify import Detoxify
from loguru import logger
from lxml.html import HtmlElement
from pydantic import BaseModel, HttpUrl
from requests import Response
from waybackpy import WaybackMachineAvailabilityAPI

from toxic_news import assets

# allow nesting of loops
# removing this fails the test `test_wayback_sync`
# with `RuntimeError: Timeout context manager should be used inside a task`
nest_asyncio.apply()


class Newspaper(BaseModel):
    name: str
    language: str
    url: HttpUrl
    xpath: str


class Headline(BaseModel):
    newspaper: str
    language: str
    text: str
    date: datetime
    toxicity: float
    url: HttpUrl


class ClassifierResult(BaseModel):
    toxicity: list[float]
    severe_toxicity: list[float]
    obscene: list[float]
    identity_attack: list[float]
    insult: list[float]
    threat: list[float]
    sexual_explicit: list[float]


class Fetcher:
    def __init__(self, newspaper: Newspaper):
        self.newspaper = newspaper
        self.xpath = newspaper.xpath
        self._model: Optional[Detoxify] = None
        self._response: Optional[ClientResponse] = None
        self._request_time: Optional[datetime] = None

    @property
    def model(self) -> Detoxify:
        if self._model is None:
            self._model = Detoxify("multilingual")
        return self._model

    async def _request_coroutine(self) -> ClientResponse:
        async with aiohttp.ClientSession() as session:
            result = await session.get(self.newspaper.url)
            return result

    def _request(self):
        logger.info(f"Fetching {self.newspaper.url}...")
        self._response = asyncio.run(self._request_coroutine())
        self._request_time = datetime.utcnow()
        logger.info(f"{self.newspaper.url} fetched with code: {self._response.status}")

    @property
    def content(self) -> str:
        if self._response is None:
            self._request()
        return cast(Response, self._response).text

    @property
    def request_time(self) -> datetime:
        if self._request_time is None:
            self._request()
        return cast(datetime, self._request_time)

    def parse(self, content: str) -> list[tuple[str, str]]:
        tree: HtmlElement = lxml.html.fromstring(content)
        headlines = [
            (
                "".join(x.itertext()).strip(" \t\n\r|"),  # title
                urllib.parse.urljoin(self.newspaper.url, x.get("href")),  # absolute url
            )
            for x in tree.xpath(self.xpath)
        ]
        # often the same headline appears multiple times in the page
        deduped_headlines = set(headlines)
        # sort by order in which they appear in the page
        return sorted(deduped_headlines, key=lambda x: headlines.index(x))

    def fetch(self) -> list[tuple[str, str]]:
        return self.parse(self.content)

    def _predict(self, texts: list[str]) -> dict[str, list[float]]:
        return self.model.predict(texts)

    def predict(self, texts: list[str]) -> ClassifierResult:
        return ClassifierResult.parse_obj(self._predict(texts))

    def classify(self) -> list[Headline]:
        content = self.fetch()
        results = self.predict([x[0] for x in content])
        return [
            Headline(
                newspaper=self.newspaper.name,
                language=self.newspaper.language,
                text=t,
                date=self.request_time,
                toxicity=s,
                url=cast(HttpUrl, u),
            )
            for s, (t, u) in zip(results.toxicity, content)
        ]


class WaybackFetcher(Fetcher):
    def __init__(
        self,
        date: datetime,
        newspaper: Newspaper,
        session: Optional[ClientSession] = None,
    ):
        self.date = date
        self.availability_api = WaybackMachineAvailabilityAPI(newspaper.url)
        self._wayback_url: Optional[str] = None

        self.session = session if session is not None else aiohttp.ClientSession()
        # close session when object ends
        atexit.register(self._close_session)
        super().__init__(newspaper)

        self._response: Optional[ClientResponse] = None

    def _close_session(self):
        asyncio.run(self.session.close())

    def get_wayback_url(self) -> str:
        if self._wayback_url is None:
            archive = self.availability_api.near(
                year=self.date.year, month=self.date.month, day=self.date.day, hour=12
            )
            self._request_time = archive.timestamp()

            # use the `id_` flag to get the original copy
            # see https://webapps.stackexchange.com/a/155393
            self._wayback_url = archive.archive_url.replace("/http", "id_/http")
        return cast(str, self._wayback_url)

    async def _request_coroutine(self) -> ClientResponse:
        url = self.get_wayback_url()
        logger.debug(f"Fetching {url}...")
        result = await self.session.get(url)
        return result

    def _request(self):
        logger.info(f"Fetching {self.newspaper.url}...")
        self._response = asyncio.run(self._request_coroutine())
        logger.info(f"{self.newspaper.url} fetched with code: {self._response.status}")

    async def run_request_coroutine(self) -> ClientResponse:
        self._response = await self._request_coroutine()
        return self._response

    @property
    def content(self) -> str:
        if self._response is None:
            self._request()
        return asyncio.run(self.get_async_content()).decode()

    async def get_async_content(self) -> bytes:
        if self._response is None:
            await self.run_request_coroutine()
        response = cast(ClientResponse, self._response)
        result = await response.content.read()
        return result


# load newspapers from the CSV file
newspapers_text = pkg_resources.read_text(assets, "newspapers.csv")
newspapers: list[Newspaper] = [
    Newspaper.parse_obj(
        {
            "name": row[0],
            "language": row[1],
            "url": row[2],
            "xpath": row[3],
        }
    )
    for row in csv.reader(newspapers_text.strip().split("\n"))
]
