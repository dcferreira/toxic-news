import csv
import importlib.resources as pkg_resources
import urllib.parse
from datetime import datetime
from typing import Optional, cast

import lxml.html
import requests
from detoxify import Detoxify
from loguru import logger
from lxml.html import HtmlElement
from pydantic import BaseModel, HttpUrl
from requests import Response

from toxic_news import assets


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
        self._response: Optional[Response] = None
        self._request_time: Optional[datetime] = None

    @property
    def model(self) -> Detoxify:
        if self._model is None:
            self._model = Detoxify("multilingual")
        return self._model

    def _request(self):
        logger.info(f"Fetching {self.newspaper.url}...")
        self._response = requests.get(self.newspaper.url)
        self._request_time = datetime.utcnow()
        logger.info(
            f"{self.newspaper.url} fetched with code: " f"{self._response.status_code}"
        )

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


# load newspapers from the CSV file
newspapers_text = pkg_resources.read_text(assets, "newspapers.csv")
newspapers: list[Fetcher] = [
    Fetcher(
        Newspaper.parse_obj(
            {
                "name": row[0],
                "language": row[1],
                "url": row[2],
                "xpath": row[3],
            }
        )
    )
    for row in csv.reader(newspapers_text.strip().split("\n"))
]
