import csv
import importlib.resources as pkg_resources
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
    text: str
    date: datetime
    toxicity: float


class Fetcher:
    def __init__(self, newspaper: Newspaper):
        self.newspaper = newspaper
        self.xpath = newspaper.xpath
        self._response: Optional[Response] = None
        self._request_time: Optional[datetime] = None

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

    def parse(self, content: str) -> list[str]:
        tree: HtmlElement = lxml.html.fromstring(content)
        titles_list = [
            "".join(x.itertext()).strip(" \t\n\r|") for x in tree.xpath(self.xpath)
        ]
        # sort list by appearance in the page
        return sorted(set(titles_list), key=lambda x: titles_list.index(x))

    def fetch(self) -> list[str]:
        return self.parse(self.content)

    def classify(self) -> list[Headline]:
        texts = self.fetch()
        results = Detoxify("multilingual").predict(texts)
        return [
            Headline(
                newspaper=self.newspaper.name,
                text=t,
                date=self.request_time,
                toxicity=s,
            )
            for s, t in zip(results["toxicity"], texts)
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
