import csv
import importlib.resources as pkg_resources
from typing import Optional

import lxml.html
import requests
from loguru import logger
from pydantic import BaseModel, HttpUrl
from requests import Response

from toxic_news import assets


class Fetcher:
    def __init__(self, url: str, xpath: str):
        self.url = url
        self.xpath = xpath
        self._response: Optional[Response] = None

    @property
    def content(self) -> str:
        if self._response is None:
            logger.info(f"Fetching {self.url}...")
            self._response = requests.get(self.url)
            logger.info(f"{self.url} fetched with code: {self._response.status_code}")
        return self._response.text

    def parse(self, content: str) -> list[str]:
        tree = lxml.html.fromstring(content)
        titles_list = [
            "".join(x.itertext()).strip(" \t\n\r|") for x in tree.xpath(self.xpath)
        ]
        # sort list by appearance in the page
        return sorted(set(titles_list), key=lambda x: titles_list.index(x))

    def fetch(self) -> list[str]:
        return self.parse(self.content)


class Newspaper(BaseModel):
    url: HttpUrl
    xpath: str

    def fetch(self):
        return Fetcher(self.url, self.xpath).fetch()


# load newspapers' urls and xpaths from the CSV file
newspapers_text = pkg_resources.read_text(assets, "newspapers.csv")
newspapers = [
    Newspaper(url=row[0], xpath=row[1])
    for row in csv.reader(newspapers_text.split("\n"))
]
