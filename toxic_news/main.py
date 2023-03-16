import asyncio
import os
import re
from asyncio import create_task
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
import typer
from aiohttp import ClientSession
from dotenv import load_dotenv
from jinja2 import Environment, PackageLoader, select_autoescape
from loguru import logger
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.server_api import ServerApi
from tqdm import tqdm

from toxic_news.fetchers import Headline, Newspaper, Scores, WaybackFetcher, newspapers

app = typer.Typer()

load_dotenv()
client: MongoClient = MongoClient(
    os.environ["MONGODB_URL"],
    server_api=ServerApi("1"),
)
db: Database = client[os.environ["DATABASE_NAME"]]


class DailyRow(BaseModel):
    name: str
    scores: Scores
    count: int
    date: str


def average_daily_scores(name: str, date: str) -> Scores:
    cursor = db.headlines.aggregate(
        [
            {
                "$project": {
                    "dateStr": {
                        "$dateToString": {"format": "%G/%m/%d", "date": "$date"}
                    },
                    "newspaper": 1,
                    "text": 1,
                    "scores": 1,
                }
            },
            {
                "$match": {
                    "dateStr": date,
                    "newspaper": name,
                }
            },
            {"$addFields": {"scores": {"$objectToArray": "$scores"}}},
            {"$unwind": {"path": "$scores"}},
            {"$group": {"_id": "$scores.k", "average": {"$avg": "$scores.v"}}},
            {"$addFields": {"score": "$_id", "_id": 0}},
            {
                "$group": {
                    "_id": "score",
                    "scores": {"$push": {"k": "$score", "v": "$average"}},
                }
            },
            {
                "$addFields": {
                    "name": "$_id",
                    "scores": {"$arrayToObject": "$scores"},
                    "_id": 0,
                }
            },
        ]
    )
    results = list(cursor)
    return results[0]["scores"]


def count_daily_headlines(name: str, date: str) -> int:
    cursor = db.headlines.aggregate(
        [
            {
                "$project": {
                    "dateStr": {
                        "$dateToString": {"format": "%G/%m/%d", "date": "$date"}
                    },
                    "newspaper": 1,
                    "text": 1,
                    "scores": 1,
                }
            },
            {
                "$match": {
                    "dateStr": date,
                    "newspaper": name,
                }
            },
            {"$count": "count"},
        ]
    )
    results = list(cursor)
    return results[0]["count"]


def build_daily_table(date: str) -> list[DailyRow]:
    out = []
    for n in newspapers:
        out.append(
            DailyRow(
                name=n.name,
                scores=average_daily_scores(n.name, date),
                count=count_daily_headlines(n.name, date),
                date=date,
            )
        )
    table: Collection = db.daily
    table.insert_many([r.dict() for r in out])
    return out


def build_daily(date: str):
    _verify_date_string(date)
    env = Environment(
        loader=PackageLoader("toxic_news"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("index.html")
    with open("public/index.html", "w+") as fd:
        fd.write(
            template.render(
                rows=build_daily_table(date),
            )
        )


def _verify_date_string(date: str):
    if re.fullmatch(r"[0-9]{4}/[0-1][0-9]/[0-3][0-9]", date) is None:
        raise ValueError(f"Date passed is invalid: {date=}")


@app.command()
def daily_update_db(date: str = typer.Argument(..., help="Date in YYYY/MM/DD format")):
    _verify_date_string(date)
    build_daily_table(date)


async def _fetch_single(
    url: str, headers: dict, newspaper: Newspaper, session: ClientSession
) -> bytes:
    payload = {
        "name": newspaper.name,
        "url": str(newspaper.url),
        "language": newspaper.language,
        "xpath": newspaper.xpath,
    }
    result = await session.post(url, headers=headers, json=payload)
    content = await result.content.read()
    if result.status >= 300:
        logger.error(
            f"Failed request to {newspaper.name!r} with error {result.status}: "
            f"{content!r}"
        )
    else:
        logger.info(f"Fetched {newspaper.name!r}!")

    return content


async def _fetch_daily(url: str, auth_bearer: str, endpoint: str):
    headers = {
        "Authorization": f"Bearer {auth_bearer}",
        "Content-Type": "application/json",
    }
    request_url = url + "/" + endpoint

    async with aiohttp.ClientSession() as session:
        pending = []
        for newspaper in newspapers:
            logger.info(f"Requesting {newspaper.name!r}")
            pending.append(
                create_task(_fetch_single(request_url, headers, newspaper, session))
            )

        await asyncio.wait(pending)


@app.command()
def fetch_daily(
    url: str = typer.Argument(..., envvar="SERVERLESS_URL"),
    auth_bearer: str = typer.Argument(..., envvar="AUTH_BEARER"),
    endpoint: str = "fetch",
):
    asyncio.run(_fetch_daily(url=url, auth_bearer=auth_bearer, endpoint=endpoint))


async def _fetch_wayback(
    newspaper: Newspaper, date_range: list[datetime]
) -> list[list[Headline]]:
    async with aiohttp.ClientSession() as session:
        fetchers = [
            WaybackFetcher(date=date, newspaper=newspaper, session=session)
            for date in date_range
        ]
        tasks = [create_task(f.run_request_coroutine()) for f in fetchers]
        await asyncio.gather(*tasks)

        return [f.classify() for f in tqdm(fetchers)]


def find_newspaper(url: str) -> Newspaper:
    for n in newspapers:
        if n.url == url:
            return n
    raise ValueError(f"No newspaper found with {url=}")


@app.command()
def fetch_wayback(
    url: str,
    date_start: datetime,
    date_end: datetime,
    allowed_difference_headlines: float = typer.Option(
        0.3,
        help="Allow for more or less headlines. "
        "If `expected_nr_headlines` is 100 and this is 0.3, allows for #headlines "
        "between 70 and 130, and errors if there's too many/few headlines.",
    ),
    xpath: Optional[str] = None,
    auto_save: bool = False,
):
    # workaround for to stop logger from interfering with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    numdays = (date_end - date_start).days
    date_list = [date_start + timedelta(days=x) for x in range(numdays)]
    newspaper = find_newspaper(url)
    if xpath is not None:  # might need a different xpath for historic websites
        newspaper.xpath = xpath
    headlines_list = asyncio.run(_fetch_wayback(newspaper, date_list))

    def check_nr(n: int) -> bool:
        return (
            newspaper.expected_headlines * (1 - allowed_difference_headlines)
            <= n
            <= newspaper.expected_headlines * (1 + allowed_difference_headlines)
        )

    headlines_to_insert = []
    bad_dates = []
    for h, d in zip(headlines_list, date_list):
        if check_nr(len(h)):
            headlines_to_insert.append(h)
        else:
            bad_dates.append((d, len(h)))
    for d, n in bad_dates:
        logger.warning(
            f"Bad date not inserted: {d} with {n} headlines "
            f"(expected {newspaper.expected_headlines} ± "
            f"{allowed_difference_headlines * newspaper.expected_headlines:.2f})"
        )

    if auto_save or typer.confirm("Save these results?"):
        if len(headlines_to_insert) > 0:
            table: Collection = db.headlines
            table.insert_many([h.dict() for day in headlines_to_insert for h in day])
    else:
        logger.info("No results saved to database.")


if __name__ == "__main__":
    app()
