import asyncio
import os
from asyncio import create_task
from datetime import datetime, timedelta
from pathlib import Path
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

date_fmt = "%Y/%m/%d"


class DailyRow(BaseModel):
    name: str
    scores: Scores
    count: int
    date: str


class AllTimeRow(BaseModel):
    name: str
    scores: Scores


def query_average_headline_scores(name: str, date: datetime) -> Scores:
    cursor = db.headlines.aggregate(
        [
            {
                "$project": {
                    "dateStr": {
                        "$dateToString": {"format": "%Y/%m/%d", "date": "$date"}
                    },
                    "newspaper": 1,
                    "text": 1,
                    "scores": 1,
                }
            },
            {
                "$match": {
                    "dateStr": date.strftime(date_fmt),
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


def query_average_daily_scores(name: str) -> Scores:
    cursor = db.daily.aggregate(
        [
            {
                "$match": {
                    "name": name,
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


def get_alltime_rows() -> list[AllTimeRow]:
    out = []
    for n in newspapers:
        scores = query_average_daily_scores(n.name)
        out.append(AllTimeRow(name=n.name, scores=scores))
    return out


def render_alltime():
    env = Environment(
        loader=PackageLoader("toxic_news"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("index.html")
    with open("public/index.html", "w+") as fd:
        fd.write(
            template.render(
                rows=get_alltime_rows(),
            )
        )


def query_headlines_count(name: str, date: datetime) -> int:
    cursor = db.headlines.aggregate(
        [
            {
                "$project": {
                    "dateStr": {
                        "$dateToString": {"format": "%Y/%m/%d", "date": "$date"}
                    },
                    "newspaper": 1,
                    "text": 1,
                    "scores": 1,
                }
            },
            {
                "$match": {
                    "dateStr": date.strftime(date_fmt),
                    "newspaper": name,
                }
            },
            {"$count": "count"},
        ]
    )
    results = list(cursor)
    return results[0]["count"]


def insert_daily_table(date: datetime, autosave: bool = True) -> list[DailyRow]:
    out = []
    for n in newspapers:
        out.append(
            DailyRow(
                name=n.name,
                scores=query_average_headline_scores(n.name, date),
                count=query_headlines_count(n.name, date),
                date=date.strftime(date_fmt),
            )
        )

    logger.debug(f"First 5 rows to be inserted:\n\n{out[:5]}")
    if autosave or typer.confirm("Insert results into table?"):
        if len(out) > 0:
            table: Collection = db.daily
            table.insert_many([r.dict() for r in out])
    else:
        logger.info("Results not inserted")
    return out


def query_daily_rows(date: datetime) -> list[DailyRow]:
    table: Collection = db.daily
    date_str = date.strftime(date_fmt)
    cursor = table.find({"date": date_str})
    return [
        DailyRow(
            name=res["name"],
            scores=Scores.parse_obj(res["scores"]),
            date=date_str,
            count=res["count"],
        )
        for res in cursor
    ]


@app.command()
def render_daily(
    date: datetime,
    end_date: Optional[datetime] = typer.Option(
        None,
        help="If a `end_date` is provided, "
        "all the dates in [`date`, `end_date`[ will be inserted.",
    ),
):
    """
    Renders a page for one (or multiple) dates
    """
    if end_date is not None:
        date_range = get_date_range(date, end_date)
    else:
        date_range = [date]

    for d in date_range:
        env = Environment(
            loader=PackageLoader("toxic_news"),
            autoescape=select_autoescape(),
        )
        dir_path = Path("public") / str(d.year) / str(d.month)
        if not dir_path.exists():
            os.makedirs(dir_path)
        path = dir_path / f"{d.day}.html"

        template = env.get_template("daily.html")
        with open(path, "w+") as fd:
            fd.write(
                template.render(
                    rows=query_daily_rows(d),
                )
            )


@app.command()
def update_daily_db(
    date: datetime = typer.Argument(..., help="Date in YYYY/MM/DD format"),
    end_date: Optional[datetime] = typer.Option(
        None,
        help="If a `end_date` is provided, "
        "all the dates in [`date`, `end_date`[ will be inserted.",
    ),
    auto_save: bool = False,
):
    """
    Queries rows from the `headlines` collection, and stores the averages
    per day per newspaper to the `daily` collection.
    """
    if end_date is not None:
        date_range = get_date_range(date, end_date)
    else:
        date_range = [date]
    for d in date_range:
        insert_daily_table(d, auto_save)


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
    """
    Calls the serverless agent with one call per newspaper, to fetch live
    pages and populate the `headlines` collection.
    """
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


def get_date_range(start_date: datetime, end_date: datetime) -> list[datetime]:
    numdays = (end_date - start_date).days
    date_list = [start_date + timedelta(days=x) for x in range(numdays)]
    return date_list


@app.command()
def fetch_wayback(
    url: str,
    start_date: datetime,
    end_date: datetime,
    allowed_difference_headlines: float = typer.Option(
        0.4,
        help="Allow for more or less headlines. "
        "If `expected_nr_headlines` is 100 and this is 0.3, allows for #headlines "
        "between 70 and 130, and errors if there's too many/few headlines.",
    ),
    xpath: Optional[str] = None,
    auto_save: bool = False,
):
    """
    Fetch webpages using the wayback machine, classifies them and (optionally)
    updates the `headlines` collection with the results
    """
    # workaround for to stop logger from interfering with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    date_list = get_date_range(start_date, end_date)
    newspaper = find_newspaper(url)
    if xpath is not None:  # might need a different xpath for historic websites
        newspaper.xpath = xpath
    headlines_list = asyncio.run(_fetch_wayback(newspaper, date_list))

    def check_nr(nr: int) -> bool:
        return (
            newspaper.expected_headlines * (1 - allowed_difference_headlines)
            <= nr
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
            f"Bad date: {d} with {n} headlines "
            f"(expected {newspaper.expected_headlines} ± "
            f"{allowed_difference_headlines * newspaper.expected_headlines:.2f})"
        )

    if auto_save or typer.confirm("Save these results?"):
        if len(headlines_to_insert) > 0:
            table: Collection = db.headlines
            table.insert_many([h.dict() for day in headlines_to_insert for h in day])
    else:
        logger.info("No results saved to database.")


@app.command()
def render():
    """
    Renders the main page HTML.
    """
    render_alltime()


if __name__ == "__main__":
    app()
