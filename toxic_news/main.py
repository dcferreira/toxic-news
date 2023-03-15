import asyncio
import os
import re
from asyncio import create_task

import aiohttp
import typer
from aiohttp import ClientSession
from dotenv import load_dotenv
from jinja2 import Environment, PackageLoader, select_autoescape
from loguru import logger
from pymongo import MongoClient
from pymongo.command_cursor import CommandCursor
from pymongo.database import Database
from pymongo.server_api import ServerApi

from toxic_news.fetchers import Newspaper, newspapers

app = typer.Typer()

load_dotenv()
client: MongoClient = MongoClient(
    os.environ["MONGODB_URL"],
    server_api=ServerApi("1"),
)
db: Database = client[os.environ["DATABASE_NAME"]]


def build_daily_table(date: str) -> CommandCursor:
    return db.headlines.aggregate(
        [
            {
                "$project": {
                    "dateStr": {
                        "$dateToString": {"format": "%G/%m/%d", "date": "$date"}
                    },
                    "newspaper": 1,
                    "text": 1,
                    "toxicity": 1,
                }
            },
            {"$match": {"dateStr": date}},
            {
                "$group": {
                    "_id": "$newspaper",
                    "toxicityAvg": {"$avg": "$toxicity"},
                    "count": {"$sum": 1},
                }
            },
        ]
    )


def build_daily(date: str):
    if re.fullmatch(r"[0-9]{4}/[0-1][0-9]/[0-3][0-9]", date) is None:
        raise ValueError(f"Date passed is invalid: {date=}")
    env = Environment(
        loader=PackageLoader("toxic_news"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("index.html")
    with open("public/index.html", "w+") as fd:
        fd.write(
            template.render(
                rows=sorted(build_daily_table(date), key=lambda k: k["toxicityAvg"]),
            )
        )


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


if __name__ == "__main__":
    # build_daily("2023/03/13")
    app()
