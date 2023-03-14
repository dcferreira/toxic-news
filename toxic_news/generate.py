import os
import re

from dotenv import load_dotenv
from jinja2 import Environment, PackageLoader, select_autoescape
from pymongo import MongoClient
from pymongo.command_cursor import CommandCursor
from pymongo.database import Database
from pymongo.server_api import ServerApi

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


if __name__ == "__main__":
    build_daily("2023/03/13")
