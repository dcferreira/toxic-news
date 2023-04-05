import datetime
from datetime import timedelta
from typing import Optional, Union

import pymongo
import pymongo.errors
from loguru import logger
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.server_api import ServerApi

from toxic_news.fetchers import Headline, Scores

date_fmt = "%Y/%m/%d"


def get_midnight_datetime(date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(date, datetime.datetime.min.time())


def get_database(url: str, name: str, port: Optional[int] = None) -> Database:
    client: MongoClient = MongoClient(
        url,
        server_api=ServerApi("1"),
        port=port,
    )
    db: Database = client[name]
    return db


class AllTimeRow(BaseModel):
    name: str
    scores: Scores
    count: int


class DailyRow(BaseModel):
    name: str
    scores: Scores
    count: int
    date: datetime.datetime


def db_insert_headlines(headlines: list[Headline], db: Database) -> int:
    headlines_coll = _get_headlines_collection(db)
    try:
        res = headlines_coll.insert_many(h.dict() for h in headlines)
        return len(res.inserted_ids)
    except pymongo.errors.BulkWriteError as e:
        # ignore errors with duplicate index, simply don't re-insert those
        panic_list = [x for x in e.details["writeErrors"] if x["code"] != 11000]
        if len(panic_list) > 0:
            raise e
        return e.details["nInserted"]


def _get_headlines_collection(db: Database) -> Collection:
    headlines: Collection = db.get_collection("headlines")
    # ensure indices are created (if they exist these are noops)
    headlines.create_index("date", name="date")
    headlines.create_index("newspaper", name="newspaper")
    headlines.create_index(
        [
            ("url", pymongo.ASCENDING),
            ("date", pymongo.DESCENDING),
            ("newspaper", pymongo.ASCENDING),
            ("text", pymongo.ASCENDING),
        ],
        name="headlines-unique-idx",
        unique=True,
    )

    return headlines


def _get_daily_collection(db: Database) -> Collection:
    daily: Collection = db.get_collection("daily")
    # ensure indices are created (if they exist these are noops)
    daily.create_index("date", name="date")
    daily.create_index("name", name="name")
    daily.create_index(
        [
            ("name", pymongo.ASCENDING),
            ("date", pymongo.DESCENDING),
        ],
        name="daily-unique-idx",
        unique=True,
    )

    return daily


def _get_average_headline_scores_per_day_query(
    start_date: datetime.date, end_date: datetime.date
) -> list[dict]:
    return [
        {
            "$match": {
                "date": {
                    "$gte": get_midnight_datetime(start_date),
                    "$lt": get_midnight_datetime(end_date),
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "newspaper": "$newspaper",
                    "day": {"$dateTrunc": {"date": "$date", "unit": "day"}},
                },
                "count": {"$sum": 1},
                **{s: {"$avg": f"$scores.{s}"} for s in Scores.__fields__},
            }
        },
    ]


def query_average_headline_scores_per_day(
    start_date: datetime.date, end_date: datetime.date, db: Database
) -> list[DailyRow]:
    headlines_coll = _get_headlines_collection(db)
    pipeline = _get_average_headline_scores_per_day_query(start_date, end_date)
    logger.debug(f"Executing pipeline: {pipeline}")
    results = headlines_coll.aggregate(
        pipeline,
        hint="date",
    )

    def parse_row(
        _id: dict[str, Union[str, datetime.datetime]], count: int, **kwargs
    ) -> DailyRow:
        return DailyRow(
            name=_id["newspaper"], count=count, date=_id["day"], scores=Scores(**kwargs)
        )

    parsed_results = map(lambda x: parse_row(**x), results)
    return list(parsed_results)


def db_insert_daily(rows: list[DailyRow], db: Database) -> int:
    daily_coll = _get_daily_collection(db)
    try:
        res = daily_coll.insert_many([r.dict() for r in rows], ordered=False)
        return len(res.inserted_ids)
    except pymongo.errors.BulkWriteError as e:
        # ignore errors with duplicate index, simply don't re-insert those
        panic_list = [x for x in e.details["writeErrors"] if x["code"] != 11000]
        if len(panic_list) > 0:
            raise e
        return e.details["nInserted"]


def _get_average_daily_scores_query(
    start_date: datetime.date, end_date: datetime.date
) -> list[dict]:
    return [
        {
            "$match": {
                "date": {
                    "$gte": get_midnight_datetime(start_date),
                    "$lt": get_midnight_datetime(end_date),
                }
            }
        },
        {
            "$group": {
                "_id": "$name",
                "count": {"$sum": 1},
                **{s: {"$avg": f"$scores.{s}"} for s in Scores.__fields__},
            }
        },
    ]


def query_average_daily(
    start_date: datetime.date, end_date: datetime.date, db: Database
) -> list[AllTimeRow]:
    daily_coll = _get_daily_collection(db)
    results = daily_coll.aggregate(
        _get_average_daily_scores_query(start_date, end_date),
        hint="date",
    )

    def parse_row(
        _id: dict[str, Union[str, datetime.datetime]], count: int, **kwargs
    ) -> AllTimeRow:
        return AllTimeRow(name=_id, count=count, scores=Scores(**kwargs))

    parsed_results = map(lambda x: parse_row(**x), results)
    return list(parsed_results)


def query_daily_rows(date: datetime.date, db: Database) -> list[DailyRow]:
    daily_coll = _get_daily_collection(db)
    cursor = daily_coll.find(
        {
            "date": {
                "$gte": get_midnight_datetime(date),
                "$lt": get_midnight_datetime(date) + timedelta(days=1),
            }
        },
        hint="date",
    )
    return [
        DailyRow(
            name=res["name"],
            scores=Scores.parse_obj(res["scores"]),
            date=get_midnight_datetime(date),
            count=res["count"],
        )
        for res in cursor
    ]
