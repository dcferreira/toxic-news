# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""MongoDB access helpers: storing scored headlines and aggregating them."""

import datetime
from datetime import timedelta
from typing import TypedDict

import pymongo
import pymongo.errors
from loguru import logger
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.server_api import ServerApi
from tqdm import tqdm

from toxic_news.fetchers import Headline
from toxic_news.models import AllModels, Scores

date_fmt = "%Y/%m/%d"


class HeadlineDayGroupKey(TypedDict):
    """The `_id` of a per-day `$group` stage: one newspaper on one UTC day."""

    newspaper: str
    day: datetime.datetime


# MongoDB's duplicate-key error code. Raised by the unique `headlines-unique-idx`
# and `daily-unique-idx` indices when a document is inserted twice; those errors
# are expected and skipped, any other write error is a real failure.
DUPLICATE_KEY_ERROR = 11000


def get_midnight_datetime(date: datetime.date) -> datetime.datetime:
    """Return midnight UTC of the given calendar date.

    The result is timezone-aware, so it can be compared against the tz-aware
    values stored in MongoDB. `datetime.combine` otherwise takes its `tzinfo`
    from the *time* operand, which `datetime.time.min` leaves naive.

    Args:
        date: the calendar date (in UTC) whose midnight is wanted.

    """
    return datetime.datetime.combine(
        date, datetime.time.min, tzinfo=datetime.timezone.utc
    )


def get_database(url: str, name: str, port: int | None = None) -> Database:
    """Return a handle to the `name` database on the MongoDB server at `url`.

    Args:
        url: connection string, without the port.
        name: name of the database to use.
        port: optional port override for `url`.

    """
    client: MongoClient = MongoClient(
        url,
        server_api=ServerApi("1"),
        port=port,
    )
    db: Database = client[name]
    return db


class AllTimeRow(BaseModel):
    """Average scores of one newspaper across the whole queried range."""

    name: str
    scores: Scores
    count: int


class DailyRow(BaseModel):
    """Average scores of one newspaper on a single day."""

    name: str
    scores: Scores
    count: int
    date: datetime.datetime


def db_insert_headlines(headlines: list[Headline], db: Database) -> int:
    """Insert the given headlines, skipping any that are already stored.

    Returns the number of documents actually inserted. Duplicates are detected
    by the unique index rather than by a pre-read.
    """
    if len(headlines) == 0:
        return 0
    headlines_coll = _get_headlines_collection(db)
    try:
        res = headlines_coll.insert_many(h.dict() for h in headlines)
        return len(res.inserted_ids)
    except pymongo.errors.BulkWriteError as e:
        # ignore errors with duplicate index, simply don't re-insert those
        panic_list = [
            x for x in e.details["writeErrors"] if x["code"] != DUPLICATE_KEY_ERROR
        ]
        if len(panic_list) > 0:
            raise
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
    """Return average headline scores per newspaper per day.

    Covers the half-open range `[start_date, end_date)`, as UTC days.
    """
    headlines_coll = _get_headlines_collection(db)
    pipeline = _get_average_headline_scores_per_day_query(start_date, end_date)
    logger.debug(f"Executing pipeline: {pipeline}")
    results = headlines_coll.aggregate(
        pipeline,
        hint="date",
    )

    def parse_row(_id: HeadlineDayGroupKey, count: int, **kwargs: float) -> DailyRow:
        return DailyRow(
            name=_id["newspaper"], count=count, date=_id["day"], scores=Scores(**kwargs)
        )

    return [parse_row(**row) for row in results]


def db_insert_daily(rows: list[DailyRow], db: Database) -> int:
    """Insert the given daily rows, skipping any that are already stored.

    Returns the number of documents actually inserted. Duplicates are detected
    by the unique index rather than by a pre-read.
    """
    daily_coll = _get_daily_collection(db)
    try:
        res = daily_coll.insert_many([r.dict() for r in rows], ordered=False)
        return len(res.inserted_ids)
    except pymongo.errors.BulkWriteError as e:
        # ignore errors with duplicate index, simply don't re-insert those
        panic_list = [
            x for x in e.details["writeErrors"] if x["code"] != DUPLICATE_KEY_ERROR
        ]
        if len(panic_list) > 0:
            raise
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
    """Return average daily scores per newspaper.

    Averages the per-day rows over the half-open range `[start_date, end_date)`,
    so a newspaper must have at least one stored day in the range to appear.
    """
    daily_coll = _get_daily_collection(db)
    results = daily_coll.aggregate(
        _get_average_daily_scores_query(start_date, end_date),
        hint="date",
    )

    def parse_row(_id: str, count: int, **kwargs: float) -> AllTimeRow:
        return AllTimeRow(name=_id, count=count, scores=Scores(**kwargs))

    return [parse_row(**row) for row in results]


def query_daily_rows(date: datetime.date, db: Database) -> list[DailyRow]:
    """Return the stored daily rows for the UTC day starting at `date`.

    Matches on `[midnight(date), midnight(date) + 1 day)`; the returned `date`
    is that midnight rather than the document's own timestamp.
    """
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


def update_all_headlines_scores(
    db: Database,
    write_coll_name: str | None = None,
    batch_size: int = 128,
    *,
    replace: bool = True,
) -> None:
    """Recompute the scores of every stored headline with the current models.

    Useful when adding/changing models. This is a very expensive operation and
    shouldn't be executed often, which is why it isn't exposed as a CLI command
    in `main.py`.

    Example of how to run this:
    ```python
    import os
    from toxic_news.main import get_database
    from toxic_news.queries import update_all_headlines_scores

    db = get_database(os.environ["MONGODB_URL"], "test_database")
    update_all_headlines_scores(db)
    ```

    It's probably better to write to a different collection (with `replace=False`)
    and then change the name of the new collection to the same name as the old one.
    This can be done in MongoDB's shell:
    ```
    > use admin
    > db.runCommand({ renameCollection: "main.headlines", to: "main.headlines_old" })
    > db.runCommand({ renameCollection: "main.transformed_headlines",
                      to: "main.headlines" })
    ```

    Args:
        db: database holding the `headlines` collection to re-score.
        write_coll_name: collection to write to; defaults to `headlines` itself.
        batch_size: number of documents to score per model call.
        replace: replace scores in place, one document at a time. When False,
            batches are scored in a single model call and inserted into a new
            collection instead.

    """
    headlines_coll = _get_headlines_collection(db)
    write_coll = (
        headlines_coll
        if write_coll_name is None
        else db.get_collection(write_coll_name)
    )
    model = AllModels()

    def insert(buf: list[dict]) -> None:
        if replace:
            # replace one by one, slow but sometimes needed
            for write_doc in buf:
                new_headline = Headline(
                    scores=model.predict([write_doc["text"]])[0], **write_doc
                )
                write_coll.replace_one({"_id": write_doc["_id"]}, new_headline.dict())
        else:
            # replace multiple at once. a bit faster when writing
            scores = model.predict([x["text"] for x in buf])
            headlines = [
                dict(Headline(scores=s, **write_doc).dict(), _id=write_doc["_id"])
                for s, write_doc in zip(scores, buf, strict=False)
            ]
            write_coll.insert_many(headlines)

    buffer: list[dict] = []
    for read_doc in tqdm(
        headlines_coll.find(),
        total=headlines_coll.estimated_document_count(),
    ):
        del read_doc["scores"]
        buffer.append(read_doc)
        if len(buffer) == batch_size:
            insert(buffer)
            buffer = []
    insert(buffer)
