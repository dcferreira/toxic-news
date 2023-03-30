from datetime import datetime

import pytest

from toxic_news.fetchers import Headline, Scores
from toxic_news.queries import (
    DailyRow,
    _get_average_daily_scores_query,
    _get_average_headline_scores_per_day_query,
    _get_daily_collection,
    _get_headlines_collection,
    db_insert_daily,
    db_insert_headlines,
    get_database,
    query_average_daily,
    query_average_headline_scores_per_day,
)


@pytest.fixture(scope="session")
def mongodb(mongo_proc):
    return get_database(mongo_proc.host, "main", port=mongo_proc.port)


@pytest.fixture(scope="session")
def insert_headlines(mongodb) -> list[Headline]:
    headlines = [
        Headline(
            newspaper="BBC",
            language="en",
            text="this is a negative test headline on 2022-03-03",
            date=datetime(2022, 3, 3),
            scores=Scores(**{s: 0.0 for s in Scores.__fields__}),
        ),
        Headline(
            newspaper="BBC",
            language="en",
            text="this is a positive test headline on 2022-03-03",
            date=datetime(2022, 3, 3),
            scores=Scores(**{s: 1.0 for s in Scores.__fields__}),
        ),
        Headline(
            newspaper="Fox News",
            language="en",
            text="this is a test headline on 2022-03-03 for a different newspaper",
            date=datetime(2022, 3, 3),
            scores=Scores(**{s: 0.5 for s in Scores.__fields__}),
        ),
        Headline(
            newspaper="BBC",
            language="en",
            text="this is a test headline on 2022-03-05",
            date=datetime(2022, 3, 5),
            scores=Scores(**{s: 0.5 for s in Scores.__fields__}),
        ),
    ]

    db_insert_headlines(headlines, mongodb)

    return headlines


def test_insert_headlines(mongodb, insert_headlines):
    assert _get_headlines_collection(mongodb).estimated_document_count() == len(
        insert_headlines
    )


def test_query_average_headline_scores_per_day_all(mongodb, insert_headlines):
    results = query_average_headline_scores_per_day(
        datetime(2022, 3, 3), datetime(2022, 3, 6), mongodb
    )
    assert len(results) == 3
    for row in results:
        assert pytest.approx(row.scores.toxicity) == 0.5


def test_query_average_headline_scores_per_day_first_day(mongodb, insert_headlines):
    results = query_average_headline_scores_per_day(
        datetime(2022, 3, 3), datetime(2022, 3, 5), mongodb
    )
    assert len(results) == 2
    for row in results:
        assert pytest.approx(row.scores.toxicity) == 0.5


def test_query_average_headline_scores_per_day_second_day(mongodb, insert_headlines):
    results = query_average_headline_scores_per_day(
        datetime(2022, 3, 5), datetime(2022, 3, 6), mongodb
    )
    assert len(results) == 1
    for row in results:
        assert pytest.approx(row.scores.toxicity) == 0.5


def test_query_average_headline_scores_per_day_no_day(mongodb, insert_headlines):
    results = query_average_headline_scores_per_day(
        datetime(2022, 3, 10), datetime(2022, 3, 11), mongodb
    )
    assert len(results) == 0


def test_query_average_headline_scores_per_day_explain(mongodb, insert_headlines):
    pipeline = _get_average_headline_scores_per_day_query(
        datetime(2022, 3, 5), datetime(2022, 3, 6)
    )

    explanation = mongodb.command(
        {"explain": {"aggregate": "headlines", "pipeline": pipeline, "cursor": {}}},
        verbosity="executionStats",
    )

    assert (
        explanation["stages"][0]["$cursor"]["executionStats"]["totalDocsExamined"] == 1
    )


@pytest.fixture(scope="session")
def insert_daily(mongodb, insert_headlines) -> list[DailyRow]:
    rows = query_average_headline_scores_per_day(
        datetime(2022, 3, 3), datetime(2022, 3, 10), db=mongodb
    )

    db_insert_daily(rows, db=mongodb)
    return rows


def test_insert_daily(insert_daily, mongodb):
    assert _get_daily_collection(mongodb).estimated_document_count() == len(
        insert_daily
    )


def test_query_average_daily_explain(mongodb, insert_daily):
    pipeline = _get_average_daily_scores_query(
        datetime(2022, 3, 5), datetime(2022, 3, 6)
    )

    explanation = mongodb.command(
        {"explain": {"aggregate": "daily", "pipeline": pipeline, "cursor": {}}},
        verbosity="executionStats",
    )

    assert explanation["executionStats"]["totalDocsExamined"] == 1


def test_query_average_daily_all(mongodb, insert_daily):
    results = query_average_daily(
        datetime(2022, 3, 3), datetime(2022, 3, 10), db=mongodb
    )
    assert len(results) == 2
    for row in results:
        assert pytest.approx(row.scores.toxicity) == 0.5
