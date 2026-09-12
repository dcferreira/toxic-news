# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the git-backed store in ``toxic_news.store``."""

import csv
import datetime
from pathlib import Path

import pytest

from toxic_news.fetchers import Headline
from toxic_news.models import Scores
from toxic_news.store import (
    AllTimeRow,
    DailyRow,
    average_daily_scores,
    average_headline_scores_per_day,
    daily_csv_path,
    headlines_path,
    read_headlines,
    write_averages_csv,
    write_daily_csv,
    write_headlines,
)

DAY_ONE = datetime.date(2023, 5, 20)
DAY_TWO = datetime.date(2023, 5, 21)
DAY_THREE = datetime.date(2023, 5, 22)
ONE_DAY = datetime.timedelta(days=1)

# The published column orders, spelled out: the site's JavaScript indexes the
# columns positionally, so these header lines are a contract, not a description.
SCORE_COLUMNS = (
    "toxicity,severe_toxicity,obscene,identity_attack,insult,threat,"
    "sexual_explicit,positive,neutral,negative"
)
RAW_HEADER = f"newspaper,language,date,url,text,{SCORE_COLUMNS}"
DAILY_HEADER = f"name,{SCORE_COLUMNS},count,date"
AVERAGES_HEADER = f"name,{SCORE_COLUMNS},count"


def _scores(**overrides: float) -> Scores:
    """Return a `Scores` with every field at 0.5, overridden by `overrides`."""
    return Scores(**{**dict.fromkeys(Scores.__fields__, 0.5), **overrides})


def _headline(
    newspaper: str,
    scores: Scores,
    *,
    date: datetime.date = DAY_ONE,
    text: str = "a headline",
    url: str | None = "https://example.com/news/story",
) -> Headline:
    """Return a headline scored with `scores`, dated (UTC) on `date`."""
    return Headline(
        newspaper=newspaper,
        language="en",
        text=text,
        date=datetime.datetime(
            date.year, date.month, date.day, tzinfo=datetime.timezone.utc
        ),
        scores=scores,
        url=url,
    )


def _lines(path: Path) -> list[str]:
    """Return the raw lines of `path`."""
    return path.read_text().splitlines()


def test_read_headlines_of_a_missing_day_is_empty(tmp_path: Path) -> None:
    assert read_headlines(tmp_path, DAY_ONE) == []


def test_headlines_round_trip_through_the_raw_store(tmp_path: Path) -> None:
    headlines = [
        _headline("BBC", _scores(toxicity=0.125)),
        _headline("BBC", _scores(toxicity=0.375), text="a second BBC one", url=None),
        _headline("Fox News", _scores()),
    ]

    assert write_headlines(tmp_path, DAY_ONE, headlines) == 3

    path = headlines_path(tmp_path, DAY_ONE)
    assert path == tmp_path / "headlines" / "2023" / "05" / "20.csv"
    assert _lines(path)[0] == RAW_HEADER
    assert read_headlines(tmp_path, DAY_ONE) == headlines


def test_write_headlines_replaces_the_day(tmp_path: Path) -> None:
    first_run = [_headline("BBC", _scores(), text=f"headline {n}") for n in range(3)]
    assert write_headlines(tmp_path, DAY_ONE, first_run) == 3

    replacement = _headline("Fox News", _scores(toxicity=0.875), text="the survivor")
    assert write_headlines(tmp_path, DAY_ONE, [replacement]) == 1

    assert len(_lines(headlines_path(tmp_path, DAY_ONE))) == 2  # header + one row
    assert read_headlines(tmp_path, DAY_ONE) == [replacement]


def test_average_headline_scores_per_day_groups_by_day_then_name() -> None:
    headlines = [
        _headline("BBC", _scores(toxicity=0.125), text="one"),
        _headline("BBC", _scores(toxicity=0.375), text="two"),
        _headline("Fox News", _scores()),
        _headline("BBC", _scores(toxicity=0.75), date=DAY_TWO),
    ]

    rows = average_headline_scores_per_day(headlines)

    assert [(row.date, row.name, row.count) for row in rows] == [
        (DAY_ONE, "BBC", 2),
        (DAY_ONE, "Fox News", 1),
        (DAY_TWO, "BBC", 1),
    ]
    # (0.125 + 0.375) / 2, and the other score fields stay at 0.5
    assert [row.scores for row in rows] == [
        _scores(toxicity=0.25),
        _scores(),
        _scores(toxicity=0.75),
    ]


def test_write_daily_csv_writes_the_published_format(tmp_path: Path) -> None:
    rows = [
        DailyRow(name="Fox News", count=2, date=DAY_ONE, scores=_scores()),
        DailyRow(name="BBC", count=1, date=DAY_ONE, scores=_scores(toxicity=100.0)),
    ]

    path = write_daily_csv(tmp_path, DAY_ONE, rows)

    assert path == daily_csv_path(tmp_path, DAY_ONE)
    assert path == tmp_path / "daily" / "2023" / "05" / "20.csv"
    assert _lines(path)[0] == DAILY_HEADER

    with path.open(newline="") as fd:
        written = list(csv.DictReader(fd))

    # rows sorted by name, the published date format, and five decimals below
    # 1.0 against four significant digits above it
    assert [row["name"] for row in written] == ["BBC", "Fox News"]
    assert written[0]["count"] == "1"
    assert written[0]["date"] == "2023/05/20"
    assert written[0]["toxicity"] == "100.0"
    assert written[1]["toxicity"] == "0.50000"


@pytest.fixture
def public_dir(tmp_path: Path) -> Path:
    """A `public` directory holding two days of published daily CSVs."""
    write_daily_csv(
        tmp_path,
        DAY_ONE,
        [
            DailyRow(name="BBC", count=2, date=DAY_ONE, scores=_scores(toxicity=0.125)),
            DailyRow(name="Fox News", count=1, date=DAY_ONE, scores=_scores()),
        ],
    )
    write_daily_csv(
        tmp_path,
        DAY_TWO,
        [DailyRow(name="BBC", count=1, date=DAY_TWO, scores=_scores(toxicity=0.375))],
    )
    return tmp_path


def test_average_daily_scores_is_the_mean_of_the_daily_means(
    public_dir: Path,
) -> None:
    rows = average_daily_scores(public_dir, DAY_ONE, DAY_THREE)

    assert rows == [
        AllTimeRow(name="BBC", count=2, scores=_scores(toxicity=0.25)),
        AllTimeRow(name="Fox News", count=1, scores=_scores()),
    ]


def test_average_daily_scores_window_is_half_open(public_dir: Path) -> None:
    # DAY_TWO is exactly `end`, so it is excluded...
    assert average_daily_scores(public_dir, DAY_ONE, DAY_TWO) == [
        AllTimeRow(name="BBC", count=1, scores=_scores(toxicity=0.125)),
        AllTimeRow(name="Fox News", count=1, scores=_scores()),
    ]
    # ...while the day at exactly `start` is included
    assert average_daily_scores(public_dir, DAY_TWO, DAY_THREE) == [
        AllTimeRow(name="BBC", count=1, scores=_scores(toxicity=0.375)),
    ]


def test_write_averages_csv_writes_the_window_mean(public_dir: Path) -> None:
    assert write_averages_csv(public_dir, "7.csv", DAY_ONE, DAY_THREE) is True

    path = public_dir / "averages" / "7.csv"
    assert _lines(path)[0] == AVERAGES_HEADER

    with path.open(newline="") as fd:
        written = list(csv.DictReader(fd))

    assert [(row["name"], row["count"]) for row in written] == [
        ("BBC", "2"),
        ("Fox News", "1"),
    ]
    assert written[0]["toxicity"] == "0.25000"


def test_write_averages_csv_leaves_a_file_alone_when_the_window_is_empty(
    public_dir: Path,
) -> None:
    assert write_averages_csv(public_dir, "7.csv", DAY_ONE, DAY_THREE) is True
    path = public_dir / "averages" / "7.csv"
    before = path.read_text()

    # every seeded daily row is dated before DAY_THREE
    empty_window = write_averages_csv(
        public_dir, "7.csv", DAY_THREE, DAY_THREE + ONE_DAY
    )

    assert empty_window is False
    assert path.read_text() == before
