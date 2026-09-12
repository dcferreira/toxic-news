# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The git-backed store: raw headlines per day, plus the aggregated daily scores.

The pipeline keeps no database. Each run writes one CSV per day under
``<data_dir>/headlines/YYYY/MM/DD.csv`` (the ``data`` branch), and the published
per-day averages are derived from those headlines into
``<out_dir>/daily/YYYY/MM/DD.csv`` (the ``public`` branch). The rolling-window
averages are in turn derived from the published daily CSVs, so git is the only
store and every published number is reproducible from it.
"""

import csv
import datetime
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, HttpUrl, parse_obj_as

from toxic_news.fetchers import Headline
from toxic_news.models import Scores

date_fmt = "%Y/%m/%d"


def score_fields() -> list[str]:
    """Return the score column names, in the order the published CSVs use them."""
    return list(Scores.schema()["properties"].keys())


def score_vector(scores: Scores) -> dict[str, str]:
    """Return `scores` formatted for a published CSV, keyed by column name."""
    return {k: _export_float(v) for k, v in scores.dict().items()}


def _export_float(value: float) -> str:
    """Format one score the way the published CSVs always have."""
    if value > 1.0:
        # export only 4 precision digits
        return f"{value:.4}"
    # export 5 decimal places for small values
    return f"{value:.5f}"


def _export_raw_float(value: float) -> str:
    """Format one score for the raw store, which is kept at higher precision."""
    return f"{value:.6f}"


# The raw store keeps the whole `Headline` model, so the day files can be read
# back and re-aggregated without a lossy step.
HEADLINE_FIELDS = ("newspaper", "language", "date", "url", "text", *score_fields())


class DailyRow(BaseModel):
    """Average scores of one newspaper on a single UTC day."""

    name: str
    count: int
    date: datetime.date
    scores: Scores


class AllTimeRow(BaseModel):
    """Average scores of one newspaper across a whole queried range."""

    name: str
    count: int
    scores: Scores


def get_date_range(
    start_date: datetime.date, end_date: datetime.date
) -> list[datetime.date]:
    """List each day in [`start_date`, `end_date`), in ascending order."""
    numdays = (end_date - start_date).days
    return [start_date + datetime.timedelta(days=x) for x in range(numdays)]


def headlines_path(data_dir: Path, date: datetime.date) -> Path:
    """Return the raw-store CSV for `date`."""
    return (
        data_dir
        / "headlines"
        / f"{date.year:04d}"
        / f"{date.month:02d}"
        / f"{date.day:02d}.csv"
    )


def daily_csv_path(out_dir: Path, date: datetime.date) -> Path:
    """Return the published daily CSV for `date`."""
    return (
        out_dir
        / "daily"
        / f"{date.year:04d}"
        / f"{date.month:02d}"
        / f"{date.day:02d}.csv"
    )


def write_headlines(
    data_dir: Path, date: datetime.date, headlines: Sequence[Headline]
) -> int:
    """Write `date`'s headlines to their CSV, replacing whatever was there.

    Replacing rather than merging is deliberate: a day's file always holds
    exactly what the last run for that day scraped, so a re-run corrects a day
    instead of freezing a partial scrape into it.
    """
    path = headlines_path(data_dir, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=HEADLINE_FIELDS)
        writer.writeheader()
        for headline in headlines:
            writer.writerow(_headline_row(headline))
    logger.debug(f"Wrote {len(headlines)} headlines to {path}")
    return len(headlines)


def _headline_row(headline: Headline) -> dict[str, str]:
    return {
        "newspaper": headline.newspaper,
        "language": headline.language,
        "date": headline.date.isoformat(),
        "url": str(headline.url) if headline.url is not None else "",
        "text": headline.text,
    } | {
        field: _export_raw_float(getattr(headline.scores, field))
        for field in score_fields()
    }


def read_headlines(data_dir: Path, date: datetime.date) -> list[Headline]:
    """Return the headlines stored for `date`, or nothing if the day is missing."""
    path = headlines_path(data_dir, date)
    if not path.exists():
        return []
    with path.open(newline="") as fd:
        return [_parse_headline(row) for row in csv.DictReader(fd)]


def iter_headlines(
    data_dir: Path, start_date: datetime.date, end_date: datetime.date
) -> Iterator[Headline]:
    """Yield every stored headline in [`start_date`, `end_date`)."""
    for day in get_date_range(start_date, end_date):
        yield from read_headlines(data_dir, day)


def _parse_headline(row: dict[str, str]) -> Headline:
    url = row["url"].strip()
    return Headline(
        newspaper=row["newspaper"],
        language=row["language"],
        date=datetime.datetime.fromisoformat(row["date"]),
        url=parse_obj_as(HttpUrl, url) if url else None,
        text=row["text"],
        scores=Scores(**{field: float(row[field]) for field in score_fields()}),
    )


def average_headline_scores_per_day(
    headlines: Sequence[Headline],
) -> list[DailyRow]:
    """Return the average scores per newspaper per day, ordered by day then name."""
    grouped: dict[tuple[datetime.date, str], list[Headline]] = defaultdict(list)
    for headline in headlines:
        grouped[(headline.date.date(), headline.newspaper)].append(headline)
    return [
        DailyRow(
            name=name,
            count=len(group),
            date=day,
            scores=_mean_scores([one.scores for one in group]),
        )
        for (day, name), group in sorted(grouped.items())
    ]


def average_daily_scores(
    out_dir: Path, start_date: datetime.date, end_date: datetime.date
) -> list[AllTimeRow]:
    """Return the average published daily scores per newspaper in the range.

    This is a mean of daily means, over the daily CSVs already written into
    `out_dir` — the same aggregate the site has always published.
    """
    grouped: dict[str, list[DailyRow]] = defaultdict(list)
    for row in iter_daily_rows(out_dir, start_date, end_date):
        grouped[row.name].append(row)
    return [
        AllTimeRow(
            name=name,
            count=len(rows),
            scores=_mean_scores([one.scores for one in rows]),
        )
        for name, rows in sorted(grouped.items())
    ]


def _mean_scores(scores: Sequence[Scores]) -> Scores:
    logger.debug(f"Averaging {len(scores)} score vectors")
    return Scores(
        **{
            field: sum(getattr(one, field) for one in scores) / len(scores)
            for field in score_fields()
        }
    )


def iter_daily_rows(
    out_dir: Path, start_date: datetime.date, end_date: datetime.date
) -> Iterator[DailyRow]:
    """Yield the published daily rows in [`start_date`, `end_date`)."""
    for path in sorted((out_dir / "daily").glob("*/*/*.csv")):
        with path.open(newline="") as fd:
            for row in csv.DictReader(fd):
                date = datetime.date.fromisoformat(row["date"].replace("/", "-"))
                if start_date <= date < end_date:
                    yield _parse_daily_row(row, date)


def _parse_daily_row(row: dict[str, str], date: datetime.date) -> DailyRow:
    return DailyRow(
        name=row["name"],
        count=int(row["count"]),
        date=date,
        scores=Scores(**{field: float(row[field]) for field in score_fields()}),
    )


def write_daily_csv(
    out_dir: Path, date: datetime.date, rows: Sequence[DailyRow]
) -> Path:
    """Write the published per-day averages for `date`."""
    path = daily_csv_path(out_dir, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["name", *score_fields(), "count", "date"]
    with path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=headers)
        writer.writeheader()
        for row in sorted(rows, key=lambda one: one.name):
            writer.writerow(
                {
                    "name": row.name,
                    "count": row.count,
                    "date": row.date.strftime(date_fmt),
                    **score_vector(row.scores),
                }
            )
    logger.debug(f"Wrote {len(rows)} rows to {path}")
    return path


def write_averages_csv(
    out_dir: Path, filename: str, start_date: datetime.date, end_date: datetime.date
) -> bool:
    """Write one rolling-window averages CSV, reporting whether it had data.

    An empty window leaves the existing file alone: the stale averages are
    closer to right than an empty file would be.
    """
    rows = average_daily_scores(out_dir, start_date, end_date)
    if not rows:
        logger.warning(
            f"No daily rows in [{start_date}, {end_date}); "
            f"leaving averages/{filename} untouched"
        )
        return False
    path = out_dir / "averages" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["name", *score_fields(), "count"]
    with path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"name": row.name, "count": row.count, **score_vector(row.scores)}
            )
    logger.debug(f"Wrote {len(rows)} rows to {path}")
    return True
