# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Typer commands for fetching, scoring and publishing the toxic-news data."""

import asyncio
import csv
import datetime
from asyncio import create_task
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import aiohttp
import typer
from aiohttp import ClientSession
from dotenv import load_dotenv
from jinja2 import Environment, PackageLoader, select_autoescape
from loguru import logger
from pydantic import HttpUrl, parse_obj_as
from pymongo.database import Database
from requests import PreparedRequest
from tqdm import tqdm

from toxic_news.fetchers import Headline, Newspaper, WaybackFetcher
from toxic_news.models import AllModels, Scores
from toxic_news.newspapers import newspapers, newspapers_dict
from toxic_news.queries import (
    date_fmt,
    db_insert_daily,
    db_insert_headlines,
    get_database,
    query_average_daily,
    query_average_headline_scores_per_day,
    query_daily_rows,
)

app = typer.Typer()

load_dotenv()


def update_daily_db_w_date(
    date: datetime.date,
    end_date: datetime.date | None,
    mongodb_url: str,
    database_name: str,
    *,
    auto_save: bool,
) -> None:
    """Insert the average headline scores per day into the `daily` collection.

    Covers the dates in [`date`, `end_date`), defaulting to a single day, and
    asks for confirmation before writing unless `auto_save` is set.
    """
    db = get_database(mongodb_url, database_name)

    end_date_value = (
        end_date if end_date is not None else date + datetime.timedelta(days=1)
    )
    results = query_average_headline_scores_per_day(
        start_date=date, end_date=end_date_value, db=db
    )
    if auto_save or typer.confirm(f"Insert {len(results)} results into table?"):
        db_insert_daily(results, db=db)


@app.command()
def update_daily_db(
    date: datetime.datetime = typer.Argument(..., help="Date in YYYY/MM/DD format"),
    # typer 0.9.0 cannot build a command from a PEP 604 union annotation
    end_date: Optional[datetime.datetime] = typer.Option(  # noqa: UP045
        None,
        help="If a `end_date` is provided, "
        "all the dates in [`date`, `end_date`[ will be inserted.",
    ),
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
    *,
    auto_save: bool = False,
) -> None:
    """Insert per-day average headline scores into the `daily` collection."""
    update_daily_db_w_date(
        date=date.date(),
        end_date=end_date.date() if end_date is not None else None,
        mongodb_url=mongodb_url,
        database_name=database_name,
        auto_save=auto_save,
    )


async def _fetch_single(
    url: str, headers: dict, newspaper: Newspaper, session: ClientSession
) -> bytes:
    params = {
        "url": str(newspaper.url),
    }
    req = PreparedRequest()
    req.prepare_url(url, params)  # put params in URL

    if req.url is None:
        msg = f"Couldn't prepare the url to query. {url=}; {params=}"
        raise RuntimeError(msg)

    result = await session.post(req.url, headers=headers)
    content = await result.content.read()
    if result.status >= HTTPStatus.MULTIPLE_CHOICES:
        logger.error(
            f"Failed request to {newspaper.name!r} with error {result.status}: "
            f"{content!r}"
        )
        msg = f"Failed to fetch {newspaper.name}!"
        raise RuntimeError(msg)

    logger.info(f"Fetched {newspaper.name!r}!")

    return content


async def _fetch_daily(url: str, auth_bearer: str, endpoint: str) -> None:
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

        await asyncio.sleep(20)
        # wait a bit after sending the requests, but don't wait for replies
        logger.info("Terminating process...")


@app.command()
def fetch_today(
    url: str = typer.Argument(..., envvar="SERVERLESS_URL"),
    auth_bearer: str = typer.Argument(..., envvar="AUTH_BEARER"),
    endpoint: str = "fetch",
) -> None:
    """Call the serverless agent once per newspaper to fetch live pages."""
    asyncio.run(_fetch_daily(url=url, auth_bearer=auth_bearer, endpoint=endpoint))


async def _fetch_wayback(
    newspaper: Newspaper, timestamps: list[datetime.datetime], cache_dir: Path | None
) -> list[list[Headline]]:
    model = AllModels()
    async with aiohttp.ClientSession() as session:
        fetchers = [
            WaybackFetcher(
                date=ts,
                newspaper=newspaper,
                session=session,
                cache_dir=cache_dir,
                model=model,
            )
            for ts in timestamps
        ]
        tasks = [
            create_task(f.run_request_coroutine())
            for f, ts in zip(fetchers, timestamps, strict=False)
            if not f.load(ts)  # check if there's cache before making a task
        ]
        await asyncio.gather(*tasks)

        return [f.classify() for f in tqdm(fetchers)]


def get_date_range(
    start_date: datetime.date, end_date: datetime.date
) -> list[datetime.date]:
    """List each day in [`start_date`, `end_date`), in ascending order."""
    numdays = (end_date - start_date).days
    return [start_date + datetime.timedelta(days=x) for x in range(numdays)]


@app.command()
# typer command: every parameter is a CLI option, so none can be consolidated
def fetch_wayback(  # noqa: PLR0913, PLR0917
    url: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
    allowed_difference_headlines: float = typer.Option(
        0.4,
        help="Allow for more or less headlines. "
        "If `expected_nr_headlines` is 100 and this is 0.3, "
        "allows for #headlines between 70 and 130, "
        "and errors if there's too many/few headlines.",
    ),
    *,
    auto_save: bool = False,
    cache_dir: Path = Path(".requests_cache"),
    use_cache: bool = True,
) -> None:
    """Fetch webpages from the Wayback Machine and classify their headlines."""
    # workaround for to stop logger from interfering with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    db = get_database(mongodb_url, database_name)

    date_list = get_date_range(start_date.date(), end_date.date())
    newspaper = newspapers_dict[parse_obj_as(HttpUrl, url)]
    headlines_list = asyncio.run(
        _fetch_wayback(
            newspaper,
            # get one timestamp per day at 12 o'clock
            [
                datetime.datetime(
                    d.year, d.month, d.day, 12, 0, tzinfo=datetime.timezone.utc
                )
                for d in date_list
            ],
            cache_dir if use_cache else None,
        )
    )

    def check_nr(nr: int) -> bool:
        return (
            newspaper.expected_headlines * (1 - allowed_difference_headlines)
            <= nr
            <= newspaper.expected_headlines * (1 + allowed_difference_headlines)
        )

    headlines_with_good_dates = []
    bad_dates = []
    for h, d in zip(headlines_list, date_list, strict=False):
        if check_nr(len(h)):
            headlines_with_good_dates.append(h)
        else:
            bad_dates.append((d, len(h)))
    for d, n in bad_dates:
        logger.warning(
            f"Bad date: {d.strftime(date_fmt)} with {n} headlines "
            f"(expected {newspaper.expected_headlines} ± "
            f"{allowed_difference_headlines * newspaper.expected_headlines:.2f})"
        )

    _save_headlines(
        good_headlines=headlines_with_good_dates,
        all_headlines=headlines_list,
        db=db,
        auto_save=auto_save,
    )


def _save_headlines(
    good_headlines: list[list[Headline]],
    all_headlines: list[list[Headline]],
    db: Database,
    *,
    auto_save: bool,
) -> None:
    """Store the headlines fetched by a `fetch_wayback` run.

    Skips the initial confirmation prompt when `auto_save` is set, though the
    "save the bad dates too?" prompt is still asked, and warns about a day
    whose headlines turn out to be duplicates.
    """
    if not (auto_save or typer.confirm("Save these results?")):
        logger.info("No results saved to database.")
        return

    headlines_to_insert = good_headlines
    if not auto_save or typer.confirm("Save also the bad dates?"):
        headlines_to_insert = all_headlines
    if len(headlines_to_insert) == 0:
        return

    inserted_records = 0
    for one_day_headlines in tqdm(headlines_to_insert):
        if len(one_day_headlines) == 0:
            continue
        n_records = db_insert_headlines(one_day_headlines, db=db)
        inserted_records += n_records
        if n_records == 0:
            logger.warning(
                f"No records inserted for "
                f"{one_day_headlines[0].date.strftime(date_fmt)}; "
                f"probably there are duplicates on that day."
            )

    logger.debug(f"Inserted {inserted_records} records in the collection")


def render_pages(out_dir: Path) -> None:
    """Render `index.html`, `daily.html` and `about.html` into `out_dir`."""
    env = Environment(
        loader=PackageLoader("toxic_news"),
        autoescape=select_autoescape(),
    )
    today = datetime.datetime.now(datetime.timezone.utc).date()
    logger.debug("Writing index...")
    template = env.get_template("index.html")
    with (out_dir / "index.html").open("w+") as fd:
        fd.write(template.render(selected="/index.html", today=today))

    logger.debug("Writing daily...")
    template = env.get_template("daily.html")
    with (out_dir / "daily.html").open("w+") as fd:
        fd.write(template.render(selected="/daily.html", today=today))

    logger.debug("Writing about...")
    template = env.get_template("about.html")
    with (out_dir / "about.html").open("w+") as fd:
        fd.write(template.render(selected="/about.html"))

    logger.debug("Finished rendering!")


@app.command()
def render_html(out_dir: Path = Path("public/")) -> None:
    """Render the static HTML pages into `out_dir`."""
    render_pages(out_dir)


def generate_daily_csv_w_date(
    start_date: datetime.date,
    end_date: datetime.date | None,
    out_dir: Path,
    mongodb_url: str,
    database_name: str,
) -> None:
    """Write a per-day headline-score CSV for each day in the range."""
    db = get_database(mongodb_url, database_name)
    if end_date is None:
        date_range = [start_date]
    else:
        date_range = get_date_range(start_date, end_date)

    headers = [
        "name",
        *list(Scores.schema()["properties"].keys()),
        "count",
        "date",
    ]

    for d in date_range:
        fname = out_dir / str(d.year) / f"{d.month:0>{2}}" / f"{d.day:0>{2}}.csv"
        fname.parent.mkdir(parents=True, exist_ok=True)

        with fname.open("w+") as fd:
            writer = csv.DictWriter(fd, fieldnames=headers)
            writer.writeheader()

            rows = query_daily_rows(d, db)
            logger.debug(f"{len(rows)} rows for {d.strftime(date_fmt)}")
            for r in sorted(rows, key=lambda x: x.name):
                writer.writerow(
                    {
                        "name": r.name,
                        "count": r.count,
                        "date": r.date.strftime(date_fmt),
                        **{k: _export_float(v) for k, v in r.scores.dict().items()},
                    }
                )


@app.command()
def generate_daily_csv(
    start_date: datetime.datetime,
    # typer 0.9.0 cannot build a command from a PEP 604 union annotation
    end_date: Optional[datetime.datetime] = None,  # noqa: UP045
    out_dir: Path = Path("public") / "daily",
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
) -> None:
    """Write the daily headline-score CSVs for a range of days."""
    generate_daily_csv_w_date(
        start_date=start_date.date(),
        end_date=end_date.date() if end_date is not None else None,
        out_dir=out_dir,
        mongodb_url=mongodb_url,
        database_name=database_name,
    )


def _export_float(value: float) -> str:
    if value > 1.0:
        # export only 4 precision digits
        return f"{value:.4}"
    # export 5 decimal places for small values
    return f"{value:.5f}"


def generate_averages_csv(
    start_date: datetime.date,
    end_date: datetime.date,
    filename: Path,
    db: Database,
) -> None:
    """Write the average headline scores per newspaper to `filename`."""
    filename.parent.mkdir(parents=True, exist_ok=True)

    headers = ["name", *list(Scores.schema()["properties"].keys()), "count"]
    with filename.open("w+") as fd:
        writer = csv.DictWriter(fd, fieldnames=headers)
        writer.writeheader()

        rows = query_average_daily(start_date, end_date, db)
        for r in sorted(rows, key=lambda x: x.name):
            writer.writerow(
                {
                    "name": r.name,
                    "count": r.count,
                    **{k: _export_float(v) for k, v in r.scores.dict().items()},
                }
            )


@app.command()
def generate_averages(
    out_dir: Path = Path("public") / "averages",
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
) -> None:
    """Write the rolling-window average CSVs (7, 30, 90, 365 days and all)."""
    db = get_database(mongodb_url, database_name)

    filenames = {
        7: "7.csv",
        30: "30.csv",
        90: "90.csv",
        365: "year.csv",
        99999: "all.csv",
    }
    for ndays, fname in filenames.items():
        _generate_averages_file(out_dir, fname, ndays, db)


def _generate_averages_file(
    out_dir: Path, filename: str, ndays: int, db: Database
) -> None:
    """Write one averages CSV, warning instead of failing on an empty range."""
    try:
        full_fname = out_dir / filename
        logger.info(f"Generating {full_fname}...")
        today = datetime.datetime.now(datetime.timezone.utc).date()
        generate_averages_csv(
            start_date=today - datetime.timedelta(days=ndays),
            # query for tomorrow, to make sure today's data is included
            end_date=today + datetime.timedelta(days=1),
            filename=full_fname,
            db=db,
        )
    except IndexError:
        logger.warning(
            f"Couldn't generate {filename}, probably there's no data in range"
        )


@app.command()
# typer command: every parameter is a CLI option, so none can be consolidated
def update_frontend_today(  # noqa: PLR0913
    *,
    update_db: bool = True,
    daily: bool = True,
    averages: bool = True,
    render: bool = True,
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
    out_dir: Path = Path("public"),
) -> None:
    """Update all the HTML and CSV in the frontend."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    if update_db:
        update_daily_db_w_date(
            today,
            end_date=None,
            auto_save=True,
            mongodb_url=mongodb_url,
            database_name=database_name,
        )
    if daily:
        generate_daily_csv_w_date(
            today,
            end_date=None,
            out_dir=out_dir / "daily",
            mongodb_url=mongodb_url,
            database_name=database_name,
        )
    if averages:
        generate_averages(
            out_dir / "averages", mongodb_url=mongodb_url, database_name=database_name
        )
    if render:
        render_html(out_dir)


if __name__ == "__main__":
    app()
