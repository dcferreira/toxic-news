import asyncio
import csv
import datetime
import os
from asyncio import create_task
from pathlib import Path
from typing import Optional

import aiohttp
import typer
from aiohttp import ClientSession
from detoxify import Detoxify
from dotenv import load_dotenv
from jinja2 import Environment, PackageLoader, select_autoescape
from loguru import logger
from pydantic import HttpUrl, parse_obj_as
from pymongo.database import Database
from requests import PreparedRequest
from tqdm import tqdm

from toxic_news.fetchers import Headline, Newspaper, Scores, WaybackFetcher
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
    end_date: Optional[datetime.date],
    mongodb_url: str,
    database_name: str,
    auto_save: bool,
):
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
    end_date: Optional[datetime.datetime] = typer.Option(
        None,
        help="If a `end_date` is provided, "
        "all the dates in [`date`, `end_date`[ will be inserted.",
    ),
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
    auto_save: bool = False,
):
    """
    Queries rows from the `headlines` collection, and stores the averages
    per day per newspaper to the `daily` collection.
    """
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
        raise RuntimeError(f"Couldn't prepare the url to query. {url=}; {params=}")

    result = await session.post(req.url, headers=headers)
    content = await result.content.read()
    if result.status >= 300:
        logger.error(
            f"Failed request to {newspaper.name!r} with error {result.status}: "
            f"{content!r}"
        )
        raise RuntimeError(f"Failed to fetch {newspaper.name}!")
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

        await asyncio.sleep(20)
        # wait a bit after sending the requests, but don't wait for replies
        logger.info("Terminating process...")


@app.command()
def fetch_today(
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
    newspaper: Newspaper, timestamps: list[datetime.datetime], cache_dir: Optional[Path]
) -> list[list[Headline]]:
    model = Detoxify("multilingual")
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
            for f, ts in zip(fetchers, timestamps)
            if not f.load(ts)  # check if there's cache before making a task
        ]
        await asyncio.gather(*tasks)

        return [f.classify() for f in tqdm(fetchers)]


def get_date_range(
    start_date: datetime.date, end_date: datetime.date
) -> list[datetime.date]:
    numdays = (end_date - start_date).days
    date_list = [start_date + datetime.timedelta(days=x) for x in range(numdays)]
    return date_list


@app.command()
def fetch_wayback(
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
    auto_save: bool = False,
    cache_dir: Path = Path(".requests_cache"),
    use_cache: bool = True,
):
    """
    Fetch webpages using the wayback machine, classifies them and (optionally)
    updates the `headlines` collection with the results
    """
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
            [datetime.datetime(d.year, d.month, d.day, 12, 0) for d in date_list],
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
    for h, d in zip(headlines_list, date_list):
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

    if auto_save or typer.confirm("Save these results?"):
        headlines_to_insert = headlines_with_good_dates
        if not auto_save or typer.confirm("Save also the bad dates?"):
            headlines_to_insert = headlines_list
        if len(headlines_to_insert) > 0:
            inserted_records = 0
            for one_day_headlines in tqdm(headlines_to_insert):
                n_records = db_insert_headlines(one_day_headlines, db=db)
                inserted_records += n_records
                if n_records == 0:
                    logger.warning(
                        f"No records inserted for "
                        f"{one_day_headlines[0].date.strftime(date_fmt)}; "
                        f"probably there are duplicates on that day."
                    )

            logger.debug(f"Inserted {inserted_records} records in the collection")
    else:
        logger.info("No results saved to database.")


def render_pages(out_dir: Path):
    env = Environment(
        loader=PackageLoader("toxic_news"),
        autoescape=select_autoescape(),
    )
    logger.debug("Writing index...")
    template = env.get_template("index.html")
    with (out_dir / "index.html").open("w+") as fd:
        fd.write(template.render(selected="/index.html", today=datetime.date.today()))

    logger.debug("Writing daily...")
    template = env.get_template("daily.html")
    with (out_dir / "daily.html").open("w+") as fd:
        fd.write(template.render(selected="/daily.html", today=datetime.date.today()))

    logger.debug("Writing about...")
    template = env.get_template("about.html")
    with (out_dir / "about.html").open("w+") as fd:
        fd.write(template.render(selected="/about.html"))

    logger.debug("Finished rendering!")


@app.command()
def render_html(out_dir: Path = Path("public/")):
    """
    Renders the main pages HTML.
    """
    render_pages(out_dir)


def generate_daily_csv_w_date(
    start_date: datetime.date,
    end_date: Optional[datetime.date],
    out_dir: Path,
    mongodb_url: str,
    database_name: str,
):
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
        os.makedirs(fname.parent, exist_ok=True)

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
                        **{
                            k: f"{v:.5f}"  # export only 5 decimal places
                            for k, v in r.scores.dict().items()
                        },
                    }
                )


@app.command()
def generate_daily_csv(
    start_date: datetime.datetime,
    end_date: Optional[datetime.datetime] = None,
    out_dir: Path = Path("public") / "daily",
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
):
    generate_daily_csv_w_date(
        start_date=start_date.date(),
        end_date=end_date.date() if end_date is not None else None,
        out_dir=out_dir,
        mongodb_url=mongodb_url,
        database_name=database_name,
    )


def generate_averages_csv(
    start_date: datetime.date,
    end_date: datetime.date,
    filename: Path,
    db: Database,
):
    os.makedirs(filename.parent, exist_ok=True)

    headers = ["name", *list(Scores.schema()["properties"].keys()), "count"]
    with filename.open("w+") as fd:
        writer = csv.DictWriter(fd, fieldnames=headers)
        writer.writeheader()

        rows = query_average_daily(start_date, end_date, db)
        for r in rows:
            writer.writerow(
                {
                    "name": r.name,
                    "count": r.count,
                    **{
                        k: f"{v:.5f}"  # export only 5 decimal places
                        for k, v in r.scores.dict().items()
                    },
                }
            )


@app.command()
def generate_averages(
    out_dir: Path = Path("public") / "averages",
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
):
    db = get_database(mongodb_url, database_name)

    filenames = {
        7: "7.csv",
        30: "30.csv",
        90: "90.csv",
        365: "year.csv",
        99999: "all.csv",
    }
    for ndays, fname in filenames.items():
        try:
            full_fname = out_dir / fname
            logger.info(f"Generating {full_fname}...")
            today = datetime.date.today()
            generate_averages_csv(
                start_date=today - datetime.timedelta(days=ndays),
                end_date=today + datetime.timedelta(days=1),
                # query for tomorrow, to make sure today's data is included
                filename=full_fname,
                db=db,
            )
        except IndexError:
            logger.warning(
                f"Couldn't generate {fname}, probably there's no data in range"
            )


@app.command()
def update_frontend_today(
    update_db: bool = True,
    daily: bool = True,
    averages: bool = True,
    render: bool = True,
    mongodb_url: str = typer.Option(..., envvar="MONGODB_URL"),
    database_name: str = typer.Option(..., envvar="DATABASE_NAME"),
    out_dir: Path = Path("public"),
):
    """
    Updates all the HTML and CSV in the frontend.
    """
    today = datetime.date.today()
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
