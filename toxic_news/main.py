# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Typer commands for fetching, scoring and publishing the toxic-news data."""

import asyncio
import datetime
from asyncio import create_task
from collections import Counter
from pathlib import Path
from typing import Optional

import aiohttp
import typer
from aiohttp import ClientSession
from jinja2 import Environment, PackageLoader, select_autoescape
from loguru import logger
from pydantic import AnyHttpUrl, HttpUrl, parse_obj_as
from tqdm import tqdm

from toxic_news.fetchers import Fetcher, Headline, Newspaper, WaybackFetcher, clean_url
from toxic_news.models import AllModels, download_models
from toxic_news.newspapers import newspapers, newspapers_dict
from toxic_news.store import (
    average_headline_scores_per_day,
    daily_csv_path,
    date_fmt,
    get_date_range,
    write_averages_csv,
    write_daily_csv,
    write_headlines,
)

app = typer.Typer()

# An outlet whose front page yields fewer than this fraction of its expected
# headline count is logged as suspect. The run still publishes: broken scrapers
# are expected, and get repaired separately from publishing.
SUSPECT_HEADLINE_RATIO = 0.5

AVERAGE_WINDOWS = {
    7: "7.csv",
    30: "30.csv",
    90: "90.csv",
    365: "year.csv",
    99999: "all.csv",
}


def newspapers_from(base_url: str | None) -> list[Newspaper]:
    """Return the tracked outlets, fetched from `base_url` when one is given.

    The mock site serves one recorded front page per outlet under the outlet's
    own host path, so `https://bbc.com` is fetched from `<base_url>/bbc.com`.
    """
    if base_url is None:
        return newspapers
    base = base_url.rstrip("/")
    return [
        newspaper.copy(
            update={
                "url": parse_obj_as(
                    AnyHttpUrl, f"{base}/{clean_url(str(newspaper.url))}"
                )
            }
        )
        for newspaper in newspapers
    ]


async def _fetch_all(fetchers: list[Fetcher], session: ClientSession) -> None:
    results = await asyncio.gather(
        *(fetcher.fetch_with(session) for fetcher in fetchers), return_exceptions=True
    )
    for fetcher, result in zip(fetchers, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(f"Could not fetch {fetcher.newspaper.name!r}: {result!r}")


async def _scrape_newspapers(base_url: str | None) -> list[Headline]:
    model = AllModels()
    fetchers = [
        Fetcher(newspaper=newspaper, model=model)
        for newspaper in newspapers_from(base_url)
    ]
    async with aiohttp.ClientSession() as session:
        await _fetch_all(fetchers, session)
    headlines: list[Headline] = []
    for fetcher in fetchers:
        if fetcher.fetched:
            headlines.extend(fetcher.classify())
    return headlines


def scrape_newspapers(base_url: str | None = None) -> list[Headline]:
    """Scrape and score every tracked outlet's front page, concurrently."""
    return asyncio.run(_scrape_newspapers(base_url))


def report_headline_counts(headlines: list[Headline]) -> None:
    """Log how many headlines each outlet yielded, warning about the short ones."""
    counts = Counter(headline.newspaper for headline in headlines)
    for newspaper in newspapers:
        found = counts.get(newspaper.name, 0)
        message = f"{newspaper.name}: {found}/{newspaper.expected_headlines} headlines"
        if found < newspaper.expected_headlines * SUSPECT_HEADLINE_RATIO:
            logger.warning(message)
        else:
            logger.info(message)
    logger.info(f"Scraped {len(headlines)} headlines from {len(counts)} outlets")


def rebuild_averages(out_dir: Path) -> None:
    """Rebuild every rolling-window averages CSV from the published daily CSVs."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    for ndays, filename in AVERAGE_WINDOWS.items():
        write_averages_csv(
            out_dir,
            filename,
            # the window ends tomorrow, so that a full `ndays` of history is
            # averaged and today's own row is included
            today - datetime.timedelta(days=ndays),
            today + datetime.timedelta(days=1),
        )


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


def _noon(day: datetime.date) -> datetime.datetime:
    """Return midday UTC on `day`, the timestamp the Wayback Machine is asked for."""
    return datetime.datetime(
        day.year, day.month, day.day, 12, tzinfo=datetime.timezone.utc
    )


def _headline_count_is_ok(
    found: int, newspaper: Newspaper, allowed_difference_headlines: float
) -> bool:
    """Return whether `found` headlines are within the tolerated band."""
    expected = newspaper.expected_headlines
    return (
        expected * (1 - allowed_difference_headlines)
        <= found
        <= expected * (1 + allowed_difference_headlines)
    )


def _format_days(days: list[datetime.date]) -> str:
    return ", ".join(day.strftime(date_fmt) for day in days)


@app.command()
def update(
    data_dir: Path = Path("data"),
    out_dir: Path = Path("public"),
    # typer 0.9.0 cannot build a command from a PEP 604 union annotation
    base_url: Optional[str] = typer.Option(  # noqa: UP045
        None,
        help="Fetch every outlet from this origin instead of the live sites, "
        "e.g. http://mock:8000 to run against the local mock site.",
    ),
) -> None:
    """Scrape and score today's front pages, then rebuild the published site."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    headlines = scrape_newspapers(base_url=base_url)
    report_headline_counts(headlines)
    if not headlines:
        logger.warning("Nothing was scraped; leaving the published data untouched")
        return

    write_headlines(data_dir, today, headlines)
    rows = [
        row for row in average_headline_scores_per_day(headlines) if row.date == today
    ]
    write_daily_csv(out_dir, today, rows)
    rebuild_averages(out_dir)
    render_pages(out_dir)


def _save_headlines(
    date_list: list[datetime.date],
    headlines_list: list[list[Headline]],
    good_dates: list[datetime.date],
    data_dir: Path,
) -> None:
    """Store a `fetch_wayback` run's headlines, asking before writing."""
    if not typer.confirm(f"Write the {len(good_dates)} day(s) that looked right?"):
        logger.info("No results saved.")
        return

    selected = list(good_dates)
    if typer.confirm("Include the days whose headline count was off?"):
        selected = date_list

    by_day = dict(zip(date_list, headlines_list, strict=True))
    written = sum(
        write_headlines(data_dir, day, by_day[day]) for day in selected if by_day[day]
    )
    logger.debug(f"Wrote {written} headlines into {data_dir}")


@app.command()
# typer command: every parameter is a CLI option, so none can be consolidated
def fetch_wayback(  # noqa: PLR0913
    url: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    data_dir: Path = Path("data"),
    allowed_difference_headlines: float = typer.Option(
        0.4,
        help="Allow for more or less headlines. "
        "If `expected_nr_headlines` is 100 and this is 0.3, "
        "allows for #headlines between 70 and 130, "
        "and errors if there's too many/few headlines.",
    ),
    *,
    cache_dir: Path = Path(".requests_cache"),
    use_cache: bool = True,
) -> None:
    """Fetch webpages from the Wayback Machine and store their headlines."""
    # workaround for to stop logger from interfering with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    date_list = get_date_range(start_date.date(), end_date.date())
    newspaper = newspapers_dict[parse_obj_as(HttpUrl, url)]
    headlines_list = asyncio.run(
        _fetch_wayback(
            newspaper,
            [_noon(day) for day in date_list],
            cache_dir if use_cache else None,
        )
    )

    def looks_right(day_headlines: list[Headline]) -> bool:
        return _headline_count_is_ok(
            len(day_headlines), newspaper, allowed_difference_headlines
        )

    good = {
        day
        for day, headlines in zip(date_list, headlines_list, strict=True)
        if looks_right(headlines)
    }
    for day, headlines in zip(date_list, headlines_list, strict=True):
        if day not in good:
            logger.warning(
                f"Bad date: {day.strftime(date_fmt)} with {len(headlines)} headlines "
                f"(expected {newspaper.expected_headlines} ± "
                f"{allowed_difference_headlines * newspaper.expected_headlines:.2f})"
            )

    _save_headlines(date_list, headlines_list, sorted(good), data_dir)


@app.command()
# typer command: every parameter is a CLI option, so none can be consolidated
def heal(  # noqa: PLR0913
    data_dir: Path = Path("data"),
    out_dir: Path = Path("public"),
    days: int = typer.Option(30, help="How many days back to look for missing days."),
    allowed_difference_headlines: float = typer.Option(
        0.4,
        help="Allow for more or less headlines. "
        "If `expected_nr_headlines` is 100 and this is 0.3, "
        "allows for #headlines between 70 and 130, "
        "and errors if there's too many/few headlines.",
    ),
    *,
    cache_dir: Path = Path(".requests_cache"),
    use_cache: bool = True,
) -> None:
    """Fetch the days missing from the published daily CSVs from the Wayback Machine.

    GitHub skips scheduled runs, and a skipped day is a hole nothing else can
    fill: this heals those holes from the Wayback Machine.
    """
    today = datetime.datetime.now(datetime.timezone.utc).date()
    missing = [
        day
        for day in get_date_range(today - datetime.timedelta(days=days), today)
        if not daily_csv_path(out_dir, day).exists()
    ]
    if not missing:
        logger.info(f"Nothing missing in the last {days} days")
        return

    logger.info(f"Healing {len(missing)} day(s): {_format_days(missing)}")
    by_day: dict[datetime.date, list[Headline]] = {day: [] for day in missing}
    for newspaper in newspapers:
        per_day = asyncio.run(
            _fetch_wayback(
                newspaper,
                [_noon(day) for day in missing],
                cache_dir if use_cache else None,
            )
        )
        for day, headlines in zip(missing, per_day, strict=True):
            if _headline_count_is_ok(
                len(headlines), newspaper, allowed_difference_headlines
            ):
                by_day[day].extend(headlines)
            else:
                logger.warning(
                    f"Bad date: {newspaper.name} @ {day.strftime(date_fmt)} with "
                    f"{len(headlines)} headlines (expected "
                    f"{newspaper.expected_headlines})"
                )

    healed = 0
    for day, headlines in by_day.items():
        if not headlines:
            continue
        write_headlines(data_dir, day, headlines)
        write_daily_csv(out_dir, day, average_headline_scores_per_day(headlines))
        healed += 1

    if healed == 0:
        logger.warning("No day could be healed")
        return

    logger.info(f"Healed {healed} day(s)")
    rebuild_averages(out_dir)
    render_pages(out_dir)


@app.command()
def warm_models() -> None:
    """Download both scoring models into the HuggingFace cache."""
    download_models()


def render_pages(out_dir: Path) -> None:
    """Render `index.html`, `daily.html` and `about.html` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
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
def render_html(out_dir: Path = Path("public")) -> None:
    """Render the static HTML pages into `out_dir`."""
    render_pages(out_dir)


if __name__ == "__main__":
    app()
