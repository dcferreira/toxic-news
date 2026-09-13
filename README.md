# <img src="./toxic_news/templates/logo.svg" alt="Toxic News logo" height="40"/> Toxic News

This repo contains the code for https://toxicnews.dcferreira.com,
a website enabling automatic ranking of online media outlets
using machine learning models.

Once per day, the headlines from the frontpage of multiple online media outlets
are scraped, and sent to machine learning models.
The results are displayed in the website.

For more information, check out https://toxicnews.dcferreira.com/about.html.


## How it works

Everything runs as a single scheduled GitHub Actions job (`.github/workflows/update.yml`),
and **git is the only store** — there is no database and no server:

| Branch | Holds |
| --- | --- |
| `main` | the code |
| `data` | `headlines/YYYY/MM/DD.csv`, one row per scraped headline, exactly as the models scored it |
| `public` | the published site: `daily/YYYY/MM/DD.csv`, `averages/*.csv`, and the rendered HTML |

The job checks out all three, scrapes and scores every outlet in-process, writes
that day's `data` file, aggregates it into the day's `public/daily` file,
rebuilds every rolling-window average from the daily files, re-renders the site,
and pushes both branches.

A second, weekly job refetches days that GitHub skipped from the Wayback Machine,
so a missed run leaves a hole that gets healed rather than a gap that stays.

Serving the site is the `public` branch's `CNAME` pointing at GitHub Pages;
nothing has to be deployed.

## Setup

Note that if you just want to look at the results, you don't need to install this!
Just access the frontend at https://toxicnews.dcferreira.com.

If you do want to set it up locally, you need [uv](https://docs.astral.sh/uv/) installed.
The main entrypoint can be executed with

```bash
uv sync
uv run poe --help
```

Development tasks are defined with [poe the poet](https://poethepoet.natn.io/)
in `pyproject.toml`. Run `uv run poe` with no arguments to list them all; the
main ones are:

| Task | What it does |
| --- | --- |
| `poe cov` | tests (excluding the `slow` and `integration` markers) with coverage |
| `poe no-cov` | the same tests without coverage |
| `poe integration` / `poe slow` | the corresponding marker groups |
| `poe full` | the whole test suite in parallel |
| `poe lint` | `ruff check` plus `ruff format --check` |
| `poe format` | `ruff check --fix` plus `ruff format` |
| `poe types` | `ty check` |
| `poe update` | scrape, score and publish today's front pages, then rebuild the site |
| `poe models` | download the two scoring models into the Hugging Face cache (one-off) |
| `poe heal` | refetch the days missing from `public/daily` from the Wayback Machine |
| `poe e2e` | run the whole pipeline against the recorded front pages in `tests/assets/html` — no network, no live outlets |
| `poe build` | build the pipeline container image |
| `poe update-styles` | rebuild the Tailwind stylesheet |
| `poe explore` | open the notebook environment (`uv sync --extra explore` first) |

If you want to contribute to the project, please set up also the git pre-commit hooks
before you start making changes.
Doing that will guarantee a consistent style and basic code quality.
You can set up the pre-commit hooks with:

```bash
uv run pre-commit install
```

## Running the pipeline locally

`poe update` writes into `data/` and `public/`, the same layout the workflow
uses, so it needs those directories to exist:

```bash
git archive origin/public | tar -x -C public   # seed the published site, so the averages have history
mkdir -p data
uv run poe update
```

`poe update` hits the live outlet front pages. To exercise the whole pipeline —
scrape, score, store, aggregate, average, render — without touching them, run it
against a mock site serving the recorded pages from `tests/assets/html`:

```bash
uv run poe e2e
```

That starts `tests/mock_site.py` on an ephemeral port and points every outlet at
it (`tests/local_run.py` is the runner). It proves the wiring end to end; it says
nothing about whether the live outlets are still scrapable, because the recordings
are snapshots of historical front pages.

Both commands need the two models in the local Hugging Face cache: `models.py`
loads with `local_files_only=True`, so nothing is downloaded at run time.
Populate the cache once with

```bash
uv run poe models
```

`docker-compose.yml` is the container equivalent of `poe e2e` — it runs the same
pipeline against the same mock site, with the models baked into the image, so it
needs no local model cache but does need Docker with the Compose plugin:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from pipeline
```

### Dependency constraints

The runtime pins are from 2023 and no longer resolve to a working environment on
their own, so `[tool.uv] constraint-dependencies` in `pyproject.toml` holds
`transformers`, `huggingface_hub` and `click` at compatible versions. Don't
remove them without upgrading `optimum` and `typer` at the same time.
