# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""FastAPI app that fetches, classifies and stores newspaper headlines.

Importing this module reads `MONGODB_URL` and `DATABASE_NAME` (after loading
`.env`) to open the MongoDB client, and logs the local Hugging Face cache when
`DEBUG == "1"`.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from huggingface_hub import scan_cache_dir
from loguru import logger
from pydantic import HttpUrl
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi

from toxic_news.fetchers import Fetcher
from toxic_news.newspapers import newspapers_dict
from toxic_news.queries import db_insert_headlines

app = FastAPI()

load_dotenv()
client: MongoClient = MongoClient(
    os.environ["MONGODB_URL"],
    server_api=ServerApi("1"),
)
db: Database = client[os.environ["DATABASE_NAME"]]

if os.environ.get("DEBUG", "") == "1":
    logger.debug(scan_cache_dir())


@app.post("/fetch")
async def fetch(url: HttpUrl) -> int:
    """Fetch, classify and store the headlines of one tracked newspaper.

    `url` is looked up in `newspapers_dict`, so an untracked newspaper raises a
    `KeyError`. Returns the number of headlines classified for the front page;
    headlines that are already stored are skipped by the database layer.
    """
    newspaper = newspapers_dict[url]
    fetcher = Fetcher(newspaper)
    headlines = fetcher.classify()
    logger.info(f"Inserting {len(headlines)} rows in the database...")
    db_insert_headlines(headlines, db=db)

    return len(headlines)
