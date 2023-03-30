import os

from dotenv import load_dotenv
from fastapi import FastAPI
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


@app.post("/fetch")
async def fetch(url: HttpUrl) -> int:
    newspaper = newspapers_dict[url]
    fetcher = Fetcher(newspaper)
    headlines = fetcher.classify()
    logger.info(f"Inserting {len(headlines)} rows in the database...")
    db_insert_headlines(headlines, db=db)

    return len(headlines)
