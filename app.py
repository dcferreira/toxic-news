import os

from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi

from toxic_news.fetchers import Fetcher, Newspaper

app = FastAPI()

load_dotenv()
client: MongoClient = MongoClient(
    os.environ["MONGODB_URL"],
    server_api=ServerApi("1"),
)
db: Database = client[os.environ["DATABASE_NAME"]]


@app.post("/fetch")
async def fetch(name: str, language: str, url: str, xpath: str) -> int:
    fetcher = Fetcher(
        Newspaper.parse_obj(
            {
                "name": name,
                "language": language,
                "url": url,
                "xpath": xpath,
            }
        )
    )
    headlines = fetcher.classify()
    logger.info(f"Inserting {len(headlines)} rows in the database...")
    db.headlines.insert_many(h.dict() for h in headlines)

    return len(headlines)
