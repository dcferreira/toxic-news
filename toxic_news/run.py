import os
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from toxic_news.fetchers import newspapers

load_dotenv()
client: MongoClient = MongoClient(os.environ["MONGODB_URL"])
db: Database = client[os.environ["DATABASE_NAME"]]


class Row(BaseModel):
    name: str
    url: HttpUrl
    date: datetime
    toxicity: float


def main():
    headlines_table: Collection = db.headlines
    for newspaper in newspapers:
        headlines = newspaper.classify()
        headlines_table.insert_many([h.dict() for h in headlines])


if __name__ == "__main__":
    main()
