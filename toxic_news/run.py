import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from toxic_news.fetchers import Fetcher, newspapers

load_dotenv()
client: MongoClient = MongoClient(os.environ["MONGODB_URL"])
db: Database = client[os.environ["DATABASE_NAME"]]


def main():
    headlines_table: Collection = db.headlines
    for newspaper in newspapers:
        fetcher = Fetcher(newspaper)
        headlines = fetcher.classify()
        headlines_table.insert_many([h.dict() for h in headlines])


if __name__ == "__main__":
    main()
