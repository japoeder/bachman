"""
Connect to MongoDB.
"""

import os
from pymongo import MongoClient


def mongo_conn():
    """
    Connect to MongoDB.
    """
    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST")
    mongo_port = os.getenv("MONGO_PORT")
    conn = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/stocksDB?directConnection=true&serverSelectionTimeoutMS=2000&authSource=admin&appName=mongosh+2.3.2"
    # Connect to MongoDB
    client = MongoClient(conn)
    db = client["stocksDB"]
    return db
