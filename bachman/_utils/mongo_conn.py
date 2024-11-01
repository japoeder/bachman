import os
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient

def mango_conn():
    mongo_user = os.getenv('MONGO_USER')
    mongo_password = os.getenv('MONGO_PASSWORD')
    conn = f"mongodb://{mongo_user}:{mongo_password}@192.168.1.10:27017/stocksDB?directConnection=true&serverSelectionTimeoutMS=2000&authSource=admin&appName=mongosh+2.3.2"
    # Connect to MongoDB
    client = MongoClient(conn)
    db = client['stocksDB']
    return db