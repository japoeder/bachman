"""
Verify the existence of a MongoDB collection and create it if it doesn't exist.
"""

from pymongo.errors import OperationFailure
from bachman._utils.mongo_conn import mongo_conn


def confirm_mongo_collect_exists(collection_name):
    """
    Verify the existence of a MongoDB collection and create it if it doesn't exist.
    """
    # Get the database connection
    db = mongo_conn()

    # Check if the collection exists
    if collection_name in db.list_collection_names():
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Create the collection by inserting a dummy document
        try:
            db[collection_name].insert_one({"_init": True})
            db[collection_name].delete_one({"_init": True})
            print(f"Collection '{collection_name}' created successfully.")
        except OperationFailure as e:
            print(f"Failed to create collection '{collection_name}': {e}")

    # Check privileges (this is a placeholder, as privilege management is
    # typically done at the database admin level)
    try:
        # Attempt a simple operation to check privileges
        db[collection_name].find_one()
        print(f"Privileges to operate on '{collection_name}' are confirmed.")
    except OperationFailure as e:
        print(f"Insufficient privileges to operate on '{collection_name}': {e}")
