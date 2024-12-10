"""
Load Alpaca API credentials from JSON file.
"""

import json
from dotenv import load_dotenv

load_dotenv()


def load_credentials(file_path):
    """
    Load Alpaca API credentials from JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        creds = json.load(file)
    alpaca_creds = creds["alpaca_api"]
    return (
        alpaca_creds["API_KEY"],
        alpaca_creds["API_SECRET"],
        alpaca_creds["PAPER_URL"],
    )
