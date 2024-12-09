"""
Main module for the Bachman API.
"""
import sys
import os
import logging
import signal
from functools import wraps
from logging.handlers import RotatingFileHandler
import dotenv
from flask import Flask, request, jsonify
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
from peft import PeftModel  # 0.5.0

# import torch

# from bachman._utils.mongo_conn import mongo_conn
# from bachman._utils.load_credentials import load_credentials
# from bachman._utils.mongo_coll_verification import confirm_mongo_collect_exists
from bachman._utils.get_path import get_path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv()


def handle_sigterm(*args):
    """Handle SIGTERM signal."""
    print("Received SIGTERM, shutting down gracefully...")
    # Perform any cleanup here
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)

app = Flask(__name__)


app_log_path = get_path("log")

# Configure logging with RotatingFileHandler
handler = RotatingFileHandler(
    app_log_path, maxBytes=5 * 1024 * 1024, backupCount=5
)  # 5 MB max size, keep 5 backups
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
)
handler.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)

# Add console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logging.debug("This is a test log message.")


def requires_api_key(f):
    """Decorator to require an API key for a route."""

    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != os.getenv("HENDRICKS_API_KEY"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated


@app.route("/load_tickers", methods=["POST"])
@requires_api_key
def load_tickers():
    """Endpoint to load a new stock ticker into the database."""
    data = request.json
    print(f"Received data: {data}")
    tickers = None
    collection_name = "rawPriceColl"
    return (
        jsonify(
            {"status": f"{tickers} dataframe loaded into {collection_name} collection."}
        ),
        202,
    )


if __name__ == "__main__":
    # Load Models
    base_model = "meta-llama/Meta-Llama-3-8B"
    peft_model = "FinGPT/fingpt-mt_llama3-8b_lora"
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    app.run(debug=True, host="0.0.0.0", port=8002)
