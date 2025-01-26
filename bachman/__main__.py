"""
Main module for the Bachman API.

This module provides the main entry point for the Bachman API service,
handling request routing, authentication, and core functionality initialization.
"""

# Standard library imports
import sys
import os
import logging
import signal
from functools import wraps
import argparse
from typing import Optional
import uuid
import asyncio
import json
import requests


# Third-party imports
import dotenv
from langchain.schema import Document
from flask import Flask, request, jsonify

# from asgiref.wsgi import WsgiToAsgi  # If needed for ASGI server

# Local application imports
from quantum_trade_utilities.data.load_credentials import load_credentials
from quantum_trade_utilities.core.get_path import get_path

dotenv.load_dotenv(get_path("env"))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bachman.utils.format_results import format_analysis_results
from bachman.loaders.document_loader import DocumentLoader
from bachman.processors.vectorstore import VectorStore
from bachman.processors.sentiment import SentimentAnalyzer
from bachman.models.llm import get_groq_llm
from bachman.models.embeddings import get_embeddings
from bachman.processors.text_processor import TextProcessor
from bachman.processors.chunking import ChunkingConfig
from bachman.processors.document_types import DocumentType
from bachman.processors.file_processor import FileProcessor

# Initialize logger
logger = logging.getLogger(__name__)

# Configure logging
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# Initialize Qdrant credentials
QUADRANT_HOST, QUADRANT_PORT = load_credentials(get_path("creds"), "qdrant_ds")

# Initialize Flask app
app = Flask(__name__)

app_log_path = get_path("log")

logging.debug("This is a test log message.")


def handle_sigterm(*args):
    """Handle SIGTERM signal."""
    print("Received SIGTERM, shutting down gracefully...")
    sys.exit(0)


# Register SIGTERM handler once
signal.signal(signal.SIGTERM, handle_sigterm)


def requires_api_key(f):
    """Decorator to require an API key for a route."""

    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            api_key = request.headers.get("x-api-key")
            if not api_key:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "API key is missing",
                            "error_type": "authentication",
                        }
                    ),
                    401,
                )
            if api_key != os.getenv("BACHMAN_API_KEY"):
                logging.warning(f"Invalid API key attempt: {api_key[:8]}...")
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Invalid API key",
                            "error_type": "authentication",
                        }
                    ),
                    403,
                )

            return f(*args, **kwargs)

        except Exception as e:
            logging.error(f"Authentication error: {str(e)}")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Authentication system error",
                        "error_type": "system",
                    }
                ),
                500,
            )

    return decorated


# Initialize core components
embeddings = None
vector_store = None
sentiment_analyzer = None
file_processor = None

try:
    logger.info("Starting component initialization...")

    # Initialize embeddings model
    embeddings = get_embeddings()
    logger.info("Embeddings model initialized successfully")

    # Initialize vector store client
    logger.info("Connecting to Qdrant at 192.168.1.10:8716")
    vector_store = VectorStore(
        host="192.168.1.10",
        port=8716,
        embedding_function=embeddings,
    )
    logger.info("Vector store initialized successfully at 192.168.1.10:8716")

    # Initialize text processor
    text_processor = TextProcessor(vector_store)
    logger.info("Text processor initialized successfully")

    # Initialize LLM and sentiment analyzer
    llm = get_groq_llm()
    sentiment_analyzer = SentimentAnalyzer(llm=llm)
    logger.info("Sentiment analyzer initialized successfully")

    # Initialize file processor
    file_processor = FileProcessor(text_processor)
    logger.info("File processor initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    logger.exception("Full traceback:")
    raise


@app.route("/bachman/analyze", methods=["POST"])
@requires_api_key
def analyze():
    """
    Text analysis endpoint.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract and validate collection name
        collection_name = data.get("collection_name")
        if not collection_name:
            return jsonify({"error": "collection_name is required"}), 400

        logger.info(
            f"Processing analysis request for ticker: {data.get('ticker', 'UNKNOWN')} in collection: {collection_name}"
        )

        # Get text from request
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Ensure collection exists before processing
        vector_store.ensure_collection(collection_name)

        # Create document
        doc = Document(
            page_content=text,
            metadata={
                "ticker": data.get("ticker", "UNKNOWN"),
                "type": data.get("analysis_type", "sentiment"),
                "source": data.get("load_type", "live"),
                "id": str(uuid.uuid4()),
            },
        )

        # Analyze sentiment first
        sentiment_result = sentiment_analyzer.analyze_text([doc])

        # Format results using the utility function
        format_analysis_results(
            analysis_type=doc.metadata["type"],
            result=sentiment_result,
            metadata=doc.metadata,
        )

        # Store document with metadata in the specified collection
        storage_result = vector_store.store_vectors(
            collection_name=collection_name,
            texts=[text],
            metadatas=[doc.metadata],
            # force_recreate=data.get("force_recreate", False),
        )
        logger.info(f"Successfully stored document in collection: {collection_name}")

        # Return both results
        return (
            jsonify(
                {"sentiment_analysis": sentiment_result, "storage": storage_result}
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


def analyze_pdf(
    pdf_path: str, output_dir: Optional[str] = None, print_results: bool = True
) -> dict:
    """
    Analyze sentiment from a PDF file.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to store processed files (optional)
        print_results: Whether to print results to console

    Returns:
        Dictionary containing analysis results
    """
    try:
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "processed_docs")

        # Initialize components
        loader = DocumentLoader(
            raw_dir=os.path.dirname(pdf_path), processed_dir=output_dir
        )

        embeddings = get_embeddings()
        vectorstore = VectorStore(host="192.168.1.10", port=8716)

        llm = get_groq_llm()
        analyzer = SentimentAnalyzer(llm)

        # Process PDF
        ticker = "DEMO"  # For single file demo
        report_type = "analysis"
        loader.process_pdf(pdf_path, ticker, report_type)

        # Load processed documents
        documents = loader.load_processed_documents(ticker, report_type)

        # Get embeddings for documents
        texts = [doc.page_content for doc in documents]
        vectors = embeddings.embed_documents(texts)
        metadata = [doc.metadata for doc in documents]

        # Store in Qdrant
        collection_name = f"{ticker}_{report_type}"
        vectorstore.create_collection(collection_name)
        vectorstore.store_vectors(collection_name, texts, vectors, metadata)

        # Analyze sentiment
        results = analyzer.analyze_text(documents)

        if print_results:
            print("\nSentiment Analysis Results:")
            print("==========================")
            print(f"Overall Sentiment: {results['dominant_sentiment']}")
            print(f"Confidence: {results['confidence_level']}")
            print(f"Time Horizon: {results['time_horizon']}")
            print("\nKey Themes:")
            for theme in results["key_themes"]:
                print(f"- {theme}")
            print(f"\nTrading Signal: {results['trading_signal']}")

        return results

    except Exception as e:
        logging.error(f"Error analyzing PDF: {str(e)}")
        raise


@app.route("/bachman/process_text", methods=["POST"])
@requires_api_key
def process_text():
    """
    Text processing endpoint without sentiment analysis.
    Accepts chunking configuration parameters.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract and validate collection name
        collection_name = data.get("collection_name")
        if not collection_name:
            return jsonify({"error": "collection_name is required"}), 400

        # Get text from request
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Extract chunking configuration if provided
        chunking_config = None
        if "chunking_config" in data:
            try:
                chunking_config = ChunkingConfig(
                    strategy=data["chunking_config"].get("strategy", "recursive"),
                    chunk_size=data["chunking_config"].get("chunk_size", 4096),
                    chunk_overlap=data["chunking_config"].get("chunk_overlap", 200),
                    separators=data["chunking_config"].get("separators"),
                    min_chunk_size=data["chunking_config"].get("min_chunk_size", 100),
                )
                logger.info(f"Using custom chunking configuration: {chunking_config}")
            except Exception as e:
                logger.warning(
                    f"Invalid chunking configuration provided: {e}. Using defaults."
                )

        # Run the coroutine in the event loop
        result = asyncio.run(
            text_processor.process_text(
                text=text,
                collection_name=collection_name,
                metadata=data.get("metadata"),
                skip_if_exists=data.get("skip_if_exists", False),
                chunking_config=chunking_config,
            )
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/bachman/process_file", methods=["POST"])
@requires_api_key
def process_file():
    """
    File processing endpoint with flexible options.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        file_path = data.get("file_path")
        collection_name = data.get("collection_name")

        # Optional fields with defaults
        document_type = DocumentType(data.get("document_type", "generic"))
        process_sentiment = data.get("process_sentiment", True)

        result = file_processor.process_file(
            file_path=file_path,
            collection_name=collection_name,
            document_type=document_type,
            process_sentiment=process_sentiment,
            chunking_config=data.get("chunking_config"),
            metadata=data.get("metadata"),
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/bachman/get_sentiment", methods=["POST"])
@requires_api_key
def get_sentiment():
    """
    Retrieve or generate sentiment for previously processed document.
    """
    try:
        data = request.json
        doc_id = data.get("doc_id")
        if not doc_id:
            return jsonify({"error": "doc_id is required"}), 400

        result = sentiment_analyzer.process_document(
            doc_id=doc_id, collection_name=data.get("collection_name")
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing sentiment request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/bachman/search", methods=["POST"])
@requires_api_key
def search():
    """
    Search endpoint to query existing documents in the vector store.
    Supports both doc_id lookups and general metadata searches.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Log just the search request details
        logger.info("=" * 80)
        logger.info("SEARCH REQUEST DETAILS")
        logger.info("=" * 80)
        logger.info(f"Request data: {json.dumps(data, indent=2)}")

        # Extract and validate collection name
        collection_name = data.get("collection_name")
        if not collection_name:
            return jsonify({"error": "collection_name is required"}), 400

        # Get query parameters
        metadata_filter = data.get("metadata", {})
        if not metadata_filter:
            return jsonify({"error": "Query metadata is required"}), 400

        # Prepare filter conditions
        filter_conditions = []
        for key, value in metadata_filter.items():
            filter_conditions.append({"key": key, "match": {"value": value}})

        # Make the search request
        search_url = (
            f"http://{vector_store.host}:8716/collections/{collection_name}/search"
        )
        search_payload = {
            "query_vector": [0] * 1024,
            "filter": {"must": filter_conditions},
            "limit": 100,
        }

        response = requests.post(
            search_url,
            json=search_payload,
            timeout=10,
        )

        if response.status_code == 200:
            raw_results = response.json()

            # Format the results
            formatted_results = []
            for hit in raw_results:
                metadata = hit.get("payload", {})
                formatted_result = {
                    "id": hit.get("id"),
                    "metadata": {
                        "doc_id": metadata.get("doc_id"),
                        "ticker": metadata.get("ticker"),
                        "source": metadata.get("source"),
                        "report_type": metadata.get("report_type"),
                        "timestamp": metadata.get("timestamp"),
                    },
                    "chunks": metadata.get("chunks", []),
                    "text_preview": metadata.get("text", "")[:100] + "..."
                    if metadata.get("text")
                    else None,
                }
                formatted_results.append(formatted_result)

            # Log just the formatted results
            logger.info("\nRESULTS")
            logger.info(json.dumps(formatted_results, indent=2))
            logger.info("=" * 80)

            return (
                jsonify(
                    {"count": len(formatted_results), "results": formatted_results}
                ),
                200,
            )
        else:
            logger.error(f"Search failed with status {response.status_code}")
            return (
                jsonify({"error": f"Search failed with status {response.status_code}"}),
                500,
            )

    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        return jsonify({"error": str(e)}), 500


def main():
    """
    Main method to call RAG bot.
    """
    parser = argparse.ArgumentParser(description="Analyze sentiment from PDF documents")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output-dir", help="Directory to store processed files")
    parser.add_argument(
        "--no-print", action="store_true", help="Don't print results to console"
    )

    args = parser.parse_args()

    analyze_pdf(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        print_results=not args.no_print,
    )


if __name__ == "__main__":
    if (
        embeddings is None
        or vector_store is None
        or sentiment_analyzer is None
        or file_processor is None
    ):
        logger.error("Required components not initialized. Exiting.")
        sys.exit(1)
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:8713"]
    asyncio.run(serve(app, config))
