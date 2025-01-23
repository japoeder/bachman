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
import dotenv

# Third-party imports
from langchain.schema import Document
from flask import Flask, request, jsonify

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

try:
    logger.info("Starting component initialization...")

    # Initialize embeddings model
    embeddings = get_embeddings()
    logger.info("Embeddings model initialized successfully")

    # Initialize vector store client - explicitly set host and port
    logger.info("Connecting to Qdrant at 192.168.1.10:8716")
    vector_store = VectorStore(
        host="192.168.1.10",
        port=8716,
        embedding_function=embeddings,
    )
    logger.info("Vector store initialized successfully at 192.168.1.10:8716")

    # Initialize LLM and sentiment analyzer
    llm = get_groq_llm()
    sentiment_analyzer = SentimentAnalyzer(llm=llm)
    logger.info("Sentiment analyzer initialized successfully")

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
            force_recreate=data.get("force_recreate", False),
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
    if embeddings is None or vector_store is None or sentiment_analyzer is None:
        logger.error("Required components not initialized. Exiting.")
        sys.exit(1)
    app.run(host="0.0.0.0", port=8713)
