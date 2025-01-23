"""
Test RAG queries on financial documents.
"""

import os
import sys
import dotenv
from quantum_trade_utilities.core.get_path import get_path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(get_path("env"))

# from quantum_trade_utilities.core.load_credentials import load_credentials
from bachman.rag_util import setup_rag_pipeline, query_financial_data


def test_financial_query():
    """
    Test RAG queries on financial documents.
    """
    # Load environment
    if creds_file_path is None:
        creds_file_path = get_path("creds")

    # Load Alpaca API credentials from JSON file
    # API_KEY, API_SECRET, BASE_URL = load_credentials(creds_file_path)

    # Setup paths
    persist_dir = os.path.join(os.getcwd(), "_data/chromadb")

    # Initialize RAG pipeline
    qa_chain = setup_rag_pipeline(persist_dir)

    # Test query
    query = "What was AAPL's revenue in fiscal year 2024?"
    result = query_financial_data(query, qa_chain, "FIXME")

    print("\nQuery:", query)
    print("\nAnswer:", result["answer"])
    print("\nSources:", result["sources"])


if __name__ == "__main__":
    test_financial_query()
