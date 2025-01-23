"""
Handle various service tasks.
"""

import logging
from datetime import datetime
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bachman.models.embeddings import get_embeddings
from bachman.processors.vectorstore import VectorStore
from bachman.processors.sentiment import SentimentAnalyzer
from bachman.models.llm import get_groq_llm


def process_sentiment_analysis(
    data: dict, embeddings=None, vector_store=None, sentiment_analyzer=None
) -> dict:
    """Process sentiment analysis request."""
    try:
        # Use passed components or initialize new ones if not provided
        embeddings = embeddings or get_embeddings(provider="groq")
        vector_store = vector_store or VectorStore(
            embedding_function=embeddings, host="localhost", port=6379
        )
        sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer(get_groq_llm())

        # Process the text
        text = data.get("text", "")
        if not text:
            raise ValueError("No text provided for analysis")

        # Get embeddings and store in vector database
        embedding = embeddings.embed_query(text)

        # Analyze sentiment
        result = sentiment_analyzer.analyze_text(text)

        return {
            "sentiment": result,
            "embedding_size": len(embedding),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        raise
