"""
This module initializes and manages core components for the Bachman API.
"""

import logging
from bachman.models.embeddings import get_embeddings
from bachman.processors.vectorstore import VectorStore
from bachman.processors.sentiment import SentimentAnalyzer
from bachman.processors.text_processor import TextProcessor
from bachman.processors.file_processor import FileProcessor
from bachman.models.llm import get_groq_llm

logger = logging.getLogger(__name__)


class Components:
    """
    A class to hold references to core components.
    """

    embeddings = None
    vector_store = None
    sentiment_analyzer = None
    file_processor = None
    text_processor = None


def initialize_components():
    """Initialize all core components."""
    try:
        logger.info("Starting component initialization...")

        # Initialize embeddings model
        Components.embeddings = get_embeddings()
        logger.info("Embeddings model initialized successfully")

        # Initialize vector store client
        logger.info("Connecting to Qdrant at 192.168.1.10:8716")
        Components.vector_store = VectorStore(
            host="192.168.1.10",
            port=8716,
            embedding_function=Components.embeddings,
        )
        logger.info("Vector store initialized successfully")

        # Initialize text processor
        Components.text_processor = TextProcessor(Components.vector_store)
        logger.info("Text processor initialized successfully")

        # Initialize LLM and sentiment analyzer
        llm = get_groq_llm()
        Components.sentiment_analyzer = SentimentAnalyzer(llm=llm)
        logger.info("Sentiment analyzer initialized successfully")

        # Initialize file processor
        Components.file_processor = FileProcessor(Components.text_processor)
        logger.info("File processor initialized successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        logger.exception("Full traceback:")
        return False
