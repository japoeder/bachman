"""
This module initializes and manages core components for the Bachman API.
"""

import logging

# import json
from bachman.models.embeddings import get_embeddings
from bachman.processors.vectorstore import VectorStore
from bachman.processors.sentiment import SentimentAnalyzer
from bachman.processors.text_processor import TextProcessor
from bachman.processors.file_processor import FileProcessor
from bachman.models.llm import get_groq_llm
from bachman.processors.chunking import get_chunking_config

# from bachman.core.interfaces import TaskTracker
# from bachman.core.task_tracker import AsyncTaskTracker

logger = logging.getLogger(__name__)


class Components:
    """
    A class to hold references to core components.
    """

    embeddings = None
    vector_store: VectorStore = None
    sentiment_analyzer = None
    file_processor: FileProcessor = None
    text_processor = None
    chunking_config = None  # Will store the loaded chunking configuration
    # task_tracker: TaskTracker = None
    qdrant_host: str = None
    qdrant_port: int = 8716

    @classmethod
    def initialize_components(cls, qdrant_host: str):
        """Initialize all components with dependencies."""
        try:
            logger.info("Starting component initialization...")
            logger.debug(
                f"Current Components state - vector_store: {cls.vector_store}, sentiment_analyzer: {cls.sentiment_analyzer}"
            )

            # Store Qdrant connection info
            cls.qdrant_host = qdrant_host
            logger.debug(f"Setting Qdrant host to: {qdrant_host}")

            # Initialize embeddings
            cls.embeddings = get_embeddings()
            logger.info("Embeddings model initialized successfully")

            # Initialize vector store
            cls.vector_store = VectorStore(
                host=qdrant_host,
                port=cls.qdrant_port,
                embedding_function=cls.embeddings,
            )
            logger.info(
                f"Vector store initialized successfully for {qdrant_host}:{cls.qdrant_port}"
            )
            logger.debug(f"Vector store client: {cls.vector_store.client}")

            # Initialize text processor with chunking config
            cls.text_processor = TextProcessor(
                vector_store=cls.vector_store,
                chunking_config=cls.chunking_config,  # Pass the chunking config
            )
            logger.info("Text processor initialized successfully")

            # Initialize task tracker
            # cls.task_tracker = AsyncTaskTracker()
            logger.info("Task tracker initialized successfully")

            # Initialize file processor with text processor
            cls.file_processor = FileProcessor(
                text_processor=cls.text_processor,
                # task_tracker=cls.task_tracker,
                vector_store=cls.vector_store,
            )
            logger.info("File processor initialized successfully")

            # Initialize LLM and sentiment analyzer
            llm = get_groq_llm()
            cls.sentiment_analyzer = SentimentAnalyzer(llm=llm)
            logger.info("Sentiment analyzer initialized successfully")

            logger.debug(
                f"Final Components state - vector_store: {cls.vector_store}, sentiment_analyzer: {cls.sentiment_analyzer}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return False

    @classmethod
    def initialize_text_processor(cls):
        """Initialize the text processor."""
        cls.text_processor = TextProcessor(cls.vector_store)

    @classmethod
    def initialize_llm(cls):
        """Initialize the LLM."""
        llm = get_groq_llm()
        cls.sentiment_analyzer = SentimentAnalyzer(llm=llm)

    @classmethod
    def get_chunking_config(cls, doc_type: str) -> dict:
        """Get chunking configuration for a specific document type."""
        return get_chunking_config(cls.chunking_config, doc_type)
