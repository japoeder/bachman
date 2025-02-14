"""
This module initializes and manages core components for the Bachman API.
"""

import logging
import json

# import json

from quantum_trade_utilities.core.get_path import get_path

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

    def __init__(self, qdrant_host: str, qdrant_port: int = 8716):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.embeddings = None
        self.vector_store = None
        self.sentiment_analyzer = None
        self.file_processor = None
        self.text_processor = None
        self.chunking_config = None
        self.embedding_config = None
        self.collection_config = None

    def initialize_components(self):
        """Initialize all components with dependencies."""
        try:
            logger.info("Starting component initialization...")

            # Load configs first
            self.chunking_config = self.load_chunking_config()
            self.embedding_config = self.load_embedding_config()
            self.collection_config = self.load_qdrant_collection_configs()

            # Initialize non-GPU components
            llm = get_groq_llm()
            self.sentiment_analyzer = SentimentAnalyzer(llm=llm)
            logger.info("Sentiment analyzer initialized successfully")

            return True
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return False

    def initialize_embeddings(self):
        """Initialize embedding-related components on demand."""
        self.embeddings = get_embeddings()
        self.vector_store = VectorStore(
            host=self.qdrant_host,
            port=self.qdrant_port,
            embedding_function=self.embeddings,
        )
        self.text_processor = TextProcessor(
            vector_store=self.vector_store,
            chunking_config=self.chunking_config,
        )
        self.file_processor = FileProcessor(
            text_processor=self.text_processor,
            vector_store=self.vector_store,
        )
        logger.info("Embedding components initialized successfully")

    def cleanup_embeddings(self):
        """Cleanup embedding resources."""
        if hasattr(self.embeddings, "clear_gpu_memory"):
            self.embeddings.clear_gpu_memory()
        self.embeddings = None
        self.vector_store = None
        self.text_processor = None
        self.file_processor = None
        logger.info("Embedding components cleaned up")

    def initialize_text_processor(self, vector_store):
        """Initialize the text processor."""
        return TextProcessor(vector_store=vector_store)

    def initialize_llm(self):
        """Initialize the LLM."""
        llm = get_groq_llm()
        return SentimentAnalyzer(llm=llm)

    def get_chunking_config(self, doc_type: str) -> dict:
        """Get chunking configuration for a specific document type."""
        return get_chunking_config(self.chunking_config, doc_type)

    def load_chunking_config(self):
        """Load chunking configuration from JSON file."""
        try:
            chunk_cfg = get_path("bachman_rag")
            with open(chunk_cfg, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info("Successfully loaded chunking configuration")
                return config.get("qdrant_chunking_config", {})
        except Exception as e:
            logger.error(f"Error loading chunking configuration: {str(e)}")
            return {}

    def load_embedding_config(self):
        """Load embedding configuration from JSON file."""
        try:
            embedding_cfg = get_path("bachman_rag")
            with open(embedding_cfg, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info("Successfully loaded embedding configurations")
                return config.get("embedding_configs", {})
        except Exception as e:
            logger.error(f"Error loading embedding configurations: {str(e)}")
            return {}

    def load_qdrant_collection_configs(self):
        """Load Qdrant collection configurations from JSON file."""
        try:
            collection_cfg = get_path("bachman_rag")
            with open(collection_cfg, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info("Successfully loaded Qdrant collection configurations")
                return config.get("collection_configs", {})
        except Exception as e:
            logger.error(f"Error loading Qdrant collection configurations: {str(e)}")
            return {}
