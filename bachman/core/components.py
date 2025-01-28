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
    # task_tracker: TaskTracker = None
    qdrant_host: str = None
    qdrant_port: int = 8716

    @classmethod
    def initialize_components(cls, qdrant_host: str):
        """Initialize all components with dependencies."""
        try:
            logger.info("Starting component initialization...")

            # Store Qdrant connection info
            cls.qdrant_host = qdrant_host

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

            # Initialize text processor first
            cls.text_processor = TextProcessor(cls.vector_store)
            logger.info("Text processor initialized successfully")

            # Initialize task tracker
            # cls.task_tracker = AsyncTaskTracker()
            logger.info("Task tracker initialized successfully")

            # Initialize file processor with text processor
            cls.file_processor = FileProcessor(
                text_processor=cls.text_processor,  # Now text_processor is initialized
                # task_tracker=cls.task_tracker,
                vector_store=cls.vector_store,
            )
            logger.info("File processor initialized successfully")

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

    # @classmethod
    # def initialize_sentiment_analyzer(cls):
    #     """Initialize the sentiment analyzer."""
    #     pass


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

        # Initialize task tracker
        # Components.task_tracker = AsyncTaskTracker()
        logger.info("Task tracker initialized successfully")

        # Initialize file processor with all required arguments
        Components.file_processor = FileProcessor(
            text_processor=Components.text_processor,
            task_tracker=Components.task_tracker,
            vector_store=Components.vector_store,
        )
        logger.info("File processor initialized successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        logger.exception("Full traceback:")
        return False
