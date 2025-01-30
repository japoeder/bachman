"""
Configuration module for the Bachman API.

This module handles logging setup and component initialization.
"""

import logging
import os
import json

from quantum_trade_utilities.core.get_path import get_path

from bachman.core.components import Components

logger = logging.getLogger(__name__)

# Global configuration variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.1.10")
CHUNKING_CONFIGS = None
EMBEDDING_CONFIGS = None
COLLECTION_CONFIGS = None


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_chunking_config():
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


def load_embedding_config():
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


def load_qdrant_collection_configs():
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


# def initialize_app_config():
#     """Initialize application configurations."""
#     global CHUNKING_CONFIGS, EMBEDDING_CONFIGS, COLLECTION_CONFIGS

#     try:
#         logger.info("Loading configurations...")
#         CHUNKING_CONFIGS = load_chunking_config()
#         EMBEDDING_CONFIGS = load_embedding_config()
#         COLLECTION_CONFIGS = load_qdrant_collection_configs()
#         logger.info("All configurations loaded successfully")
#         return True
#     except Exception as e:
#         logger.error(f"Error loading configurations: {str(e)}")
#         return False


def initialize_components():
    """Initialize all core components."""
    try:
        logger.info("Starting component initialization...")

        # Get Qdrant host from environment or use default
        qdrant_host = os.getenv("QDRANT_HOST", "192.168.1.10")
        host_components = Components(qdrant_host=qdrant_host)

        # Load chunking configuration first
        chunk_cfg = load_chunking_config()
        Components.chunking_config = chunk_cfg
        logger.info("Chunking configurations loaded")

        # Load embedding configuration
        embedding_cfg = load_embedding_config()
        Components.embedding_config = embedding_cfg
        logger.info("Embedding configurations loaded")

        # Load RAG collection configurations
        collection_cfg = load_qdrant_collection_configs()
        Components.collection_config = collection_cfg
        logger.info("Qdrant collection configurations loaded")

        # Initialize all components using the Components class method

        success = host_components.initialize_components()

        if not success:
            logger.error("Failed to initialize components")
            return False

        logger.info("All components initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error during component initialization: {str(e)}")
        logger.exception("Full traceback:")
        return False
