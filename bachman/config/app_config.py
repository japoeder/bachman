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


def initialize_components():
    """Initialize all core components."""
    try:
        logger.info("Starting component initialization...")

        # Get Qdrant host from environment or use default
        qdrant_host = os.getenv("QDRANT_HOST", "192.168.1.10")

        # Load chunking configuration first
        chunk_cfg = load_chunking_config()
        Components.chunking_config = chunk_cfg
        logger.info("Chunking configurations loaded")

        # Initialize all components using the Components class method
        success = Components.initialize_components(qdrant_host=qdrant_host)

        if not success:
            logger.error("Failed to initialize components")
            return False

        logger.info("All components initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error during component initialization: {str(e)}")
        logger.exception("Full traceback:")
        return False
