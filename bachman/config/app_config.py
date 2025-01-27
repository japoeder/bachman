"""
Configuration module for the Bachman API.

This module handles logging setup and component initialization.
"""

import logging
import os
from bachman.core.components import Components

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def initialize_components():
    """Initialize all core components."""
    try:
        logger.info("Starting component initialization...")

        # Get Qdrant host from environment or use default
        qdrant_host = os.getenv("QDRANT_HOST", "192.168.1.10")

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
