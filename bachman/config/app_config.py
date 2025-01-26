"""
Configuration module for the Bachman API.

This module handles logging setup and component initialization.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up logging format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "bachman.log"), maxBytes=10485760, backupCount=5  # 10MB
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    # Set logging levels for specific modules
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def initialize_components():
    """Initialize all required components for the application."""
    try:
        from bachman.core.components import initialize_components

        return initialize_components()
    except Exception as e:
        logging.error(f"Failed to initialize components: {str(e)}")
        return False
