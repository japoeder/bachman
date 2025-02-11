"""
Main module for the Bachman API.

This module provides the main entry point for the Bachman API service,
handling core functionality initialization.
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bachman.config.app_config import setup_logging
from bachman.api.endpoints import create_app

# Initialize logger
logger = logging.getLogger(__name__)

# Create the Flask application instance
app = create_app()


def main():
    """Main entry point for the Bachman API service."""
    try:
        logger.info("Starting Bachman API...")
        setup_logging()
        logger.info("Starting Flask server on port 8713...")
        app.run(host="0.0.0.0", port=8713)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
