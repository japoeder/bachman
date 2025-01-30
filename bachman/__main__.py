"""
Main module for the Bachman API.

This module provides the main entry point for the Bachman API service,
handling core functionality initialization.
"""

import sys
import os
import logging
import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bachman.config.app_config import (
    setup_logging,
    # initialize_components,
    # initialize_app_config,
)
from bachman.api.endpoints import create_app

# Initialize logger
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the Bachman API service."""
    try:
        logger.info("Starting Bachman API...")

        # Setup logging
        setup_logging()

        # Initialize components
        # logger.info("Initializing components...")
        app = create_app()

        # Initialize core components
        # success = initialize_components()
        # if not success:
        #     logger.error("Required components not initialized. Exiting.")
        #     sys.exit(1)

        # Initialize configurations
        # logger.info("Initializing app config...")
        # initialize_app_config()

        # Run the server
        logger.info("Starting Flask server on port 8713...")
        config = Config()
        config.bind = ["0.0.0.0:8713"]
        asyncio.run(serve(app, config))
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
