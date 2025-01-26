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
from bachman.config.app_config import setup_logging, initialize_components
from bachman.api.endpoints import create_app

# Initialize logger
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the Bachman API service."""
    # Setup logging
    setup_logging()

    # Initialize components
    app = create_app()

    # Initialize core components
    success = initialize_components()
    if not success:
        logger.error("Required components not initialized. Exiting.")
        sys.exit(1)

    # Run the server
    config = Config()
    config.bind = ["0.0.0.0:8713"]
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
