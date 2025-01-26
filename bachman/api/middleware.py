"""
Middleware for the Bachman API.

This module provides authentication and error handling for the Bachman API.
"""

import os
import logging
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)


def requires_api_key(f):
    """Decorator to require an API key for a route."""

    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            api_key = request.headers.get("x-api-key")
            if not api_key:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "API key is missing",
                            "error_type": "authentication",
                        }
                    ),
                    401,
                )
            if api_key != os.getenv("BACHMAN_API_KEY"):
                logging.warning(f"Invalid API key attempt: {api_key[:8]}...")
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Invalid API key",
                            "error_type": "authentication",
                        }
                    ),
                    403,
                )

            return f(*args, **kwargs)

        except Exception as e:
            logging.error(f"Authentication error: {str(e)}")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Authentication system error",
                        "error_type": "system",
                    }
                ),
                500,
            )

    return decorated
