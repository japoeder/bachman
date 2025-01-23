"""
Utility functions for formatting analysis results.

This module provides functions to format and log different types of analysis results
in a consistent and readable way. It supports multiple analysis types including:
- Sentiment analysis
- Technical analysis (planned)
- Fundamental analysis (planned)

Each formatter takes the raw analysis results and metadata and outputs formatted logs
with relevant information organized in a clear structure.

Example:
    format_analysis_results(
        analysis_type="sentiment",
        result=sentiment_results,
        metadata=document_metadata
    )

"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def format_sentiment_results(result: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Format and log sentiment analysis results.

    Args:
        result: The sentiment analysis results dictionary
        metadata: The document metadata containing ticker and analysis type
    """
    # logger.info("\n" + "=" * 50)
    logger.info("Sentiment Analysis Results:")
    logger.info("Ticker: %s", metadata.get("ticker", "N/A"))
    logger.info("Analysis Type: %s", metadata.get("type", "N/A"))
    logger.info("-" * 50)
    logger.info("Dominant Sentiment: %s", result.get("dominant_sentiment", "N/A"))
    logger.info("Confidence Level: %s", result.get("confidence_level", "N/A"))
    logger.info("Time Horizon: %s", result.get("time_horizon", "N/A"))
    logger.info("Trading Signal: %s", result.get("trading_signal", "N/A"))
    if "key_themes" in result:
        logger.info("Key Themes:")
        for theme in result["key_themes"]:
            logger.info("• %s", theme)
    logger.info("=" * 50)


def format_analysis_results(
    analysis_type: str, result: Dict[str, Any], metadata: Dict[str, Any]
) -> None:
    """Route the formatting based on analysis type.

    Args:
        analysis_type: Type of analysis (e.g., 'sentiment', 'technical', etc.)
        result: The analysis results dictionary
        metadata: The document metadata
    """
    formatters = {
        "sentiment": format_sentiment_results,
        # Add more formatters as needed:
        # 'technical': format_technical_results,
        # 'fundamental': format_fundamental_results,
    }

    formatter = formatters.get(analysis_type)
    if formatter:
        formatter(result, metadata)
    else:
        logger.warning(f"No formatter found for analysis type: {analysis_type}")
        logger.info(f"Raw results: {result}")
