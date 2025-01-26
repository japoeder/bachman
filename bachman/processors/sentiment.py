"""
Method for analyzing sentiment
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from langchain.schema import Document


logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Method for analyzing sentiment
    """

    def __init__(self, llm):
        """Initialize the sentiment analyzer."""
        self.llm = llm
        self.system_prompt = """You are a financial sentiment analyzer. Analyze the given text and return ONLY a JSON object with these keys:
{
    "dominant_sentiment": string (Bullish/Bearish/Neutral),
    "confidence_level": string (High/Medium/Low),
    "time_horizon": string (Short-term/Medium-term/Long-term),
    "key_themes": list[string] (3-5 main points),
    "trading_signal": string (Strong Buy/Buy/Hold/Sell/Strong Sell)
}

Do not include any other text, explanation, or markdown formatting. Return only the JSON object."""

    def analyze_text(
        self, documents: List[Document], instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze text with optional custom instructions.

        Args:
            documents: List of documents to analyze
            instructions: Custom instructions for the analysis (from construct_analysis)

        Returns:
            Analysis results as a dictionary
        """
        try:
            # Combine document content
            combined_text = "\n\n".join(doc.page_content for doc in documents)

            # Add timestamp using timezone.utc
            timestamp = datetime.now(timezone.utc).isoformat()

            # Create chat messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Analyze this text:\n\n{combined_text}"},
            ]

            # Get response from LLM
            response = self.llm.invoke(messages)

            # Parse JSON response
            try:
                result = json.loads(response.content)
                # Add timestamp to result
                result["timestamp"] = timestamp
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                logger.error(f"Raw response: {response.content}")
                raise

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    def analyze_by_source(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze sentiment grouped by document source.

        Args:
            documents: List of Document objects

        Returns:
            Dictionary with analysis results per source
        """
        try:
            # Group documents by source
            sources = {}
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                if source not in sources:
                    sources[source] = []
                sources[source].append(doc)

            # Analyze each source
            results = {}
            for source, docs in sources.items():
                results[source] = self.analyze_text(docs)

            return results

        except Exception as e:
            logging.error(f"Error in source-based analysis: {str(e)}")
            raise

    def analyze_with_timeframe(
        self, documents: List[Document], timeframe: str = "all"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment with specific timeframe focus.

        Args:
            documents: List of Document objects
            timeframe: One of "short", "medium", "long", or "all"

        Returns:
            Dictionary containing timeframe-specific analysis
        """
        try:
            # Modify prompt based on timeframe
            timeframe_prompt = (
                f"Focus on {timeframe}-term implications in your analysis.\n"
            )
            original_template = self.system_prompt

            self.system_prompt = timeframe_prompt + original_template
            result = self.analyze_text(documents)

            # Restore original prompt
            self.system_prompt = original_template

            return result

        except Exception as e:
            logging.error(f"Error in timeframe analysis: {str(e)}")
            raise
