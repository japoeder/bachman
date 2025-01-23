"""
Language Model module.

This module provides functionality for initializing and configuring language models,
specifically focusing on the Groq LLM implementation. It handles model initialization,
API key management, and default parameter settings.
"""

# Standard library imports
import os
import logging
from typing import Dict, Any, List

# Third-party imports
from groq import Groq
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, Document
from langchain.prompts import ChatPromptTemplate

from quantum_trade_utilities.data.load_credentials import load_credentials
from quantum_trade_utilities.core.get_path import get_path

# Set up logger
logger = logging.getLogger(__name__)


class GroqCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Groq LLM."""

    def __init__(self):
        # Load Groq API key
        self.groq_api_key = load_credentials(get_path("creds"), "groq_api")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Log when LLM starts processing."""
        logging.debug(f"Starting Groq LLM with prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log when LLM completes processing."""
        logging.debug(f"Groq LLM completed processing: {response}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Log any LLM errors."""
        logging.error(f"Groq LLM error: {str(error)}")


def get_groq_llm():
    """Initialize and return a Groq client."""
    try:
        logger.info("Initializing Groq client...")

        # Load API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        # Initialize Groq client
        client = Groq(api_key=api_key)

        def invoke(messages):
            response = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message

        # Add invoke method to client
        client.invoke = invoke

        logger.info("Groq client initialized successfully")
        return client

    except Exception as e:
        logger.error(f"Error initializing Groq client: {str(e)}")
        raise


def get_model_info(llm: ChatGroq) -> Dict[str, Any]:
    """
    Get information about the configured LLM.

    Args:
        llm: ChatGroq instance

    Returns:
        Dictionary containing model information
    """
    return {
        "model_name": llm.model,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "model_type": "Groq LLM",
        "streaming": llm.streaming,
    }


class LLMConfig:
    """Configuration settings for LLM usage."""

    DEFAULT_PROMPTS = {
        "sentiment": """
        Analyze the sentiment in the following financial text and relevant context:
        
        Text: {text}
        
        Related Context:
        {context}
        
        Provide a detailed analysis including:
        1. Overall sentiment (bullish/bearish/neutral)
        2. Confidence level (high/medium/low)
        3. Key factors influencing the sentiment
        4. Time horizon (short/medium/long term)
        5. Trading implications
        """,
        "summary": """
        Summarize the key points from the following financial document and related context:
        
        Document: {text}
        
        Related Context:
        {context}
        
        Focus on:
        1. Material information
        2. Significant changes
        3. Important trends
        4. Key metrics and their implications
        """,
        "risk": """
        Identify potential risks and concerns in the following text and related context:
        
        Text: {text}
        
        Related Context:
        {context}
        
        Provide:
        1. Specific risk factors
        2. Potential impact severity
        3. Likelihood of occurrence
        4. Mitigation suggestions
        5. Areas requiring further analysis
        """,
    }

    @staticmethod
    def get_prompt(prompt_type: str, text: str, context_docs: List[Document]) -> str:
        """
        Get a formatted prompt with context.

        Args:
            prompt_type: Type of analysis
            text: Main text to analyze
            context_docs: Related documents from vector store
        """
        # Format context from similar documents
        context = "\n\n".join(
            [
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_docs)
            ]
        )

        # Get prompt template
        template = LLMConfig.DEFAULT_PROMPTS.get(
            prompt_type, LLMConfig.DEFAULT_PROMPTS["sentiment"]
        )

        # Create chat prompt
        prompt = ChatPromptTemplate.from_template(template)

        return prompt.format(text=text, context=context)

    @staticmethod
    def get_model_params(task_type: str) -> Dict[str, Any]:
        """Get recommended model parameters for different tasks."""
        params = {
            "sentiment": {"temperature": 0.0, "max_tokens": 2048},
            "summary": {"temperature": 0.1, "max_tokens": 4096},
            "risk": {"temperature": 0.0, "max_tokens": 2048},
        }
        return params.get(task_type, params["sentiment"])
