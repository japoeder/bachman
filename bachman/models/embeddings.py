"""
Embeddings module for text vectorization.

This module provides functionality for generating text embeddings using various providers:
- HuggingFace local models
- Groq API embeddings

The module automatically handles device selection (CPU/GPU/MPS) and provides a unified
interface for embedding generation across different backends.

Example:
    embeddings = get_embeddings(provider="groq")
    vector = embeddings.embed_query("Some text to embed")

"""

import logging
import platform
from typing import List, Union
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
import torch

from quantum_trade_utilities.data.load_credentials import load_credentials
from quantum_trade_utilities.core.get_path import get_path

logger = logging.getLogger(__name__)


def get_default_device() -> str:
    """Determine the default device based on system architecture."""
    system = platform.system()
    processor = platform.processor()

    if system == "Darwin" and "arm" in processor.lower():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    else:
        return "cpu"


class GroqEmbeddings(Embeddings):
    """Groq API embeddings wrapper."""

    def __init__(self, model_name: str = "mixtral-8x7b-32768", batch_size: int = 8):
        """Initialize Groq embeddings."""
        self.model_name = model_name
        self.batch_size = batch_size

        try:
            # Load Groq API key from nested structure
            # Initialize PRAW with credentials
            self.groq_api_key = load_credentials(get_path("creds"), "groq_api")

            # Initialize Groq client
            self.client = Groq(api_key=self.groq_api_key)
            logger.info(f"Initialized Groq client with model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            # Use chat completions API
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": text}], model=self.model_name
            )
            # Extract the embedding from the response
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting Groq response: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = [self._get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query text."""
        return self._get_embedding(text)


def get_embeddings(
    provider: str = "huggingface", **kwargs
) -> Union[HuggingFaceEmbeddings, GroqEmbeddings]:
    """Initialize and return embeddings model.

    Args:
        provider: The embeddings provider to use ('huggingface' or 'groq')
        **kwargs: Additional arguments passed to the embeddings model

    Returns:
        An initialized embeddings model
    """
    try:
        if provider == "groq":
            logger.info("Initializing Groq embeddings model")
            return GroqEmbeddings(**kwargs)

        logger.info("Initializing HuggingFace BGE embeddings model")
        model_kwargs = {"device": get_default_device()}
        encode_kwargs = {"normalize_embeddings": True}

        # Override defaults with any provided kwargs
        model_kwargs.update(kwargs.get("model_kwargs", {}))
        encode_kwargs.update(kwargs.get("encode_kwargs", {}))

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        logger.info("Successfully initialized embeddings model")
        return embeddings

    except Exception as e:
        logger.error(f"Error initializing embeddings model: {str(e)}")
        raise


def get_embeddings_info(
    embeddings: Union[HuggingFaceEmbeddings, GroqEmbeddings]
) -> dict:
    """
    Get information about the configured embeddings model.

    Args:
        embeddings: Embeddings instance

    Returns:
        Dictionary containing model information
    """
    if isinstance(embeddings, GroqEmbeddings):
        return {
            "provider": "groq",
            "model_name": embeddings.model_name,
            "batch_size": embeddings.batch_size,
        }
    else:
        return {
            "provider": "huggingface",
            "model_name": embeddings.model_name,
            "normalize": embeddings.encode_kwargs.get("normalize_embeddings", False),
            "device": embeddings.model_kwargs.get("device", "cpu"),
            "model_type": "HuggingFace BGE",
        }


def get_bge_embeddings() -> HuggingFaceEmbeddings:
    """Initialize BGE embeddings model."""
    logger.info("Initializing HuggingFace BGE embeddings model")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings
