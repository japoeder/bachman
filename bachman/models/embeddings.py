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
from typing import List, Union
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
import torch

# import platform
# import os

from quantum_trade_utilities.data.load_credentials import load_credentials
from quantum_trade_utilities.core.get_path import get_path
from bachman.utils.gpu import check_gpu_availability, get_system_info

logger = logging.getLogger(__name__)


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


class HuggingFaceEmbeddingsWithCleanup(HuggingFaceEmbeddings):
    """Extended HuggingFace embeddings with memory cleanup."""

    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        device_type, _ = check_gpu_availability()
        if device_type in ["cuda", "mps"]:
            if device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
            elif device_type == "mps":
                torch.mps.empty_cache()
            logger.debug(f"Cleared {device_type.upper()} memory cache")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with memory cleanup."""
        try:
            embeddings = super().embed_documents(texts)
            self.clear_gpu_memory()
            return embeddings
        except Exception as e:
            logger.error(f"Error in embed_documents: {str(e)}")
            self.clear_gpu_memory()  # Clean up even on error
            raise

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query text with memory cleanup."""
        try:
            embedding = super().embed_query(text)
            self.clear_gpu_memory()
            return embedding
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            self.clear_gpu_memory()  # Clean up even on error
            raise


def get_embeddings(
    provider: str = "huggingface", **kwargs
) -> Union[HuggingFaceEmbeddingsWithCleanup, GroqEmbeddings]:
    """Initialize and return embeddings model."""
    try:
        if provider == "groq":
            logger.info("Initializing Groq embeddings model")
            return GroqEmbeddings(**kwargs)

        # Get system info and device
        system_info = get_system_info()
        logger.info(f"System information: {system_info}")

        device_type, device_name = check_gpu_availability()
        logger.info(f"Using device: {device_name} ({device_type})")

        model_kwargs = {"device": device_type, "trust_remote_code": True}
        encode_kwargs = {
            "device": device_type,
            "normalize_embeddings": True,
            "batch_size": 16,
        }

        model_kwargs.update(kwargs.get("model_kwargs", {}))
        encode_kwargs.update(kwargs.get("encode_kwargs", {}))

        logger.info("Initializing HuggingFace embeddings")
        logger.info(f"Model kwargs: {model_kwargs}")
        logger.info(f"Encode kwargs: {encode_kwargs}")

        embeddings = HuggingFaceEmbeddingsWithCleanup(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        return embeddings

    except Exception as e:
        logger.error(f"Error initializing embeddings model: {str(e)}")
        raise


def get_embeddings_info(
    embeddings: Union[HuggingFaceEmbeddingsWithCleanup, GroqEmbeddings]
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
        device_type, device_name = check_gpu_availability()
        return {
            "provider": "huggingface",
            "model_name": embeddings.model_name,
            "normalize": embeddings.encode_kwargs.get("normalize_embeddings", False),
            "device": device_name,
            "device_type": device_type,
            "model_type": "HuggingFace BGE",
        }


def get_bge_embeddings() -> HuggingFaceEmbeddings:
    """Initialize BGE embeddings model."""
    logger.info("Initializing HuggingFace BGE embeddings model")
    device_type, device_name = check_gpu_availability()
    logger.info(f"Using device: {device_name} ({device_type})")

    model_kwargs = {"device": device_type, "trust_remote_code": True}
    encode_kwargs = {
        "device": device_type,
        "normalize_embeddings": True,
        "batch_size": 32,
    }

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings
