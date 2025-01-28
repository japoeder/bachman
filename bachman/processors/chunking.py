"""
Configuration module for text chunking strategies.

This module provides configuration classes and enums for controlling how text
is split into chunks for processing, with optimized defaults for different models.
"""

from enum import Enum
from typing import Optional, List
import logging
import json
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Enumeration of available text chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"


class ChunkingConfig(BaseModel):
    """
    Configuration for text chunking with Groq-optimized defaults.

    Groq context window is 32k tokens.
    Using conservative defaults to ensure:
    - Chunks are small enough for efficient processing
    - Overlap helps maintain context
    - Stays well within Groq's limits even with system prompts

    Attributes:
        strategy: The chunking strategy to use
        chunk_size: Maximum size of each chunk in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        separators: Separators to use for recursive splitting
        min_chunk_size: Minimum size of each chunk in tokens
    """

    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.RECURSIVE,
        description="Strategy to use for chunking text",
    )
    chunk_size: int = Field(
        default=4096,  # ~1/8 of Groq's context window
        description="Maximum size of each chunk in tokens",
    )
    chunk_overlap: int = Field(
        default=200,  # Reasonable overlap for context preservation
        description="Number of overlapping tokens between chunks",
    )
    separators: Optional[List[str]] = Field(
        default=["\\n\\n", "\\n", "\\. ", " ", ""],
        description="Separators to use for recursive splitting, in order of preference",
    )
    min_chunk_size: Optional[int] = Field(
        default=100,  # Minimum chunk size to avoid too small segments
        description="Minimum size of each chunk in tokens",
    )


def get_chunking_config(config_store: dict, doc_type: str) -> dict:
    """Get chunking configuration for a specific document type."""
    if config_store and doc_type in config_store:
        logger.info(f"Using document-specific chunking config for {doc_type}")
        logger.debug(f"Config details: {json.dumps(config_store[doc_type], indent=2)}")
        return config_store[doc_type]

    logger.warning(f"No specific chunking config found for {doc_type}, using default")
    default_config = {
        "strategy": "recursive",
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "separators": ["\n\n", "\n", ". ", " ", ""],
        "min_chunk_size": 50,
    }
    logger.debug(f"Default config details: {json.dumps(default_config, indent=2)}")
    return default_config
