"""
Type definitions for the Bachman API processors.

This module contains custom type definitions and type aliases used throughout
the Bachman processor modules for improved type safety and code clarity.
"""

from typing import Dict, List, TypedDict  # Optional, Union


class ChunkInfo(TypedDict):
    """Type definition for chunk information."""

    chunk_id: str
    parent_id: str
    sequence: int


class HashInfo(TypedDict):
    """Type definition for document hash information."""

    parent_id: str
    chunks: List[ChunkInfo]


class ProcessingResult(TypedDict):
    """Type definition for text processing results."""

    hash_info: HashInfo
    processing_info: Dict
