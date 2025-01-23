"""
Text splitter utility module.

This module provides functionality for splitting text into chunks while preserving
semantic meaning and context. It includes custom splitting logic for handling
various document formats and maintaining coherent text segments.
"""

import logging
from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import numpy as np

logger = logging.getLogger(__name__)


def get_text_splitter(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    splitter_type: str = "recursive",
    separators: Optional[List[str]] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Initialize and return a text splitter.

    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        splitter_type: Type of splitter ("recursive" or "character")
        separators: Custom separators for recursive splitter

    Returns:
        Configured text splitter
    """
    try:
        default_separators = ["\n\n", "\n", ". ", " ", ""]

        if splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=separators or default_separators,
                keep_separator=True,
            )
        elif splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separator="\n",
            )
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

    except Exception as e:
        logger.error(f"Error initializing text splitter: {str(e)}")
        raise


def analyze_chunks(
    text: str, splitter: RecursiveCharacterTextSplitter, include_samples: bool = False
) -> Dict[str, Any]:
    """
    Analyze text chunks created by the splitter.

    Args:
        text: Text to analyze
        splitter: Configured text splitter
        include_samples: Whether to include sample chunks in output

    Returns:
        Dictionary with chunk statistics
    """
    try:
        chunks = splitter.split_text(text)
        chunk_lengths = [len(chunk) for chunk in chunks]

        stats = {
            "num_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks) if chunks else 0,
            "max_chunk_length": max(chunk_lengths) if chunks else 0,
            "min_chunk_length": min(chunk_lengths) if chunks else 0,
            "total_length": sum(chunk_lengths),
            "std_dev": float(np.std(chunk_lengths)) if chunks else 0,
            "config": {
                "chunk_size": splitter.chunk_size,
                "chunk_overlap": splitter.chunk_overlap,
                "separators": splitter.separators,
            },
        }

        if include_samples:
            stats["samples"] = {
                "shortest": min(chunks, key=len) if chunks else "",
                "longest": max(chunks, key=len) if chunks else "",
                "median": chunks[len(chunks) // 2] if chunks else "",
            }

        return stats

    except Exception as e:
        logger.error(f"Error analyzing chunks: {str(e)}")
        raise


def get_optimal_chunk_size(
    text: str,
    target_chunks: int = 10,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 100,
) -> Dict[str, Any]:
    """
    Calculate optimal chunk size to achieve target number of chunks.

    Args:
        text: Text to analyze
        target_chunks: Desired number of chunks
        max_chunk_size: Maximum allowed chunk size
        min_chunk_size: Minimum allowed chunk size

    Returns:
        Dictionary with recommended settings
    """
    try:
        text_length = len(text)
        optimal_size = max(
            min_chunk_size, min(max_chunk_size, text_length // target_chunks)
        )
        overlap = min(optimal_size // 4, 100)  # 25% overlap, max 100 chars

        return {
            "chunk_size": optimal_size,
            "chunk_overlap": overlap,
            "estimated_chunks": text_length // (optimal_size - overlap),
            "text_length": text_length,
        }

    except Exception as e:
        logger.error(f"Error calculating optimal chunk size: {str(e)}")
        raise


def split_by_section(
    text: str,
    section_markers: Optional[List[str]] = None,
    min_section_length: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split text into sections based on markers.

    Args:
        text: Text to split
        section_markers: List of section headings to split on
        min_section_length: Minimum length for valid section

    Returns:
        List of dictionaries containing section info
    """
    try:
        default_markers = [
            "Introduction",
            "Background",
            "Methods",
            "Results",
            "Discussion",
            "Conclusion",
            "Financial Statements",
            "Risk Factors",
        ]

        markers = section_markers or default_markers
        sections = []
        current_section = {"title": "Header", "content": ""}

        for line in text.split("\n"):
            marker_found = False
            for marker in markers:
                if marker.lower() in line.lower():
                    if (
                        current_section["content"]
                        and len(current_section["content"]) >= min_section_length
                    ):
                        sections.append(current_section)
                    current_section = {"title": line.strip(), "content": ""}
                    marker_found = True
                    break

            if not marker_found:
                current_section["content"] += line + "\n"

        if (
            current_section["content"]
            and len(current_section["content"]) >= min_section_length
        ):
            sections.append(current_section)

        return [
            {
                "title": section["title"],
                "content": section["content"].strip(),
                "length": len(section["content"]),
                "position": i,
            }
            for i, section in enumerate(sections)
        ]

    except Exception as e:
        logger.error(f"Error splitting sections: {str(e)}")
        raise
