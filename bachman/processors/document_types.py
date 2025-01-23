"""
Document type definitions and processing strategies.

This module defines the supported document types and their associated
processing configurations for the Bachman API.
"""

from enum import Enum
from typing import Dict
from pydantic import BaseModel


class DocumentType(Enum):
    """
    Enumeration of supported document types for processing.

    Each type may have specific processing rules and strategies.
    """

    FINANCIAL_STATEMENT = "financial_statement"
    EARNINGS_CALL = "earnings_call"
    RESEARCH_REPORT = "research_report"
    NEWS_ARTICLE = "news_article"
    GENERIC = "generic"
    # Add more as needed


class ProcessingStrategy(BaseModel):
    """Configuration for how different document types should be processed"""

    section_markers: Dict[str, str]  # Markers to identify sections
    important_metrics: list[str]  # Key metrics to track
    structure_template: Dict  # Output structure template
    chunking_rules: Dict  # Special chunking rules for this doc type
