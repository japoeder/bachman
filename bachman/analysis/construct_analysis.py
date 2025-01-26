"""
Analysis construction module for document processing.

This module defines the document and analysis types, along with their corresponding
instruction templates for the LLM. It provides a structured way to construct
analysis prompts based on document type and desired analysis focus.
"""

from enum import Enum
from typing import Dict, Tuple  # , Optional


class DocumentType(Enum):
    """
    Enum for document types.
    """

    QUARTERLY_REPORT = "quarterly_report"
    ANNUAL_REPORT = "annual_report"
    EARNINGS_CALL = "earnings_call"
    PRESS_RELEASE = "press_release"
    NEWS_ARTICLE = "news_article"
    GENERIC = "generic"


class AnalysisType(Enum):
    """
    Enum for analysis types.
    """

    SENTIMENT = "sentiment"
    RISK = "risk"
    COMPETITIVE = "competitive"
    FINANCIAL = "financial"
    STRATEGIC = "strategic"


def get_analysis_instructions(
    doc_type: DocumentType, analysis_type: AnalysisType
) -> str:
    """
    Get specific instructions for the LLM based on document and analysis types.

    Args:
        doc_type: Type of document being analyzed
        analysis_type: Type of analysis to perform

    Returns:
        Structured instruction template for the LLM
    """
    instructions: Dict[Tuple[DocumentType, AnalysisType], str] = {
        (
            DocumentType.QUARTERLY_REPORT,
            AnalysisType.SENTIMENT,
        ): """
            Analyze this quarterly report with focus on:
            - Overall sentiment and tone
            - Key financial metrics and their implications
            - Management's forward-looking statements
            - Changes from previous quarter
            Format the response with clear sections for each aspect.
        """,
        (
            DocumentType.QUARTERLY_REPORT,
            AnalysisType.FINANCIAL,
        ): """
            Analyze this quarterly report with focus on:
            - Key financial metrics and ratios
            - Revenue and profit trends
            - Cash flow analysis
            - Balance sheet health
            Provide quantitative analysis where possible.
        """,
        # Add more combinations as needed
        (
            DocumentType.GENERIC,
            AnalysisType.SENTIMENT,
        ): """
            Provide a general sentiment analysis focusing on:
            - Overall tone
            - Key themes
            - Notable statements
            - Potential implications
        """,
    }

    # Get specific instructions or fall back to generic
    return instructions.get(
        (doc_type, analysis_type),
        instructions.get((DocumentType.GENERIC, AnalysisType.SENTIMENT)),
    )
