"""File processing module for handling different document types."""

import os
import logging
from typing import Optional, Dict, List
import tabula
import pymupdf

# from langchain.document_loaders import PyPDFLoader, TextLoader
from bachman.processors.chunking import ChunkingConfig

# from bachman.processors.types import ChunkInfo, HashInfo, ProcessingResult
from bachman.processors.document_types import DocumentType

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Handles processing of different file types with configurable strategies.

    Supports PDF and text files with customizable chunking and metadata options.
    Provides tracking of processed documents and their chunks.
    """

    def __init__(self, text_processor):
        """Initialize with text processor dependency."""
        self.text_processor = text_processor
        self._document_processors = {
            DocumentType.FINANCIAL_STATEMENT: self._process_financial_statement,
            DocumentType.EARNINGS_CALL: self._process_earnings_call,
            DocumentType.GENERIC: self._process_generic,
        }

    async def process_file(
        self,
        file_path: str,
        collection_name: str,
        document_type: DocumentType = DocumentType.GENERIC,
        process_sentiment: bool = True,
        skip_if_exists: bool = False,
        chunking_config: Optional[ChunkingConfig] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Process file based on document type."""
        try:
            # Validate file exists and is accessible
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Validate file type
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in [".pdf", ".txt"]:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Get appropriate processor for document type
            processor = self._document_processors.get(
                document_type, self._process_generic
            )

            # Process document using type-specific processor
            result = await processor(
                file_path=file_path, chunking_config=chunking_config, metadata=metadata
            )

            # Add to vector store with document type metadata
            storage_result = await self.text_processor.process_text(
                text=result["processed_text"],
                collection_name=collection_name,
                metadata={
                    **(metadata or {}),
                    "document_type": document_type.value,
                    "sections": result.get("sections", {}),
                    "file_path": file_path,
                    "tables": result.get("tables", []),  # Store table info in metadata
                },
                skip_if_exists=skip_if_exists,
                chunking_config=chunking_config,
            )

            return {
                "status": "success",
                "document_type": document_type.value,
                "processing_result": result,
                "storage": storage_result,
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def _extract_tables_from_pdf(self, file_path: str, page_num: int) -> List[Dict]:
        """Extract tables from PDF page using tabula."""
        tables = []
        try:
            pdf_tables = tabula.read_pdf(
                file_path,
                pages=page_num + 1,
                multiple_tables=True,
                force_subprocess=True,
            )

            if pdf_tables:
                for table_idx, table in enumerate(pdf_tables):
                    # Convert table to string format
                    table_text = "\n".join(
                        [" | ".join(map(str, row)) for row in table.values]
                    )
                    tables.append(
                        {"page": page_num, "index": table_idx, "content": table_text}
                    )

        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {str(e)}")

        return tables

    def _process_pdf_document(self, file_path: str) -> Dict:
        """Process PDF document with table extraction."""
        doc = pymupdf.open(file_path)
        all_text = []
        all_tables = []

        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            all_text.append(text)

            # Extract tables
            tables = self._extract_tables_from_pdf(file_path, page_num)
            all_tables.extend(tables)

        return {"text": "\n\n".join(all_text), "tables": all_tables}
