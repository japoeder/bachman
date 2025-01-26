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
        """Process file based on file type and apply document-specific processing."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Get file extension and choose appropriate processor
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == ".pdf":
                result = await self._process_pdf(file_path)
            elif file_ext == ".txt":
                result = await self._process_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Apply document-type-specific processing
            processed_result = self._apply_document_type_processing(
                result, document_type
            )

            # Add to vector store with document type metadata
            storage_result = await self.text_processor.process_text(
                text=processed_result["processed_text"],
                collection_name=collection_name,
                metadata={
                    **(metadata or {}),
                    "document_type": document_type.value,
                    "sections": processed_result.get("sections", {}),
                    "file_path": file_path,
                    "tables": processed_result.get("tables", []),
                },
                skip_if_exists=skip_if_exists,
                chunking_config=chunking_config,
            )

            return {
                "status": "success",
                "document_type": document_type.value,
                "processing_result": processed_result,
                "storage": storage_result,
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def _process_pdf(self, file_path: str) -> Dict:
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

        return {
            "text": "\n\n".join(all_text),
            "tables": all_tables,
            "raw_content": all_text,  # Keep raw content for type-specific processing
        }

    async def _process_text_file(self, file_path: str) -> Dict:
        """Process text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "text": content,
            "tables": [],
            "raw_content": content,
        }

    def _apply_document_type_processing(
        self, result: Dict, document_type: DocumentType
    ) -> Dict:
        """Apply document-type-specific processing to extracted content."""
        if document_type == DocumentType.FINANCIAL_STATEMENT:
            return self._process_as_financial_statement(result)
        elif document_type == DocumentType.EARNINGS_CALL:
            return self._process_as_earnings_call(result)
        else:
            return self._process_as_generic(result)

    def _process_as_financial_statement(self, result: Dict) -> Dict:
        """Process content as a financial statement."""
        sections = {
            "balance_sheet": [],
            "income_statement": [],
            "cash_flow": [],
            "notes": [],
        }

        # Process tables and identify financial sections
        processed_tables = []
        for table in result["tables"]:
            table_type = self._identify_financial_table_type(table["content"])
            if table_type:
                processed_tables.append({**table, "type": table_type})
                sections[table_type].append(table["content"])

        return {
            "processed_text": result["text"],
            "tables": processed_tables,
            "sections": sections,
        }

    def _process_as_earnings_call(self, result: Dict) -> Dict:
        """Process content as an earnings call transcript."""
        sections = {
            "introduction": [],
            "prepared_remarks": [],
            "qa_session": [],
            "closing_remarks": [],
        }

        # Identify speakers and segment Q&A
        # content = result["raw_content"]
        # Add logic to segment earnings call sections

        return {
            "processed_text": result["text"],
            "tables": result["tables"],
            "sections": sections,
            "speakers": [],  # Add speaker identification
        }

    def _process_as_generic(self, result: Dict) -> Dict:
        """Process content as a generic document."""
        return {
            "processed_text": result["text"],
            "tables": result["tables"],
            "sections": {},
        }

    def _identify_financial_table_type(self, content: str) -> Optional[str]:
        """Identify type of financial table."""
        content_lower = content.lower()
        if "balance sheet" in content_lower:
            return "balance_sheet"
        elif "income statement" in content_lower:
            return "income_statement"
        elif "cash flow" in content_lower:
            return "cash_flow"
        return None

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
