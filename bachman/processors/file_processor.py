"""File processing module for handling different document types."""

import logging
from typing import Optional, Dict, List
import uuid
from datetime import datetime
import tabula
import pymupdf
import PyPDF2

# from langchain.document_loaders import PyPDFLoader, TextLoader
# from bachman.processors.chunking import ChunkingConfig

# from bachman.processors.types import ChunkInfo, HashInfo, ProcessingResult
from bachman.processors.document_types import DocumentType
from bachman.core.interfaces import TaskTracker
from bachman.processors.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class ProcessingStatus:
    """Status tracking for document processing."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.status = "pending"
        self.components = {
            "text": {"status": "pending", "pages": []},
            "tables": {"status": "pending", "pages": []},
            "metadata": {"status": "pending"},
        }
        self.errors = []
        self.last_updated = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        """Convert status to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "status": self.status,
            "components": self.components,
            "errors": self.errors,
            "last_updated": self.last_updated,
        }

    def update(self, component: str, status: str, error: Optional[str] = None):
        """Update component status."""
        if component in self.components:
            self.components[component]["status"] = status
        if error:
            self.errors.append(error)
        self.last_updated = datetime.utcnow().isoformat()


class FileProcessor:
    """
    Handles processing of different file types with configurable strategies.

    Supports PDF and text files with customizable chunking and metadata options.
    Provides tracking of processed documents and their chunks.
    """

    def __init__(
        self, text_processor, task_tracker: TaskTracker, vector_store: VectorStore
    ):
        """Initialize with required dependencies."""
        self.text_processor = text_processor
        self.task_tracker = task_tracker
        self.vector_store = vector_store

    def read_pdf(self, file_path: str) -> tuple[str, int]:
        """Read text content from a PDF file and return text and word count."""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                word_count = len(text.split())
                return text, word_count
        except Exception as e:
            logger.error(f"Error reading PDF file: {str(e)}")
            raise

    async def process_file(
        self,
        file_path: str,
        collection_name: str,
        doc_type: str,
        process_sentiment: bool = False,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Process a file and store its contents."""
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())

            # Initialize processing status
            status = ProcessingStatus(doc_id)
            await self.task_tracker.update_status(doc_id, status.to_dict())

            # Update metadata
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "file_path": file_path,
                    "processed_at": datetime.utcnow().isoformat(),
                }
            )

            # Ensure collection exists
            logger.info(f"Ensuring collection {collection_name} exists")
            await self.vector_store.ensure_collection(collection_name)

            # Read file content
            try:
                if file_path.lower().endswith(".pdf"):
                    text_content, word_count = self.read_pdf(file_path)
                    metadata["word_count"] = word_count
                else:
                    with open(file_path, "r", encoding="utf-8") as file:
                        text_content = file.read()
                        metadata["word_count"] = len(text_content.split())

                # Process text content
                text_result = await self.text_processor.process_text(
                    text=text_content,
                    collection_name=collection_name,
                    metadata=metadata,
                )
                status.update("text", "completed")
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                status.update("text", "failed", str(e))
                raise

            # Update final status
            if any(comp["status"] == "failed" for comp in status.components.values()):
                status.status = "failed"
            else:
                status.status = "completed"

            await self.task_tracker.update_status(doc_id, status.to_dict())

            return {
                "status": status.status,
                "doc_id": doc_id,
                "collection": collection_name,
                "metadata": metadata,
                "text_result": text_result,
            }

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            if "status" in locals():
                status.status = "failed"
                status.errors.append(str(e))
                await self.task_tracker.update_status(doc_id, status.to_dict())
            raise

    async def _process_text_components(
        self, file_path: str, status: ProcessingStatus
    ) -> dict:
        """Process text components of the file."""
        try:
            # Read text content and get word count
            text_content, word_count = self.read_pdf(file_path)

            status.update("text", "completed")
            return {"text": text_content, "word_count": word_count}

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            status.update("text", "failed", str(e))
            return {"text": "", "word_count": 0}

    async def _process_table_components(
        self, file_path: str, status: ProcessingStatus
    ) -> dict:
        """Process table components of the file."""
        try:
            # Read PDF document
            pdf = pymupdf.open(file_path)
            tables = []

            # Process each page
            for page_num in range(len(pdf)):
                page_tables = self._extract_tables_from_pdf(file_path, page_num)
                if page_tables:
                    tables.extend(page_tables)
                    status.components["tables"]["pages"].append(
                        {"page": page_num, "count": len(page_tables)}
                    )

            status.update("tables", "completed")
            return {"tables": tables}

        except Exception as e:
            logger.error(f"Error processing tables: {str(e)}")
            status.update("tables", "failed", str(e))
            return {"tables": []}

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
                pages=page_num + 1,  # tabula uses 1-based page numbers
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
